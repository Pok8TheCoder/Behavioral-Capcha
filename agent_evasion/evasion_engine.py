"""Closed-loop red-team engine: attack dashboard bots, harvest winners, retrain Xmodel."""

from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bots.dashboard_bots import DASHBOARD_BOTS
from challenge_server.generator import CHALLENGE_TYPES
from agent_evasion.evasion_store import (
    count_windows,
    episodes_for_dataset,
    events_from_loop_csv,
    export_failed_windows,
    list_probe_datasets,
    list_recent_episodes,
    save_episode,
)
from agent_evasion.replica_pool import (
    clone_replica,
    evaluate_event_windows,
    evaluate_events,
    load_replica,
    run_local_attempt,
)
from backend.model import BehavioralClassifier
from backend.xmodel_registry import get_xmodel_entry, next_ea_xmodel_id

API = os.getenv("BEHAVIORAL_API", "http://127.0.0.1:8010")
STATE_PATH = os.path.join(ROOT, "agent_evasion", "loop_state.json")
RECORD_HZ = 60.0
WIN_HC = 0.65
MARK_HC = 0.60
WINDOW_TARGET = int(os.getenv("EVASION_WINDOW_TARGET", "5000"))
EXPAND_N = int(os.getenv("EVASION_EXPAND_N", "100"))
STALL_SEC = int(os.getenv("EVASION_STALL_SEC", str(30 * 60)))
PROBE_ATTEMPTS = int(os.getenv("EVASION_PROBE_ATTEMPTS", "25"))
WORKER_MIN = 1
WORKER_MAX = 128
WORKER_SAFE = 64
WORKER_DEFAULT = max(1, int(os.getenv("EVASION_WORKERS", "32")))


def clamp_workers(n: Any) -> int:
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = WORKER_DEFAULT
    return max(WORKER_MIN, min(WORKER_MAX, value))


def clamp_window_target(n: Any) -> int:
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = WINDOW_TARGET
    return max(1, min(1_000_000, value))


def _read_loop_state() -> Optional[Dict[str, Any]]:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or _now(), timezone.utc).isoformat()


def api(method: str, path: str, payload: Optional[dict] = None, timeout: int = 90) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Origin": API,
            "Referer": API + "/",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc


def trail_from_events(events: List[Dict[str, Any]], max_pts: int = 140) -> List[Dict[str, float]]:
    moves = [
        e for e in events
        if "move" in str(e.get("type") or e.get("event_type") or "").lower()
    ]
    if not moves:
        return []
    if len(moves) <= max_pts:
        return [{"x": float(e["x"]), "y": float(e["y"])} for e in moves]
    step = len(moves) / max_pts
    return [
        {"x": float(moves[int(i * step)]["x"]), "y": float(moves[int(i * step)]["y"])}
        for i in range(max_pts)
    ]


class EvasionEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._workers: List[threading.Thread] = []
        self._replicas: Dict[str, BehavioralClassifier] = {}
        self._template: Optional[BehavioralClassifier] = None
        self._subs: List[queue.Queue] = []
        self._in_flight = 0
        self._hold = False
        self._run_id = 0
        self._worker_n = WORKER_DEFAULT
        self._assigned: Dict[str, int] = {}
        self._last_persist = 0.0
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self._train_lock = threading.Lock()
        self._auto_train_armed = True
        self._probe_replica = None
        self._probe_replica_id = ""
        self._last_dataset_probe: Optional[Dict[str, Any]] = None
        self.state: Dict[str, Any] = self._empty_state()
        saved = _read_loop_state()
        if saved:
            self._adopt_saved_state(saved)
            self._worker_n = clamp_workers(self.state.get("workers") or WORKER_DEFAULT)
            self._restore_fail_probe()

    def _empty_state(self) -> Dict[str, Any]:
        families = [
            {
                "id": b["id"],
                "label": b["label"],
                "group": b["group"],
                "best_hc": 0.0,
                "attempts": 0,
                "wins": 0,
                "marks": 0,
                "expanded": False,
                "phase": "idle",
                "expand_left": 0,
                "expand_done": 0,
                "workers": 0,
                "recent_ctypes": [],
            }
            for b in DASHBOARD_BOTS
        ]
        return {
            "running": False,
            "paused": False,
            "phase": "idle",
            "model_id": "",
            "iteration": 1,
            "windows_cycle": 0,
            "windows_target": WINDOW_TARGET,
            "windows_total": 0,
            "attempts": 0,
            "beats": 0,
            "marks": 0,
            "workers": WORKER_DEFAULT,
            "workers_per_family": max(1, WORKER_DEFAULT // max(1, len(DASHBOARD_BOTS))),
            "worker_safe": WORKER_SAFE,
            "worker_max": WORKER_MAX,
            "in_flight": 0,
            "last_win_at": None,
            "stall_sec": STALL_SEC,
            "started_at": None,
            "stop_reason": None,
            "families": families,
            "iterations": [],
            "last_attempt": None,
            "expand": None,
            "probe": None,
            "selected_model_id": "",
            "log_tail": [],
        }

    def _adopt_saved_state(self, saved: Dict[str, Any]) -> None:
        base = self._empty_state()
        families = []
        saved_by_id = {f.get("id"): f for f in (saved.get("families") or []) if isinstance(f, dict)}
        for fam in base["families"]:
            old = saved_by_id.get(fam["id"])
            if old:
                merged = {**fam, **old}
                merged.setdefault("recent_ctypes", [])
                merged.setdefault("expand_left", 0)
                merged.setdefault("expand_done", 0)
                merged.setdefault("workers", 0)
                families.append(merged)
            else:
                families.append(fam)
        for key, value in saved.items():
            if key != "families":
                base[key] = value
        base["families"] = families
        base.setdefault("probe", None)
        base.setdefault("dataset_probe", None)
        base["probe"] = None
        base.setdefault("windows_target", WINDOW_TARGET)
        base.setdefault("selected_model_id", base.get("model_id") or "")
        self.state = base

    def _window_target(self) -> int:
        return max(1, int(self.state.get("windows_target") or WINDOW_TARGET))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=400)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def emit(self, kind: str, payload: Optional[dict] = None, persist: bool = True, force: bool = False) -> None:
        msg = {"type": kind, "ts": _iso(), **(payload or {})}
        with self._lock:
            if kind == "log":
                tail = self.state.setdefault("log_tail", [])
                tail.append(msg.get("message", ""))
                self.state["log_tail"] = tail[-80:]
            self.state["in_flight"] = self._in_flight
            if persist:
                self._persist(force=force)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        self.emit("log", {"message": line})

    def _persist(self, force: bool = False) -> None:
        now = _now()
        if not force and (now - self._last_persist) < 0.45:
            return
        self._last_persist = now
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2)
        except OSError:
            pass

    def start(self, model_id: str, workers: Optional[int] = None) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Loop already running.")
        self.stop_flag.set()
        for worker in list(self._workers):
            worker.join(timeout=5)
        wanted = clamp_workers(workers if workers is not None else self._worker_n)
        with self._lock:
            self.stop_flag.clear()
            self.pause_flag.clear()
            self._hold = False
            self._in_flight = 0
            self._workers = []
            self._replicas = {}
            self._template = None
            self._assigned = {}
            self._worker_n = wanted
            self._run_id += 1
            run_id = self._run_id
            self.state = self._empty_state()
            self.state["running"] = True
            self.state["model_id"] = model_id
            self.state["started_at"] = _iso()
            self.state["phase"] = "loading"
            self.state["workers"] = wanted
            self.state["workers_per_family"] = max(1, wanted // max(1, len(DASHBOARD_BOTS)))
            self._thread = threading.Thread(target=self._run, args=(run_id,), daemon=True)
            self._thread.start()
        self.log(
            f"Loop started against {model_id} with {wanted} workers "
            f"(equal split across {len(DASHBOARD_BOTS)} families; extras go to expand)"
        )

    def set_workers(self, workers: int) -> int:
        wanted = clamp_workers(workers)
        with self._lock:
            running = bool(self.state.get("running")) and self._thread and self._thread.is_alive()
            old = int(self._worker_n)
            self._worker_n = wanted
            self.state["workers"] = wanted
            self.state["workers_per_family"] = max(1, wanted // max(1, len(DASHBOARD_BOTS)))
            run_id = self._run_id
            model_id = self.state.get("model_id") or ""
        if not running:
            self.log(f"Worker count set to {wanted}")
            self.emit("state", {"state": self.snapshot()}, force=True)
            return wanted
        if wanted > old:
            for slot in range(old, wanted):
                self._ensure_replica(slot, model_id)
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(run_id, slot),
                    daemon=True,
                    name=f"ea-w{slot}",
                )
                self._workers.append(thread)
                thread.start()
        self.log(f"Workers {old} -> {wanted}")
        self.emit("state", {"state": self.snapshot()}, force=True)
        return wanted

    def set_windows_target(self, target: int) -> int:
        wanted = clamp_window_target(target)
        with self._lock:
            cycle = int(self.state.get("windows_cycle") or 0)
            self.state["windows_target"] = wanted
            # Lowering the target below current harvest must not auto-retrain.
            self._auto_train_armed = wanted > cycle
        self.log(
            f"Window target set to {wanted} (cycle still {self.state.get('windows_cycle') or 0}"
            f"{'' if self._auto_train_armed else '; auto-retrain off until Train now or a higher target'})"
        )
        self.emit("state", {"state": self.snapshot()}, force=True)
        return wanted

    def select_model(self, model_id: str) -> str:
        wanted = (model_id or "").strip()
        if not wanted:
            raise RuntimeError("No model selected.")
        if not get_xmodel_entry(wanted):
            raise RuntimeError(f"Unknown model '{wanted}'.")
        with self._lock:
            running = bool(self.state.get("running")) and self._thread and self._thread.is_alive()
            live = self.state.get("model_id") or ""
            self.state["selected_model_id"] = wanted
        if running and live and live != wanted:
            self.log(f"Selected {wanted} for train/probe; live loop still attacking {live}")
            self.emit("state", {"state": self.snapshot()}, force=True)
            return wanted
        with self._lock:
            self.state["model_id"] = wanted
            self._replicas = {}
            self._template = None
        self._probe_replica = None
        self._probe_replica_id = ""
        try:
            self._activate(wanted)
        except Exception as exc:
            self.log(f"Dashboard activate skipped: {exc}")
        self.log(f"Target model set to {wanted}")
        self.emit("state", {"state": self.snapshot()}, force=True)
        return wanted

    def rehydrate(self) -> bool:
        """Resume a saved running cycle without resetting window/attempt counters."""
        if not self.state.get("running"):
            return False
        if self._thread and self._thread.is_alive():
            return False
        self.stop_flag.clear()
        if self.state.get("paused"):
            self.pause_flag.set()
        else:
            self.pause_flag.clear()
        self._hold = False
        self._assigned = {}
        self._run_id += 1
        run_id = self._run_id
        self._thread = threading.Thread(target=self._run, args=(run_id, True), daemon=True)
        self._thread.start()
        return True

    def train_now(self, parent_id: Optional[str] = None) -> Dict[str, Any]:
        if not self._train_lock.acquire(blocking=False):
            raise RuntimeError("A training run is already in progress.")
        try:
            with self._lock:
                windows = int(self.state.get("windows_cycle") or 0)
                running = bool(self.state.get("running")) and self._thread and self._thread.is_alive()
                paused = bool(self.state.get("paused"))
            if windows < 1:
                raise RuntimeError("No winning windows harvested yet.")
            if running and not paused:
                self.pause()
            if running:
                self._pause_workers()
            parent = (parent_id or self.state.get("model_id") or "").strip()
            new_id = self._retrain(parent_id=parent)
            if not new_id:
                raise RuntimeError("Retrain failed.")
            probe = self.probe_recent_episodes(2)
            if running:
                self.stop(f"trained {new_id} from {windows} windows")
            return {
                "model_id": new_id,
                "windows": windows,
                "probe": probe,
                "state": self.snapshot(),
            }
        finally:
            self._train_lock.release()

    def pause(self) -> None:
        self.pause_flag.set()
        with self._cond:
            self.state["paused"] = True
            self.state["phase"] = "paused"
            self._cond.notify_all()
        self.log("Paused")
        self.emit("state", {"state": self.snapshot()}, force=True)

    def resume(self) -> None:
        self.pause_flag.clear()
        with self._cond:
            self.state["paused"] = False
            if self.state["phase"] == "paused":
                self.state["phase"] = "attacking"
            self._cond.notify_all()
        self.log("Resumed")
        self.emit("state", {"state": self.snapshot()}, force=True)

    def stop(self, reason: str = "stopped by operator") -> None:
        self.stop_flag.set()
        self.pause_flag.clear()
        with self._cond:
            self._hold = False
            self.state["running"] = False
            self.state["phase"] = "stopped"
            self.state["stop_reason"] = reason
            self._cond.notify_all()
        self.log(f"STOP: {reason}")
        self.emit("stopped", {"reason": reason, "state": self.snapshot()}, force=True)

    def _wait_gate(self) -> None:
        with self._cond:
            while not self.stop_flag.is_set() and (self.pause_flag.is_set() or self._hold):
                self._cond.wait(timeout=0.2)

    def _wait_if_paused(self) -> None:
        self._wait_gate()

    def _family(self, bot_id: str) -> Dict[str, Any]:
        for fam in self.state["families"]:
            if fam["id"] == bot_id:
                return fam
        raise KeyError(bot_id)

    def _activate(self, model_id: str) -> None:
        out = api("POST", "/api/models/activate", {"model_name": model_id})
        with self._lock:
            self.state["model_id"] = out.get("active_model") or model_id
        self.log(f"Activated {out.get('active_model_label') or model_id}")

    def _replica(self, bot_id: str, slot: int = 0) -> BehavioralClassifier:
        keyed = self._replicas.get(str(slot))
        if keyed is not None:
            return keyed
        if self._replicas:
            return next(iter(self._replicas.values()))
        raise KeyError(f"No replica loaded (bot={bot_id})")

    def _ensure_replica(self, slot: int, model_id: str) -> BehavioralClassifier:
        key = str(slot)
        existing = self._replicas.get(key)
        if existing is not None:
            return existing
        if self._template is not None:
            try:
                replica = clone_replica(self._template)
            except Exception as exc:
                self.log(f"clone failed w{slot}: {exc}; loading fresh")
                replica = load_replica(model_id)
        else:
            replica = load_replica(model_id)
            self._template = replica
        self._replicas[key] = replica
        return replica

    def _attempt(self, bot_id: str, replica: BehavioralClassifier, challenge_type: str = "random") -> tuple:
        result, events = run_local_attempt(replica, bot_id, challenge_type, RECORD_HZ)
        hc = float(result["human_confidence"] or 0.0)
        result["human_confidence"] = hc
        result["win"] = hc >= WIN_HC
        result["mark"] = hc >= MARK_HC
        result["windows"] = count_windows(events) if events else int(result.get("windows") or 0)
        result["trail"] = trail_from_events(events)
        return result, events

    def _record_attempt(self, result: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
        hc = float(result["human_confidence"])
        saved = None
        tagged = None
        if hc >= WIN_HC:
            tagged = "win"
        elif hc >= MARK_HC:
            tagged = "mark"
        if tagged:
            saved = save_episode(
                events,
                bot_id=result["bot"],
                model_id=self.state["model_id"],
                iteration=int(self.state["iteration"]),
                challenge_type=result["challenge_type"],
                seed=result["seed"],
                human_confidence=hc,
                tagged=tagged,
            )
        with self._lock:
            fam = self._family(result["bot"])
            fam["attempts"] += 1
            fam["best_hc"] = max(float(fam["best_hc"]), hc)
            self.state["attempts"] += 1
            self.state["last_attempt"] = {k: v for k, v in result.items() if k != "trail"}
            if tagged == "win":
                wins = int((saved or {}).get("windows") or result.get("windows") or 0)
                self.state["windows_cycle"] += wins
                self.state["windows_total"] += wins
                self.state["beats"] += 1
                fam["wins"] += 1
                self.state["last_win_at"] = _now()
            elif tagged == "mark":
                fam["marks"] += 1
                self.state["marks"] += 1
            fam_attempts = int(fam["attempts"])
        status = "WIN" if hc >= WIN_HC else ("MARK" if hc >= MARK_HC else "miss")
        if hc >= MARK_HC or fam_attempts <= 3 or fam_attempts % 40 == 0:
            self.log(
                f"{status} {result['bot']:16} hc={hc:.1%}  "
                f"{result['challenge_type']:16}  windows={result['windows']}"
            )
        self.emit("attempt", {"attempt": result, "saved": saved}, persist=hc >= MARK_HC)

    def _stalled(self) -> bool:
        last = self.state.get("last_win_at")
        if last is None:
            return False
        return (_now() - float(last)) >= STALL_SEC

    def _queue_expand(self, bot_id: str) -> None:
        with self._lock:
            fam = self._family(bot_id)
            if fam.get("expanded"):
                return
            fam["expanded"] = True
            fam["phase"] = "expanding"
            fam["expand_left"] = EXPAND_N
            fam["expand_done"] = 0
            expanding = [
                f["id"] for f in self.state["families"]
                if f.get("phase") == "expanding" or int(f.get("expand_left") or 0) > 0
            ]
            self.state["expand"] = {"bot": bot_id, "done": 0, "total": EXPAND_N}
            self.state["expanding"] = expanding
        self.log(f"Expanding {bot_id} x {EXPAND_N} variants (extras will pile on)")
        self.emit("state", {"state": self.snapshot()}, force=True)

    def _note_expand_progress(self, bot_id: str) -> None:
        finished = False
        with self._lock:
            fam = self._family(bot_id)
            fam["expand_done"] = int(fam.get("expand_done") or 0) + 1
            done = int(fam["expand_done"])
            if self.state.get("expand") and self.state["expand"].get("bot") == bot_id:
                self.state["expand"]["done"] = done
            if done >= EXPAND_N and int(fam.get("expand_left") or 0) <= 0:
                fam["phase"] = "attacking"
                finished = True
                self.state["expanding"] = [
                    f["id"] for f in self.state["families"] if f.get("phase") == "expanding"
                ]
                if self.state.get("expand") and self.state["expand"].get("bot") == bot_id:
                    self.state["expand"] = None
        if finished:
            self.log(f"Expand finished for {bot_id}")

    def _refund_expand(self, bot_id: str) -> None:
        with self._lock:
            fam = self._family(bot_id)
            if fam.get("expanded"):
                fam["expand_left"] = int(fam.get("expand_left") or 0) + 1
                fam["phase"] = "expanding"

    def _desired_counts(self) -> Dict[str, int]:
        fams = [f["id"] for f in self.state["families"]]
        n = max(0, int(self._worker_n))
        count = len(fams) or 1
        base, extra = divmod(n, count)
        counts = {fid: base for fid in fams}
        if extra <= 0:
            return counts
        expanding = [
            f["id"] for f in self.state["families"]
            if f.get("phase") == "expanding" or int(f.get("expand_left") or 0) > 0
        ]
        targets = expanding if expanding else fams
        for i in range(extra):
            counts[targets[i % len(targets)]] += 1
        return counts

    def _claim_job(self) -> Optional[tuple]:
        with self._lock:
            desired = self._desired_counts()
            best_id = None
            best_score = -10**9
            for fam in self.state["families"]:
                fid = fam["id"]
                need = int(desired.get(fid, 0)) - int(self._assigned.get(fid, 0))
                expanding = 1 if (fam.get("phase") == "expanding" or int(fam.get("expand_left") or 0) > 0) else 0
                score = need * 10 + expanding
                if score > best_score:
                    best_score = score
                    best_id = fid
            if best_id is None:
                return None
            fam = self._family(best_id)
            self._assigned[best_id] = int(self._assigned.get(best_id, 0)) + 1
            fam["workers"] = self._assigned[best_id]
            job = "attack"
            recent = list(fam.get("recent_ctypes") or [])
            pool = [c for c in CHALLENGE_TYPES if c not in recent[-4:]] or list(CHALLENGE_TYPES)
            ctype = random.choice(pool)
            left = int(fam.get("expand_left") or 0)
            if left > 0:
                issued = EXPAND_N - left
                fam["expand_left"] = left - 1
                job = "expand"
                ctype = CHALLENGE_TYPES[issued % len(CHALLENGE_TYPES)]
            return best_id, job, ctype

    def _release_family(self, bot_id: str) -> None:
        with self._lock:
            self._assigned[bot_id] = max(0, int(self._assigned.get(bot_id, 0)) - 1)
            try:
                self._family(bot_id)["workers"] = self._assigned[bot_id]
            except KeyError:
                pass

    def _expand_family(self, bot_id: str, replica: BehavioralClassifier) -> None:
        self._queue_expand(bot_id)

    def _retrain(self, parent_id: Optional[str] = None) -> Optional[str]:
        self.state["phase"] = "retraining"
        self.emit("state", {"state": self.snapshot()}, force=True)
        parent_id = (parent_id or self.state.get("model_id") or "").strip()
        new_id = self._invoke_trainer(parent_id, reset_cycle=True)
        if not new_id:
            return None
        probe = self.probe_recent_episodes(2)
        self.state["probe"] = probe
        self.emit("state", {"state": self.snapshot()}, force=True)
        return new_id

    def _probe_last_two(self) -> bool:
        """Return True if last 2 iteration winner families fail to beat the current model."""
        history = self.state.get("iterations") or []
        if len(history) < 2:
            return False
        families: List[str] = []
        for row in history[-2:]:
            for bot_id in row.get("winning_families") or []:
                if bot_id not in families:
                    families.append(bot_id)
        if not families:
            return False
        self.state["phase"] = "probing"
        self.log(f"Probe last-2 winner families: {', '.join(families)}")
        self.emit("state", {"state": self.snapshot()})
        any_win = False
        for bot_id in families:
            if self.stop_flag.is_set():
                return False
            best = 0.0
            for _ in range(PROBE_ATTEMPTS):
                while self.pause_flag.is_set() and not self.stop_flag.is_set():
                    time.sleep(0.2)
                if self.stop_flag.is_set():
                    return False
                try:
                    result, events = self._attempt(bot_id, self._replica(bot_id, 0), "random")
                except Exception as exc:
                    self.log(f"probe error {bot_id}: {exc}")
                    continue
                self._record_attempt(result, events)
                best = max(best, float(result["human_confidence"]))
                if result["win"]:
                    any_win = True
                    self.log(f"Probe still leaking: {bot_id} at {result['human_confidence']:.1%}")
                    break
            if not any_win:
                self.log(f"Probe held {bot_id} (best {best:.1%})")
        held = not any_win
        if held:
            self.log("Model holds against last 2 iteration winner families")
        return held

    def _probe_classifier(self, model_id: str) -> Any:
        wanted = (model_id or "").strip()
        if not wanted:
            raise RuntimeError("No model selected to probe.")
        if self._probe_replica is not None and self._probe_replica_id == wanted:
            return self._probe_replica
        replica = load_replica(wanted)
        self._probe_replica = replica
        self._probe_replica_id = wanted
        return replica

    def _resolve_probe_model(self, model_id: Optional[str] = None) -> str:
        return (
            (model_id or "").strip()
            or (self.state.get("selected_model_id") or "").strip()
            or (self.state.get("model_id") or "").strip()
        )

    def _score_episode_list(
        self,
        episodes: List[Dict[str, Any]],
        replica: Any,
        include_trails: bool = True,
        progress_every: int = 0,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        passed = 0
        failed = 0
        total = len(episodes)
        for index, entry in enumerate(episodes, start=1):
            if progress_every and index % progress_every == 0:
                self.log(f"Dataset probe {index}/{total}")
            rel = str(entry.get("csv") or "")
            path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel.replace("/", os.sep))
            item: Dict[str, Any] = {
                "filename": entry.get("filename") or os.path.basename(path),
                "bot_id": entry.get("bot_id"),
                "challenge_type": entry.get("challenge_type"),
                "original_hc": entry.get("human_confidence"),
                "original_windows": entry.get("windows"),
                "tagged": entry.get("tagged"),
                "path": rel,
                "passed": False,
                "human_confidence": None,
                "window_count": 0,
                "windows_passed": 0,
                "windows_failed": 0,
                "windows": [],
                "error": None,
            }
            try:
                events = events_from_loop_csv(path)
                scored = evaluate_event_windows(replica, events, win_hc=WIN_HC)
                window_rows = []
                best_hc = 0.0
                for window in scored:
                    hc = float(window["human_confidence"])
                    best_hc = max(best_hc, hc)
                    if window["passed"]:
                        passed += 1
                        item["windows_passed"] += 1
                    else:
                        failed += 1
                        item["windows_failed"] += 1
                    row = {
                        "index": window["index"],
                        "human_confidence": hc,
                        "passed": bool(window["passed"]),
                        "event_count": window["event_count"],
                        "duration_ms": window["duration_ms"],
                    }
                    if include_trails:
                        row["trail"] = trail_from_events(window.get("events") or [])
                    window_rows.append(row)
                item["windows"] = window_rows
                item["window_count"] = len(window_rows)
                item["human_confidence"] = round(best_hc, 4)
                item["passed"] = item["windows_passed"] > 0
            except Exception as exc:
                item["error"] = str(exc)
            rows.append(item)
        return {
            "tested": sum(int(r.get("window_count") or 0) for r in rows),
            "datasets": len(rows),
            "passed": passed,
            "failed": failed,
            "pass_means": "window still beats the detector (>=65% HC)",
            "episodes": rows,
        }

    def _store_probe(self, summary: Dict[str, Any], *, slot: str = "probe") -> Dict[str, Any]:
        slim = {k: v for k, v in summary.items() if k != "episodes"}
        slim["episodes"] = []
        if slot == "dataset_probe":
            slim["leaks"] = [
                {
                    "path": e.get("path"),
                    "windows_passed": int(e.get("windows_passed") or 0),
                    "filename": e.get("filename"),
                }
                for e in summary.get("episodes") or []
                if int(e.get("windows_passed") or 0) > 0 and e.get("path")
            ]
        self.state[slot] = slim
        return summary

    def _restore_fail_probe(self) -> None:
        saved = self.state.get("dataset_probe") or {}
        leaks = saved.get("leaks") or []
        if leaks:
            self._last_dataset_probe = {**saved, "episodes": leaks}

    def _fail_probe(self) -> Dict[str, Any]:
        probe = self._last_dataset_probe
        if probe and not probe.get("episodes") and probe.get("leaks"):
            probe = {**probe, "episodes": probe.get("leaks") or []}
        if probe and probe.get("episodes"):
            return probe
        saved = self.state.get("dataset_probe") or {}
        leaks = saved.get("leaks") or []
        if leaks:
            return {**saved, "episodes": leaks}
        raise RuntimeError("Test the entire dataset first so failed windows are known.")

    def probe_recent_episodes(self, count: int = 2, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Replay the last saved winner CSVs, scoring every window against the selected model."""
        episodes = list_recent_episodes(max(1, int(count)), tagged="win")
        if not episodes:
            raise RuntimeError("No saved winner episodes to probe.")
        wanted = self._resolve_probe_model(model_id)
        try:
            replica = self._probe_classifier(wanted)
        except Exception as exc:
            raise RuntimeError(f"No model available to probe: {exc}") from exc
        summary = self._score_episode_list(episodes, replica, include_trails=True)
        summary["model_id"] = wanted
        summary["dataset_id"] = "recent_wins"
        summary["dataset_label"] = f"Last {len(episodes)} winner episodes"
        self._store_probe(summary)
        self.log(
            f"Probe last {len(episodes)} winner datasets vs {wanted}: "
            f"{summary['passed']} windows still beat / {summary['failed']} caught "
            f"({summary['tested']} windows)"
        )
        self.emit("state", {"state": self.snapshot()}, force=True)
        return summary

    def list_datasets(self) -> List[Dict[str, Any]]:
        rows = list_probe_datasets()
        cycle_total = int(self.state.get("windows_total") or 0)
        for row in rows:
            if row.get("featured"):
                shown = cycle_total or int(row.get("windows") or 0)
                row["label"] = f"Cycle {row.get('iteration') or 1} harvest - {shown} windows"
                row["detail"] = (
                    f"{row.get('windows') or 0} stored windows / {row.get('episodes') or 0} episodes"
                    + (f" (cycle counted {cycle_total})" if cycle_total else "")
                )
        return rows

    def probe_dataset(self, dataset_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        episodes = episodes_for_dataset(dataset_id)
        if not episodes:
            raise RuntimeError(f"No episodes in dataset '{dataset_id}'.")
        wanted = self._resolve_probe_model(model_id)
        try:
            replica = self._probe_classifier(wanted)
        except Exception as exc:
            raise RuntimeError(f"No model available to probe: {exc}") from exc
        label = dataset_id
        for item in list_probe_datasets():
            if item.get("id") == dataset_id:
                label = item.get("label") or dataset_id
                break
        self.log(f"Testing full dataset {label} ({len(episodes)} episodes) vs {wanted}")
        summary = self._score_episode_list(
            episodes, replica, include_trails=False, progress_every=50,
        )
        summary["model_id"] = wanted
        summary["dataset_id"] = dataset_id
        summary["dataset_label"] = label
        self._last_dataset_probe = summary
        self._store_probe(summary, slot="dataset_probe")
        self.log(
            f"Dataset {label} vs {wanted}: "
            f"{summary['passed']} windows still beat / {summary['failed']} caught "
            f"({summary['tested']} windows)"
        )
        self.emit("state", {"state": self.snapshot()}, force=True)
        return summary

    def train_on_failed_windows(self, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrain the selected model on windows the last dataset test still failed to catch."""
        probe = self._fail_probe()
        leak_n = int(probe.get("passed") or 0)
        if leak_n < 1:
            leak_n = sum(int(e.get("windows_passed") or 0) for e in probe.get("episodes") or [])
        if leak_n < 1:
            raise RuntimeError("Last dataset test had no failed/leaking windows.")
        if not self._train_lock.acquire(blocking=False):
            raise RuntimeError("A training run is already in progress.")
        running = False
        paused = False
        try:
            exported = export_failed_windows(probe)
            with self._lock:
                running = bool(self.state.get("running")) and self._thread and self._thread.is_alive()
                paused = bool(self.state.get("paused"))
            if running and not paused:
                self.pause()
            if running:
                self._pause_workers()
            parent = (parent_id or self.state.get("selected_model_id") or self.state.get("model_id") or "").strip()
            self.state["phase"] = "retraining"
            self.emit("state", {"state": self.snapshot()}, force=True)
            self.log(
                f"Retraining {parent} on {exported['windows']} failed windows "
                f"({exported['episodes']} episodes) from {probe.get('dataset_id')}"
            )
            new_id = self._invoke_trainer(parent, reset_cycle=False)
            if not new_id:
                raise RuntimeError("Retrain on failed windows failed.")
            self._last_dataset_probe = None
            self.state["dataset_probe"] = None
            return {
                "model_id": new_id,
                "windows": exported["windows"],
                "episodes": exported["episodes"],
                "dataset_id": probe.get("dataset_id"),
                "state": self.snapshot(),
            }
        finally:
            if running:
                self._resume_workers()
                if paused:
                    self.pause()
                else:
                    self.resume()
            elif self.state.get("phase") == "retraining":
                self.state["phase"] = "stopped"
                self.emit("state", {"state": self.snapshot()}, force=True)
            self._train_lock.release()

    def _invoke_trainer(self, parent_id: str, *, reset_cycle: bool) -> Optional[str]:
        parent_id = (parent_id or "").strip()
        planned_id = next_ea_xmodel_id(parent_id)
        self.log(
            f"Retraining {planned_id} from {parent_id} "
            f"({self.state['windows_cycle']} winning windows)"
        )
        env = os.environ.copy()
        env["XMODEL_PARENT_ID"] = parent_id
        env["XMODEL_OUTPUT_ID"] = planned_id
        parent_entry = get_xmodel_entry(parent_id) or {}
        trainer = (
            os.path.join(ROOT, "train_xmodel_v03z.py")
            if str(parent_entry.get("feature_schema") or "") == "zmodel_seq"
            else os.path.join(ROOT, "train_xmodel_loop.py")
        )
        proc = subprocess.run(
            [sys.executable, trainer],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.log(f"Retrain failed: {proc.stderr[-800:] or proc.stdout[-800:]}")
            if reset_cycle:
                self.stop("retrain failed")
            return None
        new_id = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("TRAIN_COMPLETE "):
                new_id = line.split()[1]
        if not new_id:
            self.log("Retrain finished but TRAIN_COMPLETE line missing")
            if reset_cycle:
                self.stop("retrain produced no model id")
            return None
        self.log(f"Trained {new_id}")
        self.state["model_id"] = new_id
        self.state["selected_model_id"] = new_id
        if reset_cycle:
            winning = [f["id"] for f in self.state["families"] if f["wins"] > 0]
            self.state["iterations"].append({
                "from_model": parent_id,
                "to_model": new_id,
                "windows": self.state["windows_cycle"],
                "winning_families": winning,
            })
            self.state["windows_cycle"] = 0
            self.state["iteration"] = int(self.state["iteration"]) + 1
            for fam in self.state["families"]:
                fam["expanded"] = False
                fam["wins"] = 0
                fam["marks"] = 0
                fam["attempts"] = 0
                fam["best_hc"] = 0.0
                fam["phase"] = "attacking"
                fam["expand_left"] = 0
                fam["expand_done"] = 0
                fam["workers"] = 0
                fam["recent_ctypes"] = []
            self._auto_train_armed = True
        self._reload_replicas(new_id)
        try:
            self._activate(new_id)
        except Exception as exc:
            self.log(f"Dashboard activate skipped: {exc}")
        self.emit("state", {"state": self.snapshot()}, force=True)
        return new_id

    def _gated_attempt(self, bot_id: str, replica: BehavioralClassifier, challenge_type: str = "random"):
        with self._cond:
            while not self.stop_flag.is_set() and (self.pause_flag.is_set() or self._hold):
                self._cond.wait(timeout=0.2)
            if self.stop_flag.is_set() or self._hold:
                return None, None
            self._in_flight += 1
            self.state["in_flight"] = self._in_flight
        try:
            return self._attempt(bot_id, replica, challenge_type)
        finally:
            with self._cond:
                self._in_flight = max(0, self._in_flight - 1)
                self.state["in_flight"] = self._in_flight
                self._cond.notify_all()

    def _pause_workers(self) -> None:
        with self._cond:
            self._hold = True
            self.state["phase"] = "holding"
            while self._in_flight > 0 and not self.stop_flag.is_set():
                self._cond.wait(timeout=0.5)

    def _resume_workers(self) -> None:
        with self._cond:
            self._hold = False
            if self.state.get("phase") not in ("stopped", "paused"):
                self.state["phase"] = "attacking"
            self._cond.notify_all()

    def _load_replicas(self, model_id: str, count: Optional[int] = None) -> None:
        n = clamp_workers(count if count is not None else self._worker_n)
        self.log(f"Loading {n} model copies of {model_id}")
        template = load_replica(model_id)
        replicas: Dict[str, BehavioralClassifier] = {"0": template}
        for slot in range(1, n):
            if self.stop_flag.is_set():
                break
            try:
                replicas[str(slot)] = clone_replica(template)
            except Exception as exc:
                self.log(f"clone failed w{slot}: {exc}; loading fresh")
                replicas[str(slot)] = load_replica(model_id)
        self._template = template
        self._replicas = replicas
        self._worker_n = n
        with self._lock:
            self.state["workers"] = n
            self.state["workers_per_family"] = max(1, n // max(1, len(DASHBOARD_BOTS)))
        self.log(f"Loaded {len(replicas)} model copies of {model_id}")

    def _reload_replicas(self, model_id: str) -> None:
        n = clamp_workers(self._worker_n)
        self._load_replicas(model_id, n)

    def _worker_loop(self, run_id: int, slot: int) -> None:
        while not self.stop_flag.is_set() and run_id == self._run_id:
            replica = self._replicas.get(str(slot))
            if replica is None or slot >= int(self._worker_n):
                time.sleep(0.05)
                continue
            claimed = self._claim_job()
            if not claimed:
                time.sleep(0.05)
                continue
            bot_id, job, ctype = claimed
            try:
                result, events = self._gated_attempt(bot_id, replica, ctype)
                if result is None:
                    if job == "expand":
                        self._refund_expand(bot_id)
                    continue
                self._record_attempt(result, events)
                with self._lock:
                    fam = self._family(bot_id)
                    recent = list(fam.get("recent_ctypes") or [])
                    recent.append(str(result.get("challenge_type") or ctype))
                    fam["recent_ctypes"] = recent[-8:]
                if job == "expand":
                    self._note_expand_progress(bot_id)
                elif result.get("win"):
                    self._queue_expand(bot_id)
                self.emit("state", {"state": self.snapshot()}, persist=False)
            except Exception as exc:
                if job == "expand":
                    self._refund_expand(bot_id)
                self.log(f"attempt error {bot_id}/w{slot}: {exc}")
                time.sleep(0.4)
            finally:
                self._release_family(bot_id)

    def _run(self, run_id: int, restore: bool = False) -> None:
        try:
            model_id = self.state["model_id"]
            self._load_replicas(model_id, self._worker_n)
            if self.stop_flag.is_set() or run_id != self._run_id:
                return
            try:
                self._activate(model_id)
            except Exception as exc:
                self.log(f"Dashboard activate skipped: {exc}")
            with self._lock:
                if restore:
                    if not self.pause_flag.is_set():
                        self.state["phase"] = "attacking"
                    self.log(
                        f"Restored cycle at {self.state.get('windows_cycle')}/"
                        f"{self._window_target()} windows, "
                        f"{self.state.get('attempts')} attempts"
                        f"{' (paused - click Resume to keep farming)' if self.pause_flag.is_set() else ''}"
                    )
                else:
                    self.state["phase"] = "attacking"
                    self.state["last_win_at"] = _now()
                    for fam in self.state["families"]:
                        if fam.get("phase") in ("idle", "stopped", None):
                            fam["phase"] = "attacking"
            self.emit("state", {"state": self.snapshot()}, force=True)
            self._workers = []
            for slot in range(int(self._worker_n)):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(run_id, slot),
                    daemon=True,
                    name=f"ea-w{slot}",
                )
                self._workers.append(worker)
                worker.start()
            self.log(f"{len(self._workers)} workers firing (equal family split; extras -> expand)")
            while not self.stop_flag.is_set():
                time.sleep(0.25)
                if self.pause_flag.is_set():
                    continue
                if self._stalled():
                    self.stop(f"no >={WIN_HC:.0%} win for {STALL_SEC // 60} minutes")
                    return
                if self._auto_train_armed and int(self.state["windows_cycle"]) >= self._window_target():
                    self._pause_workers()
                    if self.stop_flag.is_set():
                        return
                    new_id = self._retrain()
                    if not new_id:
                        return
                    if self._probe_last_two():
                        self.stop("current model holds against bots from the last 2 trainings")
                        return
                    with self._lock:
                        self.state["phase"] = "attacking"
                        self.state["last_win_at"] = _now()
                    self._resume_workers()
                    self.emit("state", {"state": self.snapshot()}, force=True)
        except Exception as exc:
            self.log(f"engine crash: {exc}")
            self.stop(f"engine crash: {exc}")
        finally:
            if run_id != self._run_id:
                return
            self.stop_flag.set()
            with self._cond:
                self._hold = False
                self._cond.notify_all()
            for worker in list(self._workers):
                worker.join(timeout=2.0)


ENGINE = EvasionEngine()
