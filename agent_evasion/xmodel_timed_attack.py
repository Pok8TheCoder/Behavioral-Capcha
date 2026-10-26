"""20-minute timed red-team attack against Xmodel V0.1 via live dashboard API."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bots.dashboard_bots import DASHBOARD_BOTS

API = os.getenv("BEHAVIORAL_API", "http://127.0.0.1:8010")
MODEL = os.getenv("XMODEL_ATTACK_TARGET", "xmodel_v0.1")
TARGET_HC = float(os.getenv("XMODEL_ATTACK_TARGET_HC", "0.60"))
DURATION_SEC = int(os.getenv("XMODEL_ATTACK_DURATION_SEC", str(20 * 60)))
FAMILY_BUDGET_SEC = int(os.getenv("XMODEL_ATTACK_FAMILY_BUDGET_SEC", "90"))
ATTEMPT_CAP = int(os.getenv("XMODEL_ATTACK_ATTEMPT_CAP", "0"))  # 0 = no cap, time-budget only
RECORD_HZ = float(os.getenv("XMODEL_ATTACK_RECORD_HZ", "60"))

OUT_DIR = os.path.join(ROOT, "agent_evasion")
_model_tag = MODEL.replace(".", "_")
LOG_PATH = os.path.join(OUT_DIR, os.getenv("XMODEL_ATTACK_LOG", f"xmodel_attack_{_model_tag}_log.txt"))
RESULT_PATH = os.path.join(OUT_DIR, os.getenv("XMODEL_ATTACK_RESULT", f"xmodel_attack_{_model_tag}_result.json"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def api(method: str, path: str, payload: Optional[dict] = None) -> dict:
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc


def activate_model() -> None:
    out = api("POST", "/api/models/activate", {"model_name": MODEL})
    log(f"Activated {out.get('active_model_label') or out.get('active_model')} ({out.get('active_model')})")


def run_attempt(bot_id: str) -> Dict[str, Any]:
    ch = api("POST", "/api/challenge/new", {"challenge_type": "random"})
    out = api(
        "POST",
        "/api/challenge/run-bot",
        {"session_id": ch["session_id"], "bot": bot_id, "record_hz": RECORD_HZ},
    )
    hc = float(out.get("human_confidence") or 0.0)
    task = out.get("task_result") or {}
    ev = out.get("evaluation") or {}
    return {
        "bot": bot_id,
        "challenge_type": out.get("challenge_type") or ch.get("challenge_type"),
        "seed": ch.get("seed"),
        "human_confidence": hc,
        "task_passed": bool(task.get("passed")),
        "bot_probability": ev.get("bot_probability"),
        "decision": ev.get("decision"),
        "flags": ev.get("flags") or [],
        "event_count": out.get("event_count"),
    }


def save_state(state: dict) -> None:
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def main() -> None:
    started = time.time()
    deadline = started + DURATION_SEC
    families: List[Dict[str, str]] = [
        {"id": b["id"], "label": b["label"], "group": b["group"]} for b in DASHBOARD_BOTS
    ]

    log("=" * 72)
    log(f"XMODEL TIMED ATTACK — target={MODEL} hc>={TARGET_HC:.0%} duration={DURATION_SEC}s")
    log(f"Families: {len(families)}  per-family budget={FAMILY_BUDGET_SEC}s")
    log("=" * 72)

    activate_model()

    beaten: Dict[str, Any] = {}
    not_beaten: Dict[str, Any] = {}
    all_attempts: List[Dict[str, Any]] = []
    total_attempts = 0

    round_num = 0
    while time.time() < deadline:
        round_num += 1
        pending = [f for f in families if f["id"] not in beaten]
        if not pending:
            log(f"All {len(families)} families beaten — stopping early.")
            break

        log(f"=== Round {round_num}: {len(pending)} families remaining ===")
        for fam in pending:
            if time.time() >= deadline:
                log("Global deadline reached.")
                break

            bot_id = fam["id"]
            fam_start = time.time()
            fam_deadline = min(deadline, fam_start + FAMILY_BUDGET_SEC)
            prior = not_beaten.get(bot_id, {})
            best_hc = float(prior.get("best_hc", 0.0))
            attempts = int(prior.get("attempts", 0))
            round_attempts = 0
            log(f"--- Attacking family: {fam['label']} ({bot_id}) [{fam['group']}] ---")

            while time.time() < fam_deadline:
                attempts += 1
                round_attempts += 1
                total_attempts += 1
                try:
                    result = run_attempt(bot_id)
                except Exception as exc:
                    log(f"  attempt {attempts} ERROR: {exc}")
                    time.sleep(0.5)
                    continue

                hc = float(result["human_confidence"])
                best_hc = max(best_hc, hc)
                result["attempt"] = attempts
                result["elapsed_s"] = round(time.time() - started, 1)
                all_attempts.append(result)

                status = "BEAT" if hc >= TARGET_HC else "miss"
                if round_attempts <= 5 or hc >= TARGET_HC or round_attempts % 50 == 0:
                    log(
                        f"  #{attempts:4d} {status} hc={hc:.1%} task={'PASS' if result['task_passed'] else 'FAIL'} "
                        f"type={result['challenge_type']} events={result.get('event_count')} "
                        f"bot_prob={result.get('bot_probability')}"
                    )

                if hc >= TARGET_HC:
                    beaten[bot_id] = {
                        **fam,
                        "best_hc": hc,
                        "attempts_to_beat": attempts,
                        "winning_attempt": result,
                    }
                    not_beaten.pop(bot_id, None)
                    log(f"*** FAMILY BEATEN: {fam['label']} at {hc:.1%} on attempt {attempts} ***")
                    break

                if ATTEMPT_CAP > 0 and round_attempts >= ATTEMPT_CAP:
                    log(f"  family round attempt cap reached for {bot_id}")
                    break

            if bot_id not in beaten:
                not_beaten[bot_id] = {**fam, "best_hc": best_hc, "attempts": attempts}
                log(f"Family NOT beaten: {fam['label']} (best {best_hc:.1%} in {attempts} attempts)")

            state = {
                "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "deadline_sec": DURATION_SEC,
                "target_model": MODEL,
                "target_hc": TARGET_HC,
                "elapsed_s": round(time.time() - started, 1),
                "round": round_num,
                "total_attempts": total_attempts,
                "beaten_families": beaten,
                "not_beaten_families": not_beaten,
                "recent_attempts": all_attempts[-50:],
                "all_attempts_count": len(all_attempts),
            }
            save_state(state)

    elapsed = time.time() - started
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 1),
        "target_model": MODEL,
        "target_hc": TARGET_HC,
        "total_attempts": total_attempts,
        "beaten_count": len(beaten),
        "not_beaten_count": len(not_beaten),
        "beaten_families": beaten,
        "not_beaten_families": not_beaten,
        "best_overall": max((a["human_confidence"] for a in all_attempts), default=0.0),
        "all_attempts": all_attempts,
    }
    save_state(summary)

    log("=" * 72)
    log(f"DONE elapsed={elapsed:.0f}s attempts={total_attempts}")
    log(f"BEATEN ({len(beaten)}): {', '.join(beaten.keys()) or 'none'}")
    if beaten:
        for k, v in beaten.items():
            log(f"  {k}: {v['best_hc']:.1%} in {v['attempts_to_beat']} tries")
    log(f"NOT BEATEN ({len(not_beaten)}):")
    for k, v in sorted(not_beaten.items(), key=lambda x: -x[1]["best_hc"]):
        log(f"  {k}: best {v['best_hc']:.1%} ({v['attempts']} tries)")
    log(f"Results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
