"""Live Playwright solver vs Qmodel V0.13.

Win: 1 - mean(bot_probability) >= 0.70.
Avoid Issue 011 band: 100–145 events, vel 1.05–1.50, jitter > 0.28.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

DASHBOARD_URL = "http://127.0.0.1:8010/"
ACTIVE_MODEL = "qmodel_v0.13"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
TARGET_MODEL_HUMAN = 0.70
DEADLINE_SEC = 20 * 60
LOG_PATH = Path(__file__).with_name("timed_v013_live_log.txt")
RESULT_PATH = Path(__file__).with_name("timed_v013_live_result.json")
STATUS_PATH = Path(__file__).with_name("timed_v013_status.json")

ELITE_WEIGHT = 3.0
_family_scores: dict[str, float] = {}


def log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(payload: dict) -> None:
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def wait_for_result(page: Page, timeout_ms: int = 25000) -> None:
    badge = page.locator("#decision-badge")
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        text = (await badge.text_content() or "").strip()
        if text and text not in {"IN PROGRESS", "UNKNOWN"}:
            return
        await asyncio.sleep(0.12)


async def canvas_rect(page: Page) -> dict[str, float]:
    rect = await page.locator("#challenge-canvas").bounding_box()
    if not rect:
        raise RuntimeError("canvas missing")
    return rect


async def move_mouse(page: Page, rect: dict[str, float], x: float, y: float, steps: int = 1) -> None:
    sx = rect["x"] + (x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy, steps=max(1, steps))


def _min_jerk(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def families() -> list[dict]:
    # Escape #28: 100–145 events, vel 1.05–1.50, jitter > 0.28.
    return [
        {
            "name": "crawl_low_vel",
            "spacing": (6.0, 10.0),
            "min_points": 22,
            "max_points": 60,
            "arc_px": (10.0, 22.0),
            "tremor_px": 1.2,
            "aim_sigma": 2.0,
            "dwell_s": (0.05, 0.10),
            "pause_s": (0.05, 0.10),
            "min_jerk": True,
            "s_curve": True,
            "pause_p": 0.04,
            "dt": 0.018,
            "move_steps": 1,
            "irregular": True,
        },
        {
            "name": "burst_hi_vel",
            "spacing": (11.0, 18.0),
            "min_points": 10,
            "max_points": 24,
            "arc_px": (8.0, 18.0),
            "tremor_px": 1.0,
            "aim_sigma": 1.8,
            "dwell_s": (0.04, 0.08),
            "pause_s": (0.04, 0.08),
            "min_jerk": True,
            "s_curve": False,
            "pause_p": 0.03,
            "dt": 0.0,
            "move_steps": 1,
            "irregular": True,
        },
        {
            "name": "smooth_low_jit",
            "spacing": (8.0, 14.0),
            "min_points": 16,
            "max_points": 40,
            "arc_px": (6.0, 14.0),
            "tremor_px": 0.4,
            "aim_sigma": 1.5,
            "dwell_s": (0.05, 0.10),
            "pause_s": (0.05, 0.10),
            "min_jerk": True,
            "s_curve": True,
            "pause_p": 0.04,
            "dt": 0.0,
            "move_steps": 1,
            "irregular": False,
        },
        {
            "name": "wide_low_jerk",
            "spacing": (9.0, 15.0),
            "min_points": 14,
            "max_points": 36,
            "arc_px": (20.0, 42.0),
            "tremor_px": 1.1,
            "aim_sigma": 2.2,
            "dwell_s": (0.06, 0.12),
            "pause_s": (0.06, 0.12),
            "min_jerk": False,
            "s_curve": True,
            "pause_p": 0.08,
            "dt": 0.008,
            "move_steps": 2,
            "irregular": True,
        },
    ]


def sample_cfg() -> dict:
    pool = families()
    weights = []
    for fam in pool:
        score = _family_scores.get(fam["name"], 0.0)
        weights.append(1.0 + ELITE_WEIGHT * max(0.0, score))
    return random.choices(pool, weights=weights, k=1)[0]


async def travel(page: Page, rect: dict, start: tuple, end: tuple, cfg: dict) -> tuple:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        await move_mouse(page, rect, end[0], end[1])
        return end
    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg["arc_px"]) * random.choice([-1.0, 1.0])
    n = max(cfg["min_points"], int(dist / random.uniform(*cfg["spacing"])))
    n = min(n, cfg.get("max_points", 90))
    fractions = []
    if cfg.get("irregular"):
        acc = 0.0
        raw_steps = [random.uniform(0.4, 1.8) for _ in range(n)]
        total = sum(raw_steps)
        running = 0.0
        for s in raw_steps:
            running += s
            fractions.append(running / total)
    else:
        fractions = [(i / n) for i in range(1, n + 1)]
    for raw in fractions:
        poly = _min_jerk(raw) if cfg.get("min_jerk") else raw
        wave = math.sin(math.pi * raw) * arc
        if cfg.get("s_curve"):
            wave += math.sin(2 * math.pi * raw) * arc * 0.35
        tremor = random.gauss(0.0, cfg["tremor_px"])
        await move_mouse(
            page,
            rect,
            start[0] + dx * poly + nx * wave + tremor,
            start[1] + dy * poly + ny * wave + tremor * 0.5,
            steps=int(cfg.get("move_steps", 1)),
        )
        if random.random() < cfg.get("pause_p", 0.04):
            await asyncio.sleep(random.uniform(0.03, 0.10))
        elif cfg.get("dt", 0.0) > 0:
            await asyncio.sleep(cfg["dt"])
    await move_mouse(page, rect, end[0], end[1])
    return end


async def click_at(page: Page, rect: dict, x: float, y: float, cfg: dict) -> None:
    ax = x + random.gauss(0.0, cfg["aim_sigma"])
    ay = y + random.gauss(0.0, cfg["aim_sigma"])
    await move_mouse(page, rect, ax, ay)
    await asyncio.sleep(random.uniform(0.03, 0.09))
    sx = rect["x"] + (ax / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (ay / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await asyncio.sleep(random.uniform(*cfg["dwell_s"]))
    await page.mouse.up()


async def solve(page: Page, challenge: dict, cfg: dict) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    rect = await canvas_rect(page)
    await page.locator("#challenge-canvas").scroll_into_view_if_needed()
    pos = (area["x"] + random.uniform(40, 160), area["y"] + random.uniform(40, 160))
    await move_mouse(page, rect, pos[0], pos[1])
    await asyncio.sleep(random.uniform(0.06, 0.16))
    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        pos = await travel(page, rect, pos, target, cfg)
        await click_at(page, rect, target[0], target[1], cfg)
        pos = target
        await asyncio.sleep(random.uniform(*cfg["pause_s"]))


def model_human_from_verify(verify: dict) -> float:
    ev = verify.get("evaluation") or {}
    bot_prob = ev.get("bot_probability")
    if isinstance(bot_prob, (int, float)):
        return 1.0 - float(bot_prob)
    return 0.0


async def run_attempt(page: Page, attempt: int) -> dict:
    cfg = sample_cfg()
    await page.select_option("#challenge-type", "button_sequence", timeout=8000)
    try:
        await page.select_option("#active-model", ACTIVE_MODEL, timeout=8000)
    except Exception:
        pass
    async with page.expect_response(lambda r: "/api/challenge/new" in r.url and r.request.method == "POST") as ri:
        await page.click("#btn-new-challenge")
    challenge = await (await ri.value).json()
    await page.wait_for_selector("#challenge-canvas")
    async with page.expect_response(lambda r: "/api/challenge/verify" in r.url) as vr:
        await solve(page, challenge, cfg)
        badge = (await page.locator("#decision-badge").text_content() or "").strip()
        if badge == "IN PROGRESS":
            await page.click("#btn-submit")
    resp = await vr.value
    verify = {}
    try:
        verify = await resp.json()
    except Exception:
        pass
    await wait_for_result(page)
    hc_text = (await page.locator("#human-confidence").text_content() or "").strip()
    decision = (await page.locator("#decision-badge").text_content() or "").strip()
    events = (await page.locator("#meta-events").text_content() or "").strip()
    flags = (await page.locator("#flags-list").inner_text() or "").replace("\n", " | ")
    task = (await page.locator("#task-result").text_content() or "").strip()
    feats = verify.get("features") or {}
    ev = verify.get("evaluation") or {}
    model_h = model_human_from_verify(verify)
    bot_prob = ev.get("bot_probability")
    if resp.status != 200:
        model_h = 0.0
        detail = str(verify.get("detail") or task)[:180]
    else:
        detail = ""
    prev = _family_scores.get(cfg["name"], 0.0)
    _family_scores[cfg["name"]] = max(prev, model_h)
    log(
        f"attempt {attempt:02d} family={cfg['name']} model_h={model_h:.3f} "
        f"bot_prob={bot_prob} fused_hc={hc_text} http={resp.status} "
        f"decision={decision} events={events} vel={feats.get('velocity_mean')} "
        f"jerk={feats.get('jerk_mean')} acc={feats.get('acceleration_mean')} "
        f"jit={feats.get('jitter_ratio')} str={feats.get('straightness_index')} "
        f"heur={ev.get('heuristic_score')} flags={flags[:160]} {detail}"
    )
    return {
        "model_human": model_h,
        "bot_prob": bot_prob,
        "family": cfg["name"],
        "fused_hc": hc_text,
        "decision": decision,
        "features": feats,
        "evaluation": {
            "heuristic_score": ev.get("heuristic_score"),
            "risk_score": ev.get("risk_score"),
            "bot_probability": bot_prob,
            "flags": ev.get("flags"),
        },
        "http": resp.status,
    }


async def main() -> None:
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("--- qmodel_v0.13 live solver (target: 1-bot_prob >= 0.70) ---\n")
    started = time.time()
    best = -1.0
    best_row: dict = {}
    attempt = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 960})
        page.set_default_timeout(45000)
        await page.goto(DASHBOARD_URL, wait_until="commit")
        await page.wait_for_selector("#challenge-canvas", timeout=15000)
        while time.time() - started < DEADLINE_SEC:
            attempt += 1
            try:
                row = await run_attempt(page, attempt)
            except Exception as exc:
                log(f"attempt {attempt:02d} error: {exc}")
                try:
                    await page.close()
                except Exception:
                    pass
                try:
                    page = await browser.new_page(viewport={"width": 1440, "height": 960})
                    page.set_default_timeout(45000)
                    await page.goto(DASHBOARD_URL, wait_until="commit")
                    await page.wait_for_selector("#challenge-canvas", timeout=15000)
                except Exception as rec:
                    log(f"attempt {attempt:02d} recover failed: {rec}")
                continue
            mh = float(row["model_human"])
            if mh > best:
                best = mh
                best_row = row
            write_status(
                {
                    "model": ACTIVE_MODEL,
                    "attempts": attempt,
                    "elapsed_s": round(time.time() - started, 1),
                    "best_model_human": best,
                    "best_bot_prob": best_row.get("bot_prob"),
                    "best_family": best_row.get("family"),
                    "last_model_human": mh,
                    "last_family": row.get("family"),
                    "target": TARGET_MODEL_HUMAN,
                    "family_best": _family_scores,
                }
            )
            if mh >= TARGET_MODEL_HUMAN:
                log(f"TARGET_REACHED model_human={mh:.4f} after {attempt} attempts in {time.time()-started:.1f}s")
                RESULT_PATH.write_text(
                    json.dumps(
                        {
                            "model_human": mh,
                            "bot_probability": row.get("bot_prob"),
                            "attempts": attempt,
                            "elapsed_s": round(time.time() - started, 1),
                            "best": best,
                            "model": ACTIVE_MODEL,
                            "family": row.get("family"),
                            "fused_hc": row.get("fused_hc"),
                            "decision": row.get("decision"),
                            "features": row.get("features"),
                            "evaluation": row.get("evaluation"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                await browser.close()
                return
        await browser.close()
    log(f"DEADLINE_HIT best_model_human={best:.4f} attempts={attempt}")
    RESULT_PATH.write_text(
        json.dumps(
            {
                "best_model_human": best,
                "best_bot_prob": best_row.get("bot_prob"),
                "best_family": best_row.get("family"),
                "attempts": attempt,
                "elapsed_s": round(time.time() - started, 1),
                "model": ACTIVE_MODEL,
                "target_reached": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
