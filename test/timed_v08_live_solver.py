"""Live Playwright button-sequence solver for Qmodel V0.8 (no replay).

Avoids Issue 006 min-jerk overflow (events 280–550, jitter 0.10–0.22, vel ~0.86)
and prior dart / dense-hop / micro-jiggle families.
Explores mid jitter, higher straightness, ~200 events, slower velocity.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

DASHBOARD_URL = "http://127.0.0.1:8010/"
ACTIVE_MODEL = "qmodel_v0.8"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
TARGET_HC = 0.70
DEADLINE_SEC = 18 * 60
LOG_PATH = Path(__file__).with_name("timed_v08_live_log.txt")
RESULT_PATH = Path(__file__).with_name("timed_v08_live_result.json")


def log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def parse_hc(text: str) -> float:
    m = re.search(r"(\d+)", text or "")
    return float(m.group(1)) / 100.0 if m else 0.0


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


async def move_mouse(page: Page, rect: dict[str, float], x: float, y: float) -> None:
    sx = rect["x"] + (x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)


async def travel(page: Page, rect: dict, start: tuple, end: tuple, cfg: dict) -> tuple:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        await move_mouse(page, rect, end[0], end[1])
        return end
    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg["arc_px"]) * random.choice([-1.0, 1.0])
    n = max(cfg["min_points"], int(dist / random.uniform(*cfg["spacing"])))
    n = min(n, cfg.get("max_points", 80))
    for i in range(1, n + 1):
        raw = i / n
        if cfg.get("ease") == "smooth":
            poly = raw * raw * (3 - 2 * raw)
        else:
            poly = raw
        wave = math.sin(math.pi * raw) * arc
        if cfg.get("s_curve"):
            wave += math.sin(2 * math.pi * raw) * arc * 0.4
        tremor = random.gauss(0.0, cfg["tremor_px"])
        await move_mouse(
            page,
            rect,
            start[0] + dx * poly + nx * wave + tremor,
            start[1] + dy * poly + ny * wave + tremor * 0.5,
        )
        if random.random() < cfg.get("pause_p", 0.05):
            await asyncio.sleep(random.uniform(0.03, 0.10))
        else:
            await asyncio.sleep(cfg.get("dt", 0.004))
    await move_mouse(page, rect, end[0], end[1])
    return end


async def click_at(page: Page, rect: dict, x: float, y: float, cfg: dict) -> None:
    ax = x + random.gauss(0.0, cfg["aim_sigma"])
    ay = y + random.gauss(0.0, cfg["aim_sigma"])
    await move_mouse(page, rect, ax, ay)
    await asyncio.sleep(random.uniform(0.04, 0.10))
    sx = rect["x"] + (ax / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (ay / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await asyncio.sleep(random.uniform(*cfg["dwell_s"]))
    await page.mouse.up()


def sample_cfg() -> dict:
    families = [
        {
            "name": "mid_str_hops",
            "spacing": (16.0, 28.0),
            "min_points": 8,
            "max_points": 22,
            "arc_px": (4.0, 12.0),
            "tremor_px": 1.6,
            "aim_sigma": 2.2,
            "dwell_s": (0.08, 0.16),
            "pause_s": (0.14, 0.32),
            "ease": "smooth",
            "s_curve": False,
            "pause_p": 0.08,
            "dt": 0.006,
        },
        {
            "name": "slow_low_vel",
            "spacing": (10.0, 18.0),
            "min_points": 10,
            "max_points": 28,
            "arc_px": (12.0, 28.0),
            "tremor_px": 1.8,
            "aim_sigma": 2.4,
            "dwell_s": (0.12, 0.22),
            "pause_s": (0.28, 0.55),
            "ease": "linear",
            "s_curve": True,
            "pause_p": 0.16,
            "dt": 0.012,
        },
        {
            "name": "high_str_linear",
            "spacing": (18.0, 32.0),
            "min_points": 6,
            "max_points": 16,
            "arc_px": (1.0, 5.0),
            "tremor_px": 0.6,
            "aim_sigma": 1.5,
            "dwell_s": (0.09, 0.18),
            "pause_s": (0.16, 0.36),
            "ease": "smooth",
            "s_curve": False,
            "pause_p": 0.04,
            "dt": 0.008,
        },
        {
            "name": "mid_jit_turns",
            "spacing": (11.0, 20.0),
            "min_points": 12,
            "max_points": 26,
            "arc_px": (20.0, 42.0),
            "tremor_px": 2.2,
            "aim_sigma": 2.6,
            "dwell_s": (0.07, 0.14),
            "pause_s": (0.12, 0.28),
            "ease": "linear",
            "s_curve": True,
            "pause_p": 0.10,
            "dt": 0.007,
        },
    ]
    return random.choice(families)


async def solve(page: Page, challenge: dict, cfg: dict) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    rect = await canvas_rect(page)
    await page.locator("#challenge-canvas").scroll_into_view_if_needed()
    pos = (area["x"] + random.uniform(40, 140), area["y"] + random.uniform(40, 140))
    await move_mouse(page, rect, pos[0], pos[1])
    await asyncio.sleep(random.uniform(0.08, 0.16))
    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        pos = await travel(page, rect, pos, target, cfg)
        await click_at(page, rect, target[0], target[1], cfg)
        pos = target
        await asyncio.sleep(random.uniform(*cfg["pause_s"]))


async def run_attempt(page: Page, attempt: int) -> float:
    cfg = sample_cfg()
    await page.select_option("#challenge-type", "button_sequence")
    await page.select_option("#active-model", ACTIVE_MODEL)
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
    hc = parse_hc(hc_text)
    if not hc and isinstance(verify.get("human_confidence"), (int, float)):
        hc = float(verify["human_confidence"])
    detail = ""
    if resp.status != 200:
        detail = str(verify.get("detail") or task)[:180]
        hc = 0.0
    log(
        f"attempt {attempt:02d} family={cfg['name']} hc={hc_text or hc} "
        f"http={resp.status} decision={decision} events={events} "
        f"vel={feats.get('velocity_mean')} jerk={feats.get('jerk_mean')} "
        f"acc={feats.get('acceleration_mean')} jit={feats.get('jitter_ratio')} "
        f"str={feats.get('straightness_index')} heur={ev.get('heuristic_score')} "
        f"flags={flags[:150]} {detail}"
    )
    return hc


async def main() -> None:
    LOG_PATH.write_text("--- qmodel_v0.8 live solver ---\n", encoding="utf-8")
    started = time.time()
    best = -1.0
    attempt = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 960})
        await page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        await page.wait_for_selector("#challenge-canvas")
        while time.time() - started < DEADLINE_SEC:
            attempt += 1
            try:
                hc = await run_attempt(page, attempt)
            except Exception as exc:
                log(f"attempt {attempt:02d} error: {exc}")
                try:
                    await page.reload(wait_until="domcontentloaded")
                except Exception:
                    pass
                continue
            best = max(best, hc)
            if hc >= TARGET_HC:
                log(f"TARGET_REACHED {hc:.4f} after {attempt} attempts in {time.time()-started:.1f}s")
                RESULT_PATH.write_text(
                    json.dumps(
                        {
                            "human_confidence": hc,
                            "attempts": attempt,
                            "elapsed_s": round(time.time() - started, 1),
                            "best": best,
                            "model": ACTIVE_MODEL,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                await browser.close()
                return
        await browser.close()
    log(f"DEADLINE_HIT best={best:.4f} attempts={attempt}")
    RESULT_PATH.write_text(
        json.dumps(
            {"best": best, "attempts": attempt, "elapsed_s": round(time.time() - started, 1), "model": ACTIVE_MODEL},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
