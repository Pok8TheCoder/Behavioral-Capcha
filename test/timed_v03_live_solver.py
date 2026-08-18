"""Live Playwright button-sequence solver for Qmodel V0.3 (no replay).

Aims outside V0.3 wobble heuristics: higher straightness, moderate jitter,
slower mean velocity, longer click dwell.
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
ACTIVE_MODEL = "qmodel_v0.4"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
TARGET_HC = 0.70
DEADLINE_SEC = 11 * 60
LOG_PATH = Path(__file__).with_name("timed_v03_live_log.txt")


def log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def parse_hc(text: str) -> float:
    m = re.search(r"(\d+)", text or "")
    return float(m.group(1)) / 100.0 if m else 0.0


def ease_out_in(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


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


async def travel(
    page: Page,
    rect: dict[str, float],
    start: tuple[float, float],
    end: tuple[float, float],
    cfg: dict,
) -> tuple[float, float]:
    """Fast, bursty 1px-OS-like motion targeting human vel_mean ~1.5 px/ms."""
    x, y = start
    tx, ty = end
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist < 1:
        await move_mouse(page, rect, tx, ty)
        return end

    nx, ny = (-dy / dist, dx / dist)
    arc = random.uniform(*cfg["arc_px"]) * random.choice([-1.0, 1.0])
    traveled = 0.0
    while traveled < dist:
        remain = dist - traveled
        # Mix tiny and large steps so velocity_std stays high like real mice.
        if random.random() < 0.18:
            step = min(remain, random.uniform(2.0, 6.0))
        else:
            step = min(remain, random.uniform(*cfg["step_px"]))
        traveled += step
        raw = min(1.0, traveled / dist)
        eased = ease_out_in(raw)
        base_x = start[0] + dx * eased
        base_y = start[1] + dy * eased
        wave = math.sin(math.pi * raw) * arc
        tremor = random.gauss(0.0, cfg["tremor_px"])
        await move_mouse(page, rect, base_x + nx * wave + tremor, base_y + ny * wave + tremor * 0.65)
        # Let Playwright's native hop set dt; extra sleep only for rare stalls.
        if random.random() < 0.05:
            await asyncio.sleep(random.uniform(0.012, 0.04))
        else:
            await asyncio.sleep(0)
        # 1px OS-like micro-samples to lift jerk/acc toward human windows.
        for _ in range(random.randint(1, 3)):
            await move_mouse(
                page,
                rect,
                base_x + nx * wave + random.uniform(-1.4, 1.4),
                base_y + ny * wave + random.uniform(-1.4, 1.4),
            )
            await asyncio.sleep(0)
    await move_mouse(page, rect, tx, ty)
    return end


async def click_at(page: Page, rect: dict[str, float], x: float, y: float, cfg: dict) -> None:
    ax = x + random.gauss(0.0, cfg["aim_sigma"])
    ay = y + random.gauss(0.0, cfg["aim_sigma"])
    await move_mouse(page, rect, ax, ay)
    await asyncio.sleep(random.uniform(0.04, 0.12))
    sx = rect["x"] + (ax / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (ay / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await asyncio.sleep(random.uniform(*cfg["dwell_s"]))
    # tiny press wobble
    if random.random() < 0.55:
        await page.mouse.move(sx + random.uniform(-1.2, 1.2), sy + random.uniform(-1.2, 1.2))
    await page.mouse.up()


def sample_cfg() -> dict:
    # Mutate around the 54% os_direct winner (vel~1.19, str~0.33, events~103).
    return {
        "step_px": (18.0, 42.0),
        "speed_px_ms": (1.40, 2.60),
        "arc_px": (22.0, 55.0),
        "tremor_px": random.uniform(0.8, 1.6),
        "aim_sigma": random.uniform(1.4, 2.4),
        "dwell_s": (0.03, 0.07),
        "pause_s": (0.04, 0.12),
        "name": "os_direct_v2",
    }


async def solve(page: Page, challenge: dict, cfg: dict) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    rect = await canvas_rect(page)
    await page.locator("#challenge-canvas").scroll_into_view_if_needed()

    pos = (
        area["x"] + random.uniform(40, 140),
        area["y"] + random.uniform(40, 140),
    )
    await move_mouse(page, rect, pos[0], pos[1])
    await asyncio.sleep(random.uniform(0.05, 0.12))

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
    async with page.expect_response(lambda r: "/api/challenge/verify" in r.url and r.request.method == "POST") as vr:
        await solve(page, challenge, cfg)
        badge = (await page.locator("#decision-badge").text_content() or "").strip()
        if badge == "IN PROGRESS":
            await page.click("#btn-submit")
    verify = {}
    try:
        verify = await (await vr.value).json()
    except Exception:
        pass
    await wait_for_result(page)

    hc_text = (await page.locator("#human-confidence").text_content() or "").strip()
    decision = (await page.locator("#decision-badge").text_content() or "").strip()
    events = (await page.locator("#meta-events").text_content() or "").strip()
    flags = (await page.locator("#flags-list").inner_text() or "").replace("\n", " | ")
    feats = verify.get("features") or {}
    hc = parse_hc(hc_text)
    if not hc and isinstance(verify.get("human_confidence"), (int, float)):
        hc = float(verify["human_confidence"])
    log(
        f"attempt {attempt:02d} family={cfg['name']} hc={hc_text} "
        f"decision={decision} events={events} "
        f"vel={feats.get('velocity_mean')} jerk={feats.get('jerk_mean')} "
        f"acc={feats.get('acceleration_mean')} jit={feats.get('jitter_ratio')} "
        f"str={feats.get('straightness_index')} flags={flags[:120]}"
    )
    return hc


async def main() -> None:
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    log("--- restart os_direct_v2 ---")
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
                result = {
                    "human_confidence": hc,
                    "attempts": attempt,
                    "elapsed_s": round(time.time() - started, 1),
                    "best": best,
                }
                Path(__file__).with_name("timed_v03_live_result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                await browser.close()
                return

        await browser.close()

    log(f"DEADLINE_HIT best={best:.4f} attempts={attempt}")
    Path(__file__).with_name("timed_v03_live_result.json").write_text(
        json.dumps({"best": best, "attempts": attempt, "elapsed_s": round(time.time() - started, 1)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
