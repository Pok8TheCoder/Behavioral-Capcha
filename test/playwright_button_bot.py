"""Visible Playwright runner — live page.mouse with tuned pacing for Qmodel."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

from qmodel_evasion_bot import EvasionConfig, ease_out_in, evasion_cfg_from_dict

DASHBOARD_URL = "http://127.0.0.1:8010/"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
ACTIVE_MODEL = "qmodel_v0.2"
BEST_RESULT_PATH = Path(__file__).with_name("qmodel_best_result.json")
# >1.0 = faster movement, fewer recorded pointer events
SPEED_FACTOR = 1.5


def load_best_cfg() -> EvasionConfig:
    data = json.loads(BEST_RESULT_PATH.read_text(encoding="utf-8"))
    return evasion_cfg_from_dict(data["_cfg"])


async def wait_for_result(page: Page, timeout_ms: int = 25000) -> str | None:
    badge = page.locator("#decision-badge")
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        text = (await badge.text_content() or "").strip()
        if text and text not in {"IN PROGRESS", "UNKNOWN"}:
            return text
        await asyncio.sleep(0.2)
    return None


async def canvas_rect(page: Page) -> dict[str, float]:
    rect = await page.locator("#challenge-canvas").bounding_box()
    if not rect:
        raise RuntimeError("Challenge canvas not found")
    return rect


async def move_mouse(page: Page, rect: dict[str, float], logical_x: float, logical_y: float, steps: int = 1) -> None:
    sx = rect["x"] + (logical_x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (logical_y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy, steps=steps)


async def move_along_live(
    page: Page,
    rect: dict[str, float],
    start: tuple[float, float],
    end: tuple[float, float],
    cfg: EvasionConfig,
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return end

    points = max(8, int(dist / random.uniform(7.0, 11.0)))
    duration_ms = dist / random.uniform(cfg.min_speed_px_ms, cfg.max_speed_px_ms)
    duration_ms = max(duration_ms, points * random.uniform(*cfg.move_interval_ms))
    duration_ms /= SPEED_FACTOR
    step_delay = duration_ms / points / 1000.0

    nx, ny = (-dy / dist, dx / dist)
    wobble = cfg.arc_strength * (1.0 - cfg.directness)

    for i in range(1, points + 1):
        raw = i / points
        eased = ease_out_in(raw)
        base_x = start[0] + dx * eased
        base_y = start[1] + dy * eased
        wave = math.sin(raw * math.pi) * wobble * random.uniform(0.6, 1.2)
        tremor = cfg.tremor_px * random.uniform(-1, 1)
        x = base_x + nx * wave + tremor
        y = base_y + ny * wave + tremor * 0.7
        await move_mouse(page, rect, x, y, steps=1)
        await asyncio.sleep(step_delay * random.uniform(0.85, 1.05))

    return end


async def click_live(page: Page, rect: dict[str, float], x: float, y: float, cfg: EvasionConfig) -> None:
    ox = x + random.uniform(-cfg.aim_jitter_px, cfg.aim_jitter_px)
    oy = y + random.uniform(-cfg.aim_jitter_px, cfg.aim_jitter_px)
    await move_mouse(page, rect, ox, oy, steps=1)
    await asyncio.sleep(random.uniform(0.015, 0.035))
    sx = rect["x"] + (ox / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (oy / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    dwell_s = random.uniform(*cfg.dwell_ms) / 1000.0 / SPEED_FACTOR
    await asyncio.sleep(dwell_s)
    await page.mouse.up()


async def solve_challenge_live(page: Page, challenge: dict[str, Any], cfg: EvasionConfig) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})

    canvas = page.locator("#challenge-canvas")
    await canvas.scroll_into_view_if_needed()
    rect = await canvas_rect(page)
    await move_mouse(page, rect, area["x"] + 60, area["y"] + 60, steps=1)
    await asyncio.sleep(random.uniform(0.05, 0.1))

    pos = (area["x"] + random.uniform(30, 90), area["y"] + random.uniform(30, 90))
    await move_mouse(page, rect, pos[0], pos[1], steps=1)

    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        pos = await move_along_live(page, rect, pos, target, cfg)
        await asyncio.sleep(random.uniform(0.015, 0.04))
        await click_live(page, rect, target[0], target[1], cfg)
        pos = target
        await asyncio.sleep(random.uniform(*cfg.pause_ms) / 1000.0 / SPEED_FACTOR)


async def run_visible_bot() -> None:
    cfg = load_best_cfg()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=0)
        context = await browser.new_context(viewport={"width": 1440, "height": 960})
        page = await context.new_page()

        print(f"Opening {DASHBOARD_URL}")
        await page.goto(DASHBOARD_URL, wait_until="networkidle")
        await page.select_option("#challenge-type", "button_sequence")
        await page.select_option("#active-model", ACTIVE_MODEL)

        async with page.expect_response(
            lambda r: "/api/challenge/new" in r.url and r.request.method == "POST"
        ) as response_info:
            await page.click("#btn-new-challenge")

        challenge = await (await response_info.value).json()
        await page.wait_for_selector("#challenge-canvas")

        print(
            f"Model={ACTIVE_MODEL} seed={challenge.get('seed')} "
            f"buttons={len(challenge['spec']['buttons'])}"
        )
        print("Moving mouse live on canvas...")

        await solve_challenge_live(page, challenge, cfg)

        badge = (await page.locator("#decision-badge").text_content() or "").strip()
        if badge == "IN PROGRESS":
            await page.click("#btn-submit")

        await wait_for_result(page)

        print("--- Result ---")
        print(f"Human confidence: {(await page.locator('#human-confidence').text_content() or '').strip()}")
        print(f"Decision: {(await page.locator('#decision-badge').text_content() or '').strip()}")
        print(f"Task: {(await page.locator('#task-result').text_content() or '').strip()}")
        print(f"Recorded events: {(await page.locator('#meta-events').text_content() or '').strip()}")

        print("Leaving browser open for 60 seconds...")
        await asyncio.sleep(60)
        await browser.close()


def main() -> None:
    asyncio.run(run_visible_bot())


if __name__ == "__main__":
    main()
