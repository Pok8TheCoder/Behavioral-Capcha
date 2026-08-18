"""Qmodel V0.3 Playwright runner using minimum-jerk humanoid motor profiles."""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
from typing import Any

from playwright.async_api import Page, async_playwright

from bots.simulated_humanoid import DEFAULT_PROFILES, MotorProfile, _minimum_jerk

DASHBOARD_URL = "http://127.0.0.1:8010/"
ACTIVE_MODEL = "qmodel_v0.3"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
TARGET_HC = 0.70
MAX_ATTEMPTS = 25


async def wait_for_result(page: Page, timeout_ms: int = 30000) -> None:
    badge = page.locator("#decision-badge")
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        text = (await badge.text_content() or "").strip()
        if text and text not in {"IN PROGRESS", "UNKNOWN"}:
            return
        await asyncio.sleep(0.15)


async def canvas_rect(page: Page) -> dict[str, float]:
    rect = await page.locator("#challenge-canvas").bounding_box()
    if not rect:
        raise RuntimeError("canvas missing")
    return rect


async def move_mouse(page: Page, rect: dict[str, float], x: float, y: float) -> None:
    sx = rect["x"] + (x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)


async def humanoid_move(
    page: Page,
    rect: dict[str, float],
    profile: MotorProfile,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    target_width: float,
) -> tuple[float, float]:
    distance = math.hypot(target_x - start_x, target_y - start_y)
    if distance < 1.0:
        await move_mouse(page, rect, target_x, target_y)
        return target_x, target_y

    fitts_id = math.log2(1.0 + distance / max(5.0, target_width))
    total_duration = profile.fitts_a + profile.fitts_b * fitts_id + random.uniform(-0.03, 0.05)
    total_duration = max(0.18, total_duration)

    ballistic_ratio = random.uniform(0.86, 0.94)
    ballistic_x = start_x + (target_x - start_x) * ballistic_ratio + random.uniform(-5.0, 5.0)
    ballistic_y = start_y + (target_y - start_y) * ballistic_ratio + random.uniform(-5.0, 5.0)

    dx_dir = ballistic_x - start_x
    dy_dir = ballistic_y - start_y
    perp_x = -dy_dir / (distance + 1e-6)
    perp_y = dx_dir / (distance + 1e-6)

    curve_style = random.choice(["arc", "s_curve", "elbow_swing"])
    curve_amp = (
        random.uniform(0.06, 0.22) * distance * profile.curvature_scale * random.choice([-1.0, 1.0])
    )

    phase1_duration = total_duration * random.uniform(0.74, 0.82)
    hz = random.uniform(*profile.sample_hz)
    steps_p1 = max(16, int(phase1_duration * hz))
    dt1 = phase1_duration / steps_p1

    f1 = random.uniform(7.5, 9.5)
    f2 = random.uniform(10.5, 12.5)
    phi1 = random.uniform(0.0, math.tau)
    phi2 = random.uniform(0.0, math.tau)
    tremor_scale = profile.tremor_scale
    t_accum = 0.0

    for i in range(steps_p1):
        tau = (i + 1) / steps_p1
        poly = _minimum_jerk(tau)
        base_x = start_x + (ballistic_x - start_x) * poly
        base_y = start_y + (ballistic_y - start_y) * poly

        if curve_style == "arc":
            curve_offset = math.sin(math.pi * tau) * curve_amp
        elif curve_style == "s_curve":
            curve_offset = math.sin(2.0 * math.pi * tau) * curve_amp * 0.6
        else:
            curve_offset = 4.0 * tau * (1.0 - tau) * curve_amp

        tremor_x = tremor_scale * (
            0.8 * math.sin(math.tau * f1 * t_accum + phi1) + 0.4 * math.cos(math.tau * f2 * t_accum + phi2)
        )
        tremor_y = tremor_scale * (
            0.8 * math.cos(math.tau * f1 * t_accum + phi1) + 0.4 * math.sin(math.tau * f2 * t_accum + phi2)
        )
        await move_mouse(page, rect, base_x + perp_x * curve_offset + tremor_x, base_y + perp_y * curve_offset + tremor_y)
        await asyncio.sleep(dt1)
        t_accum += dt1

    if random.random() < profile.pause_probability:
        await asyncio.sleep(random.uniform(0.025, 0.14))

    phase2_duration = total_duration - phase1_duration
    steps_p2 = max(8, int(phase2_duration * hz))
    dt2 = phase2_duration / steps_p2
    start_p2_x, start_p2_y = ballistic_x, ballistic_y

    for i in range(steps_p2):
        tau = (i + 1) / steps_p2
        poly = _minimum_jerk(tau)
        cx = start_p2_x + (target_x - start_p2_x) * poly
        cy = start_p2_y + (target_y - start_p2_y) * poly
        tremor_x = tremor_scale * 0.45 * math.sin(math.tau * f1 * t_accum + phi1)
        tremor_y = tremor_scale * 0.45 * math.cos(math.tau * f1 * t_accum + phi1)
        await move_mouse(page, rect, cx + tremor_x, cy + tremor_y)
        await asyncio.sleep(dt2)
        t_accum += dt2

    return target_x, target_y


async def humanoid_click(page: Page, rect: dict[str, float], profile: MotorProfile, x: float, y: float) -> None:
    await asyncio.sleep(random.uniform(*profile.hesitation_ms) / 1000.0)
    cx = x + random.gauss(0.0, 1.6)
    cy = y + random.gauss(0.0, 1.6)
    await move_mouse(page, rect, cx, cy)
    sx = rect["x"] + (cx / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (cy / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    dwell = max(0.042, random.gauss(0.088, 0.018))
    await asyncio.sleep(dwell)
    await page.mouse.up()


async def solve_challenge(page: Page, challenge: dict[str, Any], profile: MotorProfile, target_width: float) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    rect = await canvas_rect(page)

    x = area["x"] + random.uniform(50, 120)
    y = area["y"] + random.uniform(50, 120)
    await move_mouse(page, rect, x, y)
    await asyncio.sleep(random.uniform(0.1, 0.2))

    for btn_id in spec["sequence"]:
        btn = buttons[btn_id]
        x, y = await humanoid_move(page, rect, profile, x, y, btn["x"], btn["y"], target_width)
        await humanoid_click(page, rect, profile, btn["x"], btn["y"])
        await asyncio.sleep(random.uniform(0.12, 0.35))


def parse_hc(text: str) -> float:
    m = re.search(r"(\d+)", text or "")
    return float(m.group(1)) / 100.0 if m else 0.0


async def run_attempt(page: Page, profile: MotorProfile, attempt: int) -> float:
    target_width = random.uniform(28.0, 64.0)
    await page.select_option("#challenge-type", "button_sequence")
    await page.select_option("#active-model", ACTIVE_MODEL)
    async with page.expect_response(lambda r: "/api/challenge/new" in r.url) as ri:
        await page.click("#btn-new-challenge")
    challenge = await (await ri.value).json()
    await page.wait_for_selector("#challenge-canvas")
    await solve_challenge(page, challenge, profile, target_width)

    badge = (await page.locator("#decision-badge").text_content() or "").strip()
    if badge == "IN PROGRESS":
        await page.click("#btn-submit")
    await wait_for_result(page)

    hc_text = (await page.locator("#human-confidence").text_content() or "").strip()
    decision = (await page.locator("#decision-badge").text_content() or "").strip()
    events = (await page.locator("#meta-events").text_content() or "").strip()
    print(f"attempt {attempt:02d} profile={profile.name} hc={hc_text} decision={decision} events={events}")
    return parse_hc(hc_text)


async def main() -> None:
    best = -1.0
    profiles = DEFAULT_PROFILES * 2

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(DASHBOARD_URL, wait_until="domcontentloaded")

        for i, profile in enumerate(profiles[:MAX_ATTEMPTS], 1):
            try:
                hc = await run_attempt(page, profile, i)
            except Exception as exc:
                print(f"attempt {i:02d} error: {exc}")
                continue
            best = max(best, hc)
            if hc >= TARGET_HC:
                print(f"TARGET_REACHED {hc:.4f}")
                await browser.close()
                browser = await p.chromium.launch(headless=False)
                page = await browser.new_page()
                await page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
                await run_attempt(page, profile, 999)
                await asyncio.sleep(45)
                await browser.close()
                return

        await browser.close()
    print(f"MAX_HUMAN_CONFIDENCE {best:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
