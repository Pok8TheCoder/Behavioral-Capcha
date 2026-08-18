"""Live Playwright evasion runner for Qmodel V0.3 — curvy/slow profile search."""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from playwright.async_api import Page, async_playwright

DASHBOARD_URL = "http://127.0.0.1:8010/"
ACTIVE_MODEL = "qmodel_v0.3"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
TARGET_HC = 0.70
MAX_ATTEMPTS = 60


@dataclass
class LiveProfile:
    max_speed_px_ms: float
    min_speed_px_ms: float
    dwell_ms: tuple[float, float]
    pause_ms: tuple[float, float]
    aim_jitter_px: float
    arc_strength: float
    directness: float
    tremor_px: float


def random_profile() -> LiveProfile:
    return LiveProfile(
        max_speed_px_ms=random.uniform(0.55, 0.95),
        min_speed_px_ms=random.uniform(0.22, 0.42),
        dwell_ms=(random.uniform(130, 170), random.uniform(180, 240)),
        pause_ms=(random.uniform(250, 450), random.uniform(450, 750)),
        aim_jitter_px=random.uniform(4.0, 8.0),
        arc_strength=random.uniform(12.0, 28.0),
        directness=random.uniform(0.35, 0.62),
        tremor_px=random.uniform(2.5, 5.5),
    )


def mutate_profile(base: LiveProfile) -> LiveProfile:
    def n(v: float, lo: float, hi: float, s: float = 0.18) -> float:
        return max(lo, min(hi, v * (1 + random.uniform(-s, s))))

    return LiveProfile(
        max_speed_px_ms=n(base.max_speed_px_ms, 0.35, 1.2),
        min_speed_px_ms=n(base.min_speed_px_ms, 0.15, 0.6),
        dwell_ms=(n(base.dwell_ms[0], 120, 200), n(base.dwell_ms[1], 160, 280)),
        pause_ms=(n(base.pause_ms[0], 180, 600), n(base.pause_ms[1], 300, 900)),
        aim_jitter_px=n(base.aim_jitter_px, 2.5, 10.0),
        arc_strength=n(base.arc_strength, 6.0, 40.0),
        directness=n(base.directness, 0.25, 0.72),
        tremor_px=n(base.tremor_px, 1.5, 7.0),
    )


def ease_out_in(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


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


async def move_mouse(page: Page, rect: dict[str, float], lx: float, ly: float) -> None:
    sx = rect["x"] + (lx / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (ly / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)


async def move_segment(
    page: Page,
    rect: dict[str, float],
    start: tuple[float, float],
    end: tuple[float, float],
    profile: LiveProfile,
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return end

    points = max(10, int(dist / random.uniform(6.5, 10.5)))
    duration_ms = dist / random.uniform(profile.min_speed_px_ms, profile.max_speed_px_ms)
    duration_ms = max(duration_ms, points * random.uniform(16, 26))
    step_delay = duration_ms / points / 1000.0

    nx, ny = (-dy / dist, dx / dist)
    wobble = profile.arc_strength * (1.0 - profile.directness)

    for i in range(1, points + 1):
        raw = i / points
        eased = ease_out_in(raw)
        bx = start[0] + dx * eased
        by = start[1] + dy * eased
        wave = math.sin(raw * math.pi) * wobble * random.uniform(0.7, 1.3)
        tremor = profile.tremor_px * random.uniform(-1, 1)
        await move_mouse(page, rect, bx + nx * wave + tremor, by + ny * wave + tremor * 0.7)
        await asyncio.sleep(step_delay * random.uniform(0.92, 1.08))

    return end


async def click_target(page: Page, rect: dict[str, float], x: float, y: float, profile: LiveProfile) -> None:
    ox = x + random.uniform(-profile.aim_jitter_px, profile.aim_jitter_px)
    oy = y + random.uniform(-profile.aim_jitter_px, profile.aim_jitter_px)
    await move_mouse(page, rect, ox, oy)
    await asyncio.sleep(random.uniform(0.03, 0.07))
    sx = rect["x"] + (ox / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (oy / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    dwell = random.uniform(*profile.dwell_ms) / 1000.0
    micro = random.randint(2, 4)
    for i in range(1, micro + 1):
        await asyncio.sleep(dwell / micro)
        await move_mouse(
            page,
            rect,
            ox + random.uniform(-1.5, 1.5),
            oy + random.uniform(-1.5, 1.5),
        )
    await page.mouse.up()


async def solve_once(page: Page, challenge: dict[str, Any], profile: LiveProfile) -> None:
    spec = challenge["spec"]
    buttons = {b["id"]: b for b in spec["buttons"]}
    area = challenge.get("area", {"x": 80, "y": 120, "width": 840, "height": 480})
    rect = await canvas_rect(page)

    pos = (area["x"] + random.uniform(40, 100), area["y"] + random.uniform(40, 100))
    await move_mouse(page, rect, pos[0], pos[1])
    await asyncio.sleep(random.uniform(0.12, 0.22))

    for btn_id in spec["sequence"]:
        target = (buttons[btn_id]["x"], buttons[btn_id]["y"])
        pos = await move_segment(page, rect, pos, target, profile)
        await asyncio.sleep(random.uniform(0.04, 0.10))
        await click_target(page, rect, target[0], target[1], profile)
        pos = target
        await asyncio.sleep(random.uniform(*profile.pause_ms) / 1000.0)


def parse_confidence(text: str) -> float:
    m = re.search(r"(\d+)", text or "")
    return float(m.group(1)) / 100.0 if m else 0.0


async def run_attempt(page: Page, profile: LiveProfile, attempt: int) -> dict[str, Any]:
    await page.select_option("#challenge-type", "button_sequence")
    await page.select_option("#active-model", ACTIVE_MODEL)

    async with page.expect_response(
        lambda r: "/api/challenge/new" in r.url and r.request.method == "POST"
    ) as response_info:
        await page.click("#btn-new-challenge")

    challenge = await (await response_info.value).json()
    await page.wait_for_selector("#challenge-canvas")
    await solve_once(page, challenge, profile)

    badge = (await page.locator("#decision-badge").text_content() or "").strip()
    if badge == "IN PROGRESS":
        await page.click("#btn-submit")

    await wait_for_result(page)

    hc_text = (await page.locator("#human-confidence").text_content() or "").strip()
    decision = (await page.locator("#decision-badge").text_content() or "").strip()
    events = (await page.locator("#meta-events").text_content() or "").strip()
    hc = parse_confidence(hc_text)

    print(
        f"attempt {attempt:02d}: hc={hc_text} decision={decision} events={events} "
        f"direct={profile.directness:.2f} arc={profile.arc_strength:.1f} tremor={profile.tremor_px:.1f}"
    )
    return {"human_confidence": hc, "decision": decision, "events": events, "profile": profile}


async def main() -> None:
    best_hc = -1.0
    elite: list[tuple[float, LiveProfile]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(DASHBOARD_URL, wait_until="networkidle")

        for i in range(1, MAX_ATTEMPTS + 1):
            profile = mutate_profile(random.choice(elite)[1]) if elite and random.random() < 0.6 else random_profile()
            try:
                result = await run_attempt(page, profile, i)
            except Exception as exc:
                print(f"attempt {i:02d}: error {exc}")
                continue

            hc = float(result["human_confidence"])
            elite.append((hc, profile))
            elite.sort(key=lambda x: x[0], reverse=True)
            elite = elite[:12]
            best_hc = max(best_hc, hc)

            if hc >= TARGET_HC:
                print(f"TARGET_REACHED {hc:.4f}")
                await browser.close()
                browser = await p.chromium.launch(headless=False)
                page = await browser.new_page()
                await page.goto(DASHBOARD_URL, wait_until="networkidle")
                await run_attempt(page, profile, 999)
                await asyncio.sleep(45)
                await browser.close()
                return

        await browser.close()

    print(f"MAX_HUMAN_CONFIDENCE {best_hc:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
