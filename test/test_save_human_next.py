"""Reproduce Keep as human data & next on the live dashboard."""

from __future__ import annotations

import asyncio
import glob
import os
import time
from pathlib import Path

from playwright.async_api import async_playwright

DASHBOARD_URL = "http://127.0.0.1:8010/?nocache=1"
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 640
HUMAN_CHALLENGE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "human" / "challenges"


async def canvas_rect(page):
    rect = await page.locator("#challenge-canvas").bounding_box()
    if not rect:
        raise RuntimeError("canvas missing")
    return rect


async def move_mouse(page, rect, x, y):
    sx = rect["x"] + (x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.move(sx, sy)


async def click_button(page, rect, x, y):
    await move_mouse(page, rect, x, y)
    await asyncio.sleep(0.05)
    sx = rect["x"] + (x / LOGICAL_WIDTH) * rect["width"]
    sy = rect["y"] + (y / LOGICAL_HEIGHT) * rect["height"]
    await page.mouse.click(sx, sy)


async def main() -> None:
    before = set(glob.glob(str(HUMAN_CHALLENGE_DIR / "**" / "*.csv"), recursive=True))
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 960})
        page.on("console", lambda msg: print(f"console:{msg.type}:{msg.text}", flush=True))
        await page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        await page.wait_for_selector("#challenge-canvas")
        await page.select_option("#challenge-type", "button_sequence")
        async with page.expect_response(lambda r: "/api/challenge/new" in r.url and r.request.method == "POST") as ri:
            await page.click("#btn-new-challenge")
        challenge = await (await ri.value).json()
        seed1 = challenge.get("seed")
        print("challenge", challenge.get("challenge_type"), "seed", seed1, flush=True)
        rect = await canvas_rect(page)
        spec = challenge["spec"]
        buttons = {b["id"]: b for b in spec["buttons"]}
        await page.locator("#challenge-canvas").scroll_into_view_if_needed()
        async with page.expect_response(lambda r: "/api/challenge/verify" in r.url) as vr:
            start = challenge.get("area", {"x": 100, "y": 140})
            await move_mouse(page, rect, start["x"] + 40, start["y"] + 40)
            for btn_id in spec["sequence"]:
                target = buttons[btn_id]
                await click_button(page, rect, target["x"], target["y"])
                await asyncio.sleep(0.12)
            badge = (await page.locator("#decision-badge").text_content() or "").strip()
            if badge == "IN PROGRESS":
                await page.click("#btn-submit")
        verify = await (await vr.value).json()
        print("verify_http", (await vr.value).status, "participant", verify.get("participant"), flush=True)
        await page.wait_for_selector("#challenge-overlay:not(.hidden)")
        save_hidden = await page.locator("#btn-save-human").evaluate("el => el.classList.contains('hidden')")
        save_disabled = await page.locator("#btn-save-human").is_disabled()
        events_meta = await page.locator("#meta-events").text_content()
        print("overlay_visible save_hidden", save_hidden, "disabled", save_disabled, "events", events_meta, flush=True)
        async with page.expect_response(lambda r: "/api/challenge/save-human" in r.url) as sr:
            async with page.expect_response(lambda r: "/api/challenge/new" in r.url and r.request.method == "POST") as nr:
                await page.locator("#btn-save-human").click(force=True)
            new_challenge = await (await nr.value).json()
        save_resp = await sr.value
        save_json = await save_resp.json()
        print("save_http", save_resp.status, save_json, flush=True)
        print("next_seed", new_challenge.get("seed"), "same_as_before", new_challenge.get("seed") == seed1, flush=True)
        overlay_hidden = await page.locator("#challenge-overlay").evaluate("el => el.classList.contains('hidden')")
        print("overlay_hidden_after", overlay_hidden, flush=True)
        await browser.close()

    time.sleep(0.3)
    after = set(glob.glob(str(HUMAN_CHALLENGE_DIR / "**" / "*.csv"), recursive=True))
    created = sorted(after - before)
    print("created_csv", created, flush=True)
    if save_resp.status != 200 or not created:
        raise SystemExit("SAVE_FAILED")
    if new_challenge.get("seed") == seed1:
        raise SystemExit("NEXT_FAILED")
    print("OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
