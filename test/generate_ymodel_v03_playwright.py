"""Capture live Playwright button-sequence traces for Ymodel V0.3 training."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.evasion_bot import events_to_csv_rows
from test.timed_y02_live_solver import (
    DASHBOARD_URL,
    canvas_rect,
    solve,
)

OUT_DIR = ROOT / "datasets" / "bot_families"
OUT_CSV = OUT_DIR / "ymodel_v03_playwright_live.csv"
OUT_CFG = OUT_DIR / "ymodel_v03_playwright_live_cfg.json"
STATUS_PATH = ROOT / "test" / "ymodel_v03_playwright_status.json"
AFK_GAP_MS = (5_500.0, 7_500.0)
SCREEN_W = 1920
SCREEN_H = 1080
EPISODES_PER_FAMILY = 12

# Winner + near-miss families, plus sparse/high-idle variants Y0.2 is likely to lose to.
FAMILIES = [
    {
        "name": "idle_correct",
        "spacing": (8.0, 15.0), "min_points": 12, "max_points": 32,
        "arc_px": (10.0, 26.0), "tremor_px": 1.25, "aim_sigma": 2.4,
        "dwell_s": (0.08, 0.16), "pause_s": (0.12, 0.28),
        "min_jerk": True, "s_curve": True, "pause_p": 0.12, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "idle_sparse",
        "spacing": (12.0, 20.0), "min_points": 8, "max_points": 18,
        "arc_px": (8.0, 20.0), "tremor_px": 1.15, "aim_sigma": 2.2,
        "dwell_s": (0.09, 0.17), "pause_s": (0.12, 0.28),
        "min_jerk": True, "s_curve": True, "pause_p": 0.10, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "pause_heavy",
        "spacing": (8.0, 15.0), "min_points": 12, "max_points": 30,
        "arc_px": (8.0, 20.0), "tremor_px": 1.2, "aim_sigma": 2.0,
        "dwell_s": (0.09, 0.16), "pause_s": (0.12, 0.26),
        "min_jerk": True, "s_curve": True, "pause_p": 0.12, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "wobble_mid",
        "spacing": (8.0, 14.0), "min_points": 14, "max_points": 38,
        "arc_px": (18.0, 38.0), "tremor_px": 1.8, "aim_sigma": 2.3,
        "dwell_s": (0.05, 0.10), "pause_s": (0.06, 0.14),
        "min_jerk": True, "s_curve": True, "pause_p": 0.07, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "hi_jit",
        "spacing": (8.0, 14.0), "min_points": 14, "max_points": 38,
        "arc_px": (14.0, 32.0), "tremor_px": 2.8, "aim_sigma": 2.6,
        "dwell_s": (0.05, 0.10), "pause_s": (0.05, 0.12),
        "min_jerk": True, "s_curve": True, "pause_p": 0.05, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "hi_jit_long",
        "spacing": (7.0, 13.0), "min_points": 20, "max_points": 50,
        "arc_px": (16.0, 36.0), "tremor_px": 2.6, "aim_sigma": 2.5,
        "dwell_s": (0.05, 0.10), "pause_s": (0.05, 0.10),
        "min_jerk": True, "s_curve": True, "pause_p": 0.05, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "smooth_low_jit",
        "spacing": (8.0, 14.0), "min_points": 16, "max_points": 40,
        "arc_px": (6.0, 14.0), "tremor_px": 0.4, "aim_sigma": 1.5,
        "dwell_s": (0.05, 0.10), "pause_s": (0.05, 0.10),
        "min_jerk": True, "s_curve": True, "pause_p": 0.04, "dt": 0.0,
        "move_steps": 1, "irregular": False,
    },
    {
        "name": "high_str_fast",
        "spacing": (14.0, 24.0), "min_points": 8, "max_points": 20,
        "arc_px": (2.0, 8.0), "tremor_px": 0.7, "aim_sigma": 1.6,
        "dwell_s": (0.04, 0.09), "pause_s": (0.04, 0.10),
        "min_jerk": True, "s_curve": False, "pause_p": 0.03, "dt": 0.0,
        "move_steps": 1, "irregular": False,
    },
    {
        "name": "sparse_mid",
        "spacing": (13.0, 22.0), "min_points": 10, "max_points": 24,
        "arc_px": (10.0, 24.0), "tremor_px": 1.5, "aim_sigma": 2.3,
        "dwell_s": (0.05, 0.11), "pause_s": (0.06, 0.14),
        "min_jerk": True, "s_curve": True, "pause_p": 0.06, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "mid_count",
        "spacing": (8.0, 14.0), "min_points": 16, "max_points": 42,
        "arc_px": (8.0, 22.0), "tremor_px": 1.2, "aim_sigma": 2.0,
        "dwell_s": (0.05, 0.11), "pause_s": (0.05, 0.13),
        "min_jerk": True, "s_curve": False, "pause_p": 0.04, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "tight_arc",
        "spacing": (9.0, 16.0), "min_points": 12, "max_points": 32,
        "arc_px": (3.0, 10.0), "tremor_px": 0.7, "aim_sigma": 1.6,
        "dwell_s": (0.04, 0.09), "pause_s": (0.04, 0.09),
        "min_jerk": True, "s_curve": False, "pause_p": 0.03, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
    {
        "name": "dwell_long",
        "spacing": (9.0, 16.0), "min_points": 10, "max_points": 26,
        "arc_px": (10.0, 24.0), "tremor_px": 1.3, "aim_sigma": 2.2,
        "dwell_s": (0.11, 0.20), "pause_s": (0.16, 0.32),
        "min_jerk": True, "s_curve": True, "pause_p": 0.14, "dt": 0.0,
        "move_steps": 1, "irregular": True,
    },
]


def log(msg: str) -> None:
    print(msg, flush=True)


def write_status(payload: dict) -> None:
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def events_to_rows(events: list[dict], t_cursor: float, family: str) -> list[dict[str, str]]:
    return events_to_csv_rows(events, t_cursor, target_item=f"playwright:{family}")


async def capture_episode(page: Page, cfg: dict) -> list[dict]:
    posted: dict = {}

    def on_request(req) -> None:
        if "/api/challenge/verify" in req.url and req.method == "POST":
            try:
                posted["body"] = req.post_data_json()
            except Exception:
                posted["raw"] = req.post_data

    page.on("request", on_request)
    try:
        await page.select_option("#challenge-type", "button_sequence", timeout=8000)
        # Do not change #active-model here — reloading XGBoost CUDA mid-loop crashes uvicorn.
        async with page.expect_response(
            lambda r: "/api/challenge/new" in r.url and r.request.method == "POST",
            timeout=15000,
        ) as ri:
            await page.click("#btn-new-challenge")
        challenge = await (await ri.value).json()
        await page.wait_for_selector("#challenge-canvas")
        await canvas_rect(page)
        async with page.expect_response(lambda r: "/api/challenge/verify" in r.url, timeout=40000) as vr:
            await solve(page, challenge, cfg)
            badge = (await page.locator("#decision-badge").text_content() or "").strip()
            if badge == "IN PROGRESS":
                await page.click("#btn-submit")
            await vr.value
        body = posted.get("body") or {}
        events = body.get("events") or []
        if not events and posted.get("raw"):
            try:
                events = json.loads(posted["raw"]).get("events") or []
            except Exception:
                events = []
        return events
    finally:
        page.remove_listener("request", on_request)


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fam in FAMILIES:
        jobs.extend([fam] * EPISODES_PER_FAMILY)
    random.shuffle(jobs)
    all_rows: list[dict[str, str]] = []
    t_cursor = 0.0
    ok = 0
    failed = 0
    by_family: dict[str, int] = {}
    started = time.time()
    log(f"PLAYWRIGHT_CAPTURE families={len(FAMILIES)} episodes={len(jobs)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 960})
        page.set_default_timeout(45000)
        await page.goto(DASHBOARD_URL, wait_until="commit")
        await page.wait_for_selector("#challenge-canvas", timeout=15000)
        for i, cfg in enumerate(jobs, start=1):
            name = cfg["name"]
            try:
                events = await capture_episode(page, cfg)
            except Exception as exc:
                failed += 1
                log(f"episode {i:03d} family={name} error={exc}")
                try:
                    await page.close()
                except Exception:
                    pass
                for attempt in range(8):
                    try:
                        page = await browser.new_page(viewport={"width": 1440, "height": 960})
                        page.set_default_timeout(45000)
                        await page.goto(DASHBOARD_URL, wait_until="commit", timeout=15000)
                        await page.wait_for_selector("#challenge-canvas", timeout=15000)
                        break
                    except Exception as rec:
                        log(f"episode {i:03d} recover {attempt+1}/8 failed: {rec}")
                        try:
                            await page.close()
                        except Exception:
                            pass
                        await asyncio.sleep(2.0 + attempt)
                else:
                    log("dashboard unreachable; aborting capture")
                    break
                continue
            if not events:
                failed += 1
                log(f"episode {i:03d} family={name} empty events")
                continue
            all_rows.extend(events_to_rows(events, t_cursor, name))
            last_t = max(float(e.get("t", 0)) for e in events)
            t_cursor += last_t + random.uniform(*AFK_GAP_MS)
            ok += 1
            by_family[name] = by_family.get(name, 0) + 1
            log(f"episode {i:03d}/{len(jobs)} family={name} events={len(events)} ok={ok} fail={failed}")
            write_status(
                {
                    "ok": ok,
                    "failed": failed,
                    "elapsed_s": round(time.time() - started, 1),
                    "by_family": by_family,
                    "rows": len(all_rows),
                }
            )
        await browser.close()

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# Display Metrics: VirtualBounds=(0,0), Size=({SCREEN_W}x{SCREEN_H})\n")
        f.write("# source=playwright_live ymodel_v03\n")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rel_time_ms", "x_px", "y_px", "x_norm", "y_norm",
                "scroll_x", "scroll_y", "scroll_dx", "scroll_dy",
                "event_type", "target_item",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    summary = {
        "csv": str(OUT_CSV),
        "episodes_ok": ok,
        "episodes_failed": failed,
        "rows": len(all_rows),
        "by_family": by_family,
        "episodes_per_family_target": EPISODES_PER_FAMILY,
        "families": [f["name"] for f in FAMILIES],
        "elapsed_s": round(time.time() - started, 1),
    }
    OUT_CFG.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    log("PLAYWRIGHT_CAPTURE_COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
