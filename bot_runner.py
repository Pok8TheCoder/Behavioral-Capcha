"""
Behavioral Bot Generator & Benchmark Suite
-------------------------------------------
Generates and benchmarks 4 tiers of bot interaction datasets:
- Tier 1: Naive Script Bot (Instant DOM clicks)
- Tier 2: Linear Vector Bot (Direct lerp interpolation)
- Tier 3: Bezier Curve Bot (Cubic Bezier splines)
- Tier 4: Humanoid Fitts Bot (Biophysical minimum jerk, Fitts's law, sub-movements, 8-12Hz tremor)

Outputs:
- datasets/bot_tier1_naive.csv
- datasets/bot_tier2_linear.csv
- datasets/bot_tier3_bezier.csv
- datasets/bot_tier4_humanoid.csv
"""

import os
import sys
import time

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from bots.tier1_naive import NaiveBot
from bots.tier2_linear import LinearBot
from bots.tier3_bezier import BezierBot
from bots.tier4_humanoid import HumanoidFittsBot

from backend.feature_extractor import extract_features
from backend.model import BehavioralClassifier

def run_bot_simulation(bot_instance, n_runs: int = 15) -> str:
    print(f"\n" + "=" * 65)
    print(f"[Bot Suite] Executing {bot_instance.name} ({n_runs} target sessions)...")
    print("=" * 65)

    bot_instance.reset()

    targets = [
        (100.0, 100.0, 450.0, 280.0),
        (450.0, 280.0, 680.0, 520.0),
        (680.0, 520.0, 320.0, 180.0),
        (320.0, 180.0, 850.0, 340.0),
        (850.0, 340.0, 200.0, 600.0)
    ]

    for run_idx in range(n_runs):
        start_x, start_y, target_x, target_y = targets[run_idx % len(targets)]
        bot_instance.move_to(start_x, start_y, target_x, target_y)
        bot_instance.click_target(target_x, target_y, button="mouse_left")

    output_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(output_dir, exist_ok=True)

    bot_slug = bot_instance.name.split(" - ")[1].lower().replace(" ", "_")
    csv_path = os.path.join(output_dir, f"bot_{bot_slug}.csv")
    bin_path = os.path.join(output_dir, f"bot_{bot_slug}.bin")

    # 1. Save binary dataset
    with open(bin_path, "wb") as f_bin:
        f_bin.write(bot_instance.binary_buffer)

    # 2. Save CSV dataset
    with open(csv_path, "w", encoding="utf-8") as f_csv:
        f_csv.write("rel_time_ms,x,y,scroll_x,scroll_y,scroll_dx,scroll_dy,event_type,target_item\n")
        for evt in bot_instance.events:
            f_csv.write(f"{evt['t']:.3f},{evt['x']:.1f},{evt['y']:.1f},{evt['scroll_x']:.1f},{evt['scroll_y']:.1f},{evt['scroll_dx']},{evt['scroll_dy']},{evt['event_type']},none\n")

    print(f"[Saver] Generated {len(bot_instance.events):,} raw events.")
    print(f"[Saver] Binary dataset: [file:///{bin_path.replace(os.sep, '/')}]")
    print(f"[Saver] CSV dataset:    [file:///{csv_path.replace(os.sep, '/')}]")

    return csv_path

def evaluate_bot_dataset(classifier: BehavioralClassifier, bot_instance):
    features = extract_features(bot_instance.events)
    evaluation = classifier.evaluate(features)

    print(f"\n--- CAPTCHA Engine Evaluation for {bot_instance.name} ---")
    print(f" -> Decision:        {evaluation['decision']}")
    print(f" -> Bot Risk Score:  {evaluation['risk_score']:.3f}")
    print(f" -> Bot Probability: {evaluation['bot_probability']:.3f}")
    print(f" -> Straightness:    {features['straightness_index']:.3f}")
    print(f" -> Velocity Mean:   {features['velocity_mean']:.3f} px/ms (Std: {features['velocity_std']:.3f})")
    print(f" -> Jitter Ratio:    {features['jitter_ratio']:.3f}")
    print(f" -> Click Dwell:     {features['avg_click_dwell_ms']:.1f} ms")
    if evaluation['flags']:
        print(f" -> Flags: {evaluation['flags']}")

def main():
    print("=" * 65)
    print(" BEHAVIORAL BOT GENERATOR & BENCHMARK SUITE")
    print("=" * 65)

    classifier = BehavioralClassifier()

    bots = [
        NaiveBot(),
        LinearBot(steps=35, step_delay_sec=0.008),
        BezierBot(steps=45, step_delay_sec=0.008),
        HumanoidFittsBot(target_width=40.0)
    ]

    for bot in bots:
        run_bot_simulation(bot, n_runs=10)
        evaluate_bot_dataset(classifier, bot)

if __name__ == "__main__":
    main()
