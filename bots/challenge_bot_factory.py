"""Factory for challenge-solving bots across all tiers."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Tuple

from bots.base_bot import BaseBot
from bots.simulated_humanoid import SimulatedHumanoidBot
from bots.tier1_naive import NaiveBot
from bots.tier2_linear import LinearBot
from bots.tier3_bezier import BezierBot
from bots.tier4_humanoid import HumanoidFittsBot


BotFactory = Callable[[random.Random], Tuple[BaseBot, str, bool]]


def _linear_fast(_: random.Random) -> Tuple[BaseBot, str, bool]:
    return LinearBot(steps=30, step_delay_sec=0.0), "linear", True


def _linear_timed(rng: random.Random) -> Tuple[BaseBot, str, bool]:
    return LinearBot(steps=rng.randint(24, 36), step_delay_sec=0.0), "linear", True


def _bezier_fast(_: random.Random) -> Tuple[BaseBot, str, bool]:
    return BezierBot(steps=40, step_delay_sec=0.0), "linear", True


def _bezier_timed(rng: random.Random) -> Tuple[BaseBot, str, bool]:
    return BezierBot(steps=rng.randint(32, 48), step_delay_sec=0.0), "linear", True


def _humanoid_fast(_: random.Random) -> Tuple[BaseBot, str, bool]:
    return HumanoidFittsBot(target_width=40.0), "humanoid", True


def _humanoid_simulated(rng: random.Random) -> Tuple[BaseBot, str, bool]:
    return SimulatedHumanoidBot.random(rng), "humanoid", False


def _naive(_: random.Random) -> Tuple[BaseBot, str, bool]:
    return NaiveBot(), "linear", True


CHALLENGE_BOT_STYLES: Dict[str, BotFactory] = {
    "linear_fast": _linear_fast,
    "linear_timed": _linear_timed,
    "bezier_fast": _bezier_fast,
    "bezier_timed": _bezier_timed,
    "humanoid_fast": _humanoid_fast,
    "humanoid_simulated": _humanoid_simulated,
    "naive": _naive,
}


def pick_challenge_bot(rng: random.Random) -> Tuple[BaseBot, str, str, bool]:
    style_id = rng.choice(list(CHALLENGE_BOT_STYLES.keys()))
    bot, solve_style, fast = CHALLENGE_BOT_STYLES[style_id](rng)
    return bot, style_id, solve_style, fast


def prelude_bot(bot: BaseBot, rng: random.Random, screen_w: float, screen_h: float) -> None:
    if getattr(bot, "simulated", False):
        start_x = rng.uniform(100.0, screen_w - 100.0)
        start_y = rng.uniform(80.0, screen_h - 120.0)
        bot.record_raw(start_x, start_y)
        bot.idle_jitter(rng.uniform(150.0, 520.0), start_x, start_y)
    else:
        start_x = rng.uniform(100.0, screen_w - 100.0)
        start_y = rng.uniform(80.0, screen_h - 120.0)
        bot.record_raw(start_x, start_y)
