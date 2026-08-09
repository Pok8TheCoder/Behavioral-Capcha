"""
Behavioral Bot Suite Package
----------------------------
Contains web scraper and bot generators:
- Tier 1: Naive Script Bot (Instant DOM clicks)
- Tier 2: Linear Vector Bot (Direct lerp)
- Tier 3: Bezier Curve Bot (Cubic Bezier splines)
- Tier 4: Humanoid Fitts Bot (Minimum jerk, Fitts's law, sub-movements, 8-12Hz tremor)
"""

from bots.tier1_naive import NaiveBot
from bots.tier2_linear import LinearBot
from bots.tier3_bezier import BezierBot
from bots.tier4_humanoid import HumanoidFittsBot
