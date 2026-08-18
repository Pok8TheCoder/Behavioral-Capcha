import random
import time

from challenge_server.generator import CHALLENGE_TYPES, generate_challenge
from test.evasion_bot import default_cfg, generate_events

cfg = default_cfg()
for ct in CHALLENGE_TYPES:
    t = time.time()
    ch = generate_challenge(seed=1, challenge_type=ct)
    ev = generate_events(ch, cfg)
    print(ct, len(ev), round(time.time() - t, 3))
