"""Fixed action semantics; shared by observation history and policy head."""

import numpy as np
from .actions import ACTION_DIM, action_desc, straight_primary_rank

KINDS = ["pass", "single", "pair", "triple", "quad", "straight", "swap_give"]
ACTION_FEATURES = np.zeros((ACTION_DIM, 24), dtype=np.float32)
ACTION_COUNTS = np.zeros((ACTION_DIM, 15), dtype=np.int8)
for a in range(ACTION_DIM):
    desc = action_desc(a)
    ACTION_FEATURES[a, KINDS.index(desc.kind)] = 1
    if desc.kind == "pass":
        continue
    if desc.kind == "straight":
        ACTION_COUNTS[a, list(desc.payload)] = 1
        rank = straight_primary_rank(desc.payload)
    else:
        size = {"single": 1, "pair": 2, "triple": 3, "quad": 4, "swap_give": 1}[
            desc.kind
        ]
        ACTION_COUNTS[a, desc.payload[0]] = size
        rank = desc.payload[0]
    ACTION_FEATURES[a, 7:22] = ACTION_COUNTS[a] / 4.0
    ACTION_FEATURES[a, 22] = ACTION_COUNTS[a].sum() / 12.0
    ACTION_FEATURES[a, 23] = rank / 14.0
