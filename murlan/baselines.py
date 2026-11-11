"""Reproducible baselines using only a player's information set."""

from functools import lru_cache
import numpy as np
from .actions import PASS, SWAP_BASE, action_desc
from .cards import counts_by_rank
from .features import ACTION_COUNTS

PLAY_ACTIONS = tuple(range(1, SWAP_BASE))
CONTAINING = [tuple(a for a in PLAY_ACTIONS if ACTION_COUNTS[a, r]) for r in range(15)]


@lru_cache(maxsize=100_000)
def minimum_plays(counts: tuple) -> int:
    """Exact minimum partition of a hand into lead combinations (ignores rivals)."""
    if not any(counts):
        return 0
    rank = next(r for r, n in enumerate(counts) if n)
    best = sum(counts)
    for a in CONTAINING[rank]:
        need = ACTION_COUNTS[a]
        if all(n >= k for n, k in zip(counts, need)):
            rest = tuple(n - int(k) for n, k in zip(counts, need))
            best = min(best, 1 + minimum_plays(rest))
    return best


def choose_action(env, name="strategic", rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    p = env.current_player
    legal = np.flatnonzero(env.legal_mask(p))
    if name == "random":
        return int(rng.choice(legal))
    if name not in ("greedy", "strategic"):
        raise ValueError(f"Unknown baseline: {name}")
    counts = np.asarray(counts_by_rank(env.hands[p]))
    if env.phase == "swap":
        return int(
            min(
                legal,
                key=lambda a: (
                    (
                        minimum_plays(tuple(np.maximum(counts - ACTION_COUNTS[a], 0)))
                        if name == "strategic"
                        else 0
                    ),
                    action_desc(int(a)).payload[0],
                ),
            )
        )
    plays = [int(a) for a in legal if a != PASS]
    if not plays:
        return PASS
    # Always take an immediate finish.
    for a in plays:
        if ACTION_COUNTS[a].sum() == counts.sum():
            return a
    if name == "greedy":
        return min(plays, key=lambda a: (-int(ACTION_COUNTS[a].sum()), a))
    danger = any(0 < len(env.hands[q]) <= 2 for q in range(4) if q != p)
    unseen = np.maximum(
        0, np.array([4] * 13 + [1, 1]) - counts - env.played_counts.sum(0)
    )

    def score(a):
        if a == PASS:
            return -4.0 * minimum_plays(tuple(counts)) - (5.0 if danger else 0.45)
        need = ACTION_COUNTS[a]
        rest = tuple(int(n) for n in counts - need)
        desc = action_desc(a)
        size = int(need.sum())
        rank = max(desc.payload)
        split = sum(1 for n, k in zip(counts, need) if k and n > k and n >= 2)
        control_cost = sum(float(need[r]) * (r / 14.0) ** 3 for r in range(15))
        unbeatable = desc.kind == "single" and unseen[rank + 1 :].sum() == 0
        return (
            -4.0 * minimum_plays(rest)
            + 0.25 * size
            - 0.5 * split
            - (0.1 if danger else 0.8) * control_cost
            - (0.0 if danger else 0.65)
            * (
                desc.kind == "quad"
                and env.current_combo is not None
                and env.current_combo.kind != "quad"
            )
            + (0.5 if unbeatable and sum(rest) <= 4 else 0)
            - 0.005 * rank
        )

    return int(max(legal, key=lambda a: (score(int(a)), -int(a))))
