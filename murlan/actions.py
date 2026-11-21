"""
Action space for Murlan.

We use a fixed discrete action space + legal-action masking.

Action IDs:
  0: PASS
  1..15: SINGLE of rank r (r=0..14)
  16..28: PAIR of rank r (r=0..12)
  29..41: TRIPLE of rank r (r=0..12)
  42..54: QUAD of rank r (r=0..12)
  55..(55+N_STRAIGHTS-1): STRAIGHT patterns (length>=5)
  ... last 8 actions: SWAP_GIVE ranks 3..10 (only used in swap phase)

Straight patterns:
  - all sequences within ranks 3..A (inclusive) with length 5..12
  - all sequences starting at 2, from 2-3-4-5-6 up to 2-3-...-K-A (5..13 cards)
  - all sequences starting at A, from A-2-3-4-5 up to A-2-...-K-A (5..14 cards)

Five-card order: A-2-3-4-5, 2-3-4-5-6, 3-4-5-6-7, ..., 10-J-Q-K-A.
A straight can only be beaten by the immediately following one of equal length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .cards import RANKS, RANK_TO_I, straight_primary_rank

PASS = 0

SINGLE_BASE = 1
SINGLE_N = 15  # 3..RJ

PAIR_BASE = SINGLE_BASE + SINGLE_N
PAIR_N = 13  # 3..2 (no jokers)

TRIPLE_BASE = PAIR_BASE + PAIR_N
TRIPLE_N = 13

QUAD_BASE = TRIPLE_BASE + TRIPLE_N
QUAD_N = 13

STRAIGHT_BASE = QUAD_BASE + QUAD_N

# Swap-give actions (ranks 3..10)
SWAP_RANKS = list(range(RANK_TO_I["3"], RANK_TO_I["10"] + 1))  # 0..7
SWAP_N = len(SWAP_RANKS)  # 8


@dataclass(frozen=True)
class ActionDesc:
    kind: str  # 'pass','single','pair','triple','quad','straight','swap_give'
    payload: Tuple[int, ...]


def _build_straights() -> List[Tuple[int, ...]]:
    # Indices for 3..A are 0..11
    straights: List[Tuple[int, ...]] = []
    for length in range(5, 13):
        for start in range(0, 12 - length + 1):
            seq = tuple(range(start, start + length))
            straights.append(seq)
    # Special low straight A-2-3-4-5
    straights.append((11, 12, 0, 1, 2))
    # Append the entire 2-low family, preserving the previous 2-3-4-5-6 ID.
    # checkpoint.load_model migrates the old policy's shifted swap outputs.
    for length in range(5, 14):
        straights.append((RANK_TO_I["2"], *range(length - 1)))
    # Preserve all previous IDs. The 14-card run uses two distinct aces.
    for length in range(6, 15):
        straights.append((RANK_TO_I["A"], RANK_TO_I["2"], *range(length - 2)))
    return straights


STRAIGHTS: List[Tuple[int, ...]] = _build_straights()
N_STRAIGHTS = len(STRAIGHTS)

SWAP_BASE = STRAIGHT_BASE + N_STRAIGHTS
ACTION_DIM = SWAP_BASE + SWAP_N


def action_desc(a: int) -> ActionDesc:
    if a == PASS:
        return ActionDesc("pass", ())

    if SINGLE_BASE <= a < SINGLE_BASE + SINGLE_N:
        r = a - SINGLE_BASE
        return ActionDesc("single", (r,))

    if PAIR_BASE <= a < PAIR_BASE + PAIR_N:
        r = a - PAIR_BASE
        return ActionDesc("pair", (r,))

    if TRIPLE_BASE <= a < TRIPLE_BASE + TRIPLE_N:
        r = a - TRIPLE_BASE
        return ActionDesc("triple", (r,))

    if QUAD_BASE <= a < QUAD_BASE + QUAD_N:
        r = a - QUAD_BASE
        return ActionDesc("quad", (r,))

    if STRAIGHT_BASE <= a < STRAIGHT_BASE + N_STRAIGHTS:
        i = a - STRAIGHT_BASE
        return ActionDesc("straight", tuple(STRAIGHTS[i]))

    if SWAP_BASE <= a < SWAP_BASE + SWAP_N:
        idx = a - SWAP_BASE
        r = SWAP_RANKS[idx]
        return ActionDesc("swap_give", (r,))

    raise ValueError(f"Unknown action id: {a}")


def pretty_action(a: int) -> str:
    d = action_desc(a)
    if d.kind == "pass":
        return "PASS"
    if d.kind in {"single", "pair", "triple", "quad"}:
        r = d.payload[0]
        return f"{d.kind.upper()}({RANKS[r]})"
    if d.kind == "straight":
        names = [RANKS[x] for x in d.payload]
        return f"STRAIGHT({','.join(names)})"
    if d.kind == "swap_give":
        r = d.payload[0]
        return f"SWAP_GIVE({RANKS[r]})"
    return str(d)
