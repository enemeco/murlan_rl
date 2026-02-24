"""
Card + rank utilities for Murlan.

Rules summary is taken from the user's provided rules file. fileciteturn0file0

We use a standard 52-card deck + 2 jokers:
  - 52 = Black Joker
  - 53 = Red Joker

Rank order (low -> high): 3,4,5,6,7,8,9,10,J,Q,K,A,2,BJ,RJ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Ranks in Murlan (low -> high)
RANKS: List[str] = ["3","4","5","6","7","8","9","10","J","Q","K","A","2","BJ","RJ"]
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}

# Suits for the 52 standard cards. Suit doesn't matter for combos, except the forced 3 of spades rule.
SUITS: List[str] = ["C", "D", "H", "S"]

BLACK_JOKER_ID = 52
RED_JOKER_ID = 53

# 3 of spades is the very first-move forced card.
THREE_OF_SPADES_ID = 0 * 4 + 3  # rank=3 => index 0; suit S => index 3


def is_joker(card_id: int) -> bool:
    return card_id in (BLACK_JOKER_ID, RED_JOKER_ID)


def card_rank_i(card_id: int) -> int:
    """Return the Murlan rank index (0..14)."""
    if card_id == BLACK_JOKER_ID:
        return RANK_TO_I["BJ"]
    if card_id == RED_JOKER_ID:
        return RANK_TO_I["RJ"]
    # Standard cards: 13 ranks x 4 suits, where rank index 0..12 maps to 3..2
    rank_i = card_id // 4  # 0..12
    return rank_i


def card_suit_i(card_id: int) -> int:
    if is_joker(card_id):
        raise ValueError("Jokers have no suit")
    return card_id % 4


def rank_name(rank_i: int) -> str:
    return RANKS[rank_i]


def make_deck() -> List[int]:
    return list(range(54))


def sort_cards_by_rank(cards: List[int]) -> List[int]:
    return sorted(cards, key=lambda c: (card_rank_i(c), card_suit_i(c) if c < 52 else 9))


def counts_by_rank(cards: List[int]) -> List[int]:
    counts = [0] * len(RANKS)
    for c in cards:
        counts[card_rank_i(c)] += 1
    return counts


@dataclass(frozen=True)
class Combo:
    """A played combination.

    kind: one of 'single','pair','triple','quad','straight'
    ranks: for n-of-a-kind: [rank_i]
           for straight: tuple of rank indices.
    """

    kind: str
    ranks: Tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.ranks) if self.kind == "straight" else {"single": 1, "pair": 2, "triple": 3, "quad": 4}[self.kind]

    @property
    def primary_rank(self) -> int:
        """Comparison key.

        - For single/pair/triple/quad: that rank
        - For straight: highest rank in the straight
          Special low straight A-2-3-4-5 is represented canonically as (0,1,2,11,12)
          but compares as ending at 5 (rank index 2).
        """
        if self.kind == "straight":
            if self.ranks == (0, 1, 2, 11, 12):
                return 2
            return self.ranks[-1]
        return self.ranks[0]