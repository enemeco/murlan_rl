"""
Turn-based Murlan environment.

Implements the rules from the provided file. fileciteturn0file0

RL-friendly:
- fixed discrete action space (murlan/actions.py)
- action masking
- turn-based (one player acts)

Interpretation choices (easy to change):
- Special low straight only as A-2-3-4-5 (length=5).
- Deal all 54 cards round-robin (two players get 14).
- If previous 1st has no card in 3..10 to swap, they give their lowest card.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .cards import (
    BLACK_JOKER_ID,
    RED_JOKER_ID,
    THREE_OF_SPADES_ID,
    card_rank_i,
    counts_by_rank,
    make_deck,
    sort_cards_by_rank,
    Combo,
)
from .actions import (
    ACTION_DIM,
    PASS,
    ActionDesc,
    action_desc,
    straight_primary_rank,
    SWAP_BASE,
    SWAP_RANKS,
)

KIND_TO_I = {"none": 0, "single": 1, "pair": 2, "triple": 3, "quad": 4, "straight": 5}


@dataclass
class StepInfo:
    phase: str
    current_player: int
    last_actor: int
    legal_mask: np.ndarray
    played: Optional[Combo]
    hand_over: bool
    match_over: bool
    hand_finish_order: Optional[List[int]]
    scores: List[int]
    goal_target: int


class MurlanEnv:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

        self.scores = [0, 0, 0, 0]
        self.goal_target = 21
        self.max_target = 51
        self.hand_index = 0
        self.prev_finish_order: Optional[List[int]] = None

        self.hands: List[List[int]] = [[] for _ in range(4)]
        self.finished: List[bool] = [False] * 4
        self.finish_order: List[int] = []

        self.current_combo: Optional[Combo] = None
        self.last_player: Optional[int] = None
        self.passed: List[bool] = [False] * 4

        self.phase: str = "play"  # 'swap' or 'play'
        self.current_player: int = 0
        self.pending_starter: Optional[int] = None
        self.swap_first: Optional[int] = None
        self.swap_last: Optional[int] = None
        self.require_three_spades: bool = True

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng.seed(seed)

        self.scores = [0, 0, 0, 0]
        self.goal_target = 21
        self.hand_index = 0
        self.prev_finish_order = None

        self._start_new_hand(first_hand=True)

        obs = self.observe(self.current_player)
        mask = self.legal_mask(self.current_player)
        return obs, mask

    def observe(self, player: int) -> np.ndarray:
        phase_swap = 1.0 if self.phase == "swap" else 0.0
        phase_play = 1.0 if self.phase == "play" else 0.0

        my_counts = np.array(counts_by_rank(self.hands[player]), dtype=np.float32) / 14.0
        sizes = np.array([len(h) for h in self.hands], dtype=np.float32) / 14.0
        scores = np.array(self.scores, dtype=np.float32) / float(self.max_target)
        goal = np.array([self.goal_target / float(self.max_target)], dtype=np.float32)

        kind = "none" if self.current_combo is None else self.current_combo.kind
        kind_oh = np.zeros(6, dtype=np.float32)
        kind_oh[KIND_TO_I[kind]] = 1.0

        if self.current_combo is None:
            trick_size = np.array([0.0], dtype=np.float32)
            trick_val = np.array([0.0], dtype=np.float32)
        else:
            trick_size = np.array([self.current_combo.size / 12.0], dtype=np.float32)
            trick_val = np.array([self.current_combo.primary_rank / 14.0], dtype=np.float32)

        passed = np.array([1.0 if x else 0.0 for x in self.passed], dtype=np.float32)
        must_3s = np.array([1.0 if self.require_three_spades else 0.0], dtype=np.float32)

        seat = np.zeros(4, dtype=np.float32)
        seat[player] = 1.0

        obs = np.concatenate([
            np.array([phase_swap, phase_play], dtype=np.float32),
            my_counts,
            sizes,
            scores,
            goal,
            kind_oh,
            trick_size,
            trick_val,
            passed,
            must_3s,
            seat,
        ])
        return obs.astype(np.float32)

    def legal_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)

        if self.finished[player]:
            mask[PASS] = True
            return mask

        if self.phase == "swap":
            if player != self.swap_first:
                mask[PASS] = True
                return mask
            counts = counts_by_rank(self.hands[player])
            eligible = [r for r in SWAP_RANKS if counts[r] > 0]
            if eligible:
                for r in eligible:
                    mask[SWAP_BASE + (r - SWAP_RANKS[0])] = True
            else:
                mask[SWAP_BASE + 0] = True
            return mask

        if self.current_combo is not None and self.passed[player]:
            mask[PASS] = True
            return mask

        counts = counts_by_rank(self.hands[player])

        def add_if(cond: bool, a: int):
            if cond:
                mask[a] = True

        def can_play_straight(seq: Tuple[int, ...]) -> bool:
            needed = [11,12,0,1,2] if seq == (11,12,0,1,2) else list(seq)
            return all(counts[r] >= 1 for r in needed)

        # lead
        if self.current_combo is None:
            for r in range(15):
                add_if(counts[r] >= 1, 1 + r)
            for r in range(13):
                add_if(counts[r] >= 2, 16 + r)
                add_if(counts[r] >= 3, 29 + r)
                add_if(counts[r] >= 4, 42 + r)
            from .actions import STRAIGHT_BASE, STRAIGHTS
            for i, seq in enumerate(STRAIGHTS):
                add_if(can_play_straight(seq), STRAIGHT_BASE + i)

            if self.require_three_spades:
                if THREE_OF_SPADES_ID not in self.hands[player]:
                    mask[:] = False
                    mask[PASS] = True
                    return mask
                must_rank = 0
                restricted = np.zeros_like(mask)
                restricted[1 + must_rank] = mask[1 + must_rank]
                restricted[16 + must_rank] = mask[16 + must_rank]
                restricted[29 + must_rank] = mask[29 + must_rank]
                restricted[42 + must_rank] = mask[42 + must_rank]
                from .actions import STRAIGHT_BASE, STRAIGHTS
                for i, seq in enumerate(STRAIGHTS):
                    if must_rank in seq:
                        restricted[STRAIGHT_BASE + i] = mask[STRAIGHT_BASE + i]
                mask = restricted

            return mask

        # respond
        mask[PASS] = True
        cur = self.current_combo
        assert cur is not None

        # quad-on-quad only
        if cur.kind == "quad":
            for r in range(13):
                if counts[r] >= 4 and r > cur.primary_rank:
                    mask[42 + r] = True
            return mask

        if cur.kind in {"single","pair","triple"}:
            n = {"single": 1, "pair": 2, "triple": 3}[cur.kind]
            r0 = cur.primary_rank
            max_r = 14 if cur.kind == "single" else 12
            for r in range(0, max_r + 1):
                if counts[r] >= n and r > r0:
                    if cur.kind == "single":
                        mask[1 + r] = True
                    elif cur.kind == "pair":
                        mask[16 + r] = True
                    else:
                        mask[29 + r] = True

        elif cur.kind == "straight":
            from .actions import STRAIGHT_BASE, STRAIGHTS
            cur_len = cur.size
            cur_val = cur.primary_rank
            for i, seq in enumerate(STRAIGHTS):
                if len(seq) != cur_len:
                    continue
                if not can_play_straight(seq):
                    continue
                if straight_primary_rank(seq) > cur_val:
                    mask[STRAIGHT_BASE + i] = True

        # any quad beats any non-quad
        for r in range(13):
            if counts[r] >= 4:
                mask[42 + r] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, StepInfo]:
        p = self.current_player
        legal = self.legal_mask(p)
        if not legal[action]:
            action = PASS

        reward_vec = np.zeros(4, dtype=np.float32)
        played_combo: Optional[Combo] = None
        hand_over = False
        match_over = False
        hand_finish_order: Optional[List[int]] = None

        if self.phase == "swap":
            assert p == self.swap_first
            desc = action_desc(action)
            give_rank = desc.payload[0]

            give_card = self._pop_one_card_by_rank(self.swap_first, give_rank)
            if give_card is None:
                give_card = self._pop_lowest_card(self.swap_first)

            take_card = self._pop_highest_card(self.swap_last)

            self.hands[self.swap_first].append(take_card)
            self.hands[self.swap_last].append(give_card)

            self.phase = "play"
            self.swap_first = None
            self.swap_last = None

            if self.pending_starter is not None:
                self.current_player = self.pending_starter
            self.pending_starter = None

        else:
            desc = action_desc(action)

            if desc.kind == "pass":
                if self.current_combo is None:
                    reward_vec[p] -= 0.02
                else:
                    self.passed[p] = True
                    reward_vec[p] -= 0.005
            else:
                played_combo = self._apply_play_action(p, desc)
                reward_vec[p] += 0.01 * played_combo.size

                self.current_combo = played_combo
                self.last_player = p
                self.passed[p] = False

                if self.require_three_spades:
                    self.require_three_spades = False

                if len(self.hands[p]) == 0 and not self.finished[p]:
                    self.finished[p] = True
                    self.finish_order.append(p)

                if sum(not f for f in self.finished) == 1:
                    last = [i for i in range(4) if not self.finished[i]][0]
                    self.finished[last] = True
                    self.finish_order.append(last)

            if self.current_combo is not None:
                if self._trick_cleared():
                    self.current_combo = None
                    self.passed = [False] * 4
                    assert self.last_player is not None
                    if self.finished[self.last_player]:
                        self.current_player = self._next_active(self.last_player)
                    else:
                        self.current_player = self.last_player
                else:
                    self.current_player = self._next_active(p)

            if len(self.finish_order) == 4:
                hand_over = True
                hand_finish_order = list(self.finish_order)

                hand_points = [0.0] * 4
                for place, pl in enumerate(hand_finish_order):
                    pts = [3, 2, 1, 0][place]
                    self.scores[pl] += pts
                    hand_points[pl] = pts / 3.0
                reward_vec += np.array(hand_points, dtype=np.float32)

                match_over = self._update_goal_and_check_end()
                if match_over:
                    winner = int(np.argmax(self.scores))
                    for i in range(4):
                        reward_vec[i] += 1.0 if i == winner else -0.2
                else:
                    self.prev_finish_order = hand_finish_order
                    self.hand_index += 1
                    self._start_new_hand(first_hand=False)

        done = match_over
        info = StepInfo(
            phase=self.phase,
            current_player=self.current_player,
            last_actor=p,
            legal_mask=self.legal_mask(self.current_player),
            played=played_combo,
            hand_over=hand_over,
            match_over=match_over,
            hand_finish_order=hand_finish_order,
            scores=list(self.scores),
            goal_target=self.goal_target,
        )

        obs_next = self.observe(self.current_player)
        mask_next = info.legal_mask
        return obs_next, mask_next, reward_vec, done, info

    def _start_new_hand(self, first_hand: bool):
        self.hands = [[] for _ in range(4)]
        self.finished = [False] * 4
        self.finish_order = []
        self.current_combo = None
        self.last_player = None
        self.passed = [False] * 4

        deck = make_deck()
        self.rng.shuffle(deck)
        for i, c in enumerate(deck):
            self.hands[i % 4].append(c)
        for i in range(4):
            self.hands[i] = sort_cards_by_rank(self.hands[i])

        self.require_three_spades = first_hand

        if first_hand:
            starter = next(i for i in range(4) if THREE_OF_SPADES_ID in self.hands[i])
            self.pending_starter = None
            self.phase = "play"
            self.current_player = starter
            self.swap_first = None
            self.swap_last = None
            return

        assert self.prev_finish_order is not None
        prev_first = self.prev_finish_order[0]
        prev_last = self.prev_finish_order[-1]

        last_has_both_jokers = (BLACK_JOKER_ID in self.hands[prev_last]) and (RED_JOKER_ID in self.hands[prev_last])

        starter = prev_first if last_has_both_jokers else prev_last

        if last_has_both_jokers:
            self.pending_starter = None
            self.phase = "play"
            self.swap_first = None
            self.swap_last = None
            self.current_player = starter
        else:
            self.pending_starter = starter
            self.phase = "swap"
            self.swap_first = prev_first
            self.swap_last = prev_last
            self.current_player = prev_first  # swap chooser acts first

    def _update_goal_and_check_end(self) -> bool:
        ge = [i for i, s in enumerate(self.scores) if s >= self.goal_target]
        if len(ge) == 1:
            return True
        if len(ge) >= 2:
            self.goal_target = min(self.goal_target + 10, self.max_target)
            return False
        return False

    def _next_active(self, p: int) -> int:
        for k in range(1, 5):
            q = (p + k) % 4
            if not self.finished[q]:
                return q
        return p

    def _trick_cleared(self) -> bool:
        assert self.last_player is not None
        for i in range(4):
            if i == self.last_player or self.finished[i]:
                continue
            if not self.passed[i]:
                return False
        return True

    def _pop_one_card_by_rank(self, p: int, rank_i: int) -> Optional[int]:
        for idx, c in enumerate(self.hands[p]):
            if card_rank_i(c) == rank_i:
                return self.hands[p].pop(idx)
        return None

    def _pop_lowest_card(self, p: int) -> int:
        return self.hands[p].pop(0)

    def _pop_highest_card(self, p: int) -> int:
        hi_idx = max(range(len(self.hands[p])), key=lambda i: card_rank_i(self.hands[p][i]))
        return self.hands[p].pop(hi_idx)

    def _apply_play_action(self, p: int, desc: ActionDesc) -> Combo:
        kind = desc.kind

        if kind in {"single", "pair", "triple", "quad"}:
            rank_i = desc.payload[0]
            need = {"single": 1, "pair": 2, "triple": 3, "quad": 4}[kind]

            removed: List[int] = []

            if self.require_three_spades and rank_i == 0 and THREE_OF_SPADES_ID in self.hands[p]:
                self.hands[p].remove(THREE_OF_SPADES_ID)
                removed.append(THREE_OF_SPADES_ID)

            while len(removed) < need:
                card = self._pop_one_card_by_rank(p, rank_i)
                if card is None:
                    raise RuntimeError("Tried to play unavailable cards")
                removed.append(card)

            return Combo(kind=kind, ranks=(rank_i,))

        if kind == "straight":
            seq = tuple(desc.payload)
            needed = list(seq)
            removed: List[int] = []

            if self.require_three_spades and 0 in needed and THREE_OF_SPADES_ID in self.hands[p]:
                self.hands[p].remove(THREE_OF_SPADES_ID)
                removed.append(THREE_OF_SPADES_ID)
                needed.remove(0)

            for r in needed:
                card = self._pop_one_card_by_rank(p, r)
                if card is None:
                    raise RuntimeError("Tried to play unavailable straight")
                removed.append(card)

            if seq == (11, 12, 0, 1, 2):
                canon = (0, 1, 2, 11, 12)
            else:
                canon = seq
            return Combo(kind="straight", ranks=canon)

        raise ValueError(f"Unsupported action kind: {kind}")