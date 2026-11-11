from copy import deepcopy
import numpy as np
import pytest
from murlan.env import MurlanEnv
from murlan.actions import ACTION_DIM, PASS, STRAIGHT_BASE, STRAIGHTS, SWAP_BASE
from murlan.cards import THREE_OF_SPADES_ID, Combo
from murlan.baselines import choose_action, minimum_plays


def test_illegal_actions_never_mutate_state():
    e = MurlanEnv()
    e.reset()
    before = deepcopy(e.__dict__)
    for a in [-1, ACTION_DIM, 999, True, 1.2, PASS]:
        with pytest.raises(ValueError):
            e.step(a)
        assert e.hands == before["hands"]
        assert e.current_player == before["current_player"]
        assert e.last_actions == before["last_actions"]


def test_opening_consumes_three_spades():
    e = MurlanEnv(9)
    e.reset()
    p = e.current_player
    assert THREE_OF_SPADES_ID in e.hands[p]
    e.step(int(np.flatnonzero(e.legal_mask(p))[0]))
    assert THREE_OF_SPADES_ID not in e.hands[p]
    assert not e.require_three_spades


def test_public_observation_cannot_see_opponent_cards():
    e = MurlanEnv(7)
    e.reset()
    p = e.current_player
    before = e.observe(p).copy()
    other = [i for i in range(4) if i != p]
    e.hands[other[0]][0], e.hands[other[1]][0] = (
        e.hands[other[1]][0],
        e.hands[other[0]][0],
    )
    np.testing.assert_array_equal(before, e.observe(p))
    assert len(before) == 249
    assert len(e.observe(p, version=1)) == 43


def test_score_cap_ends_with_unique_leader_and_extends_ties():
    e = MurlanEnv()
    e.reset()
    e.goal_target = 51
    e.scores = [55, 54, 20, 10]
    assert e._update_goal_and_check_end()
    e.scores = [55, 55, 20, 10]
    assert not e._update_goal_and_check_end()
    e.goal_target = 21
    e.scores = [22, 21, 10, 9]
    assert not e._update_goal_and_check_end()
    assert e.goal_target == 31


def test_low_straight_and_bombs():
    e = MurlanEnv()
    e.reset()
    e.require_three_spades = False
    e.current_player = 0
    e.hands[0] = [r * 4 for r in [11, 12, 0, 1, 2]] + [4, 5, 6, 7]
    e.current_combo = Combo("straight", (0, 1, 2, 11, 12))
    assert e.current_combo.primary_rank == 2
    low = STRAIGHT_BASE + STRAIGHTS.index((11, 12, 0, 1, 2))
    assert not e.legal_mask(0)[low]
    assert e.legal_mask(0)[43]  # four fours beats a straight
    e.current_combo = Combo("quad", (0,))
    assert set(np.flatnonzero(e.legal_mask(0))) == {0, 43}


@pytest.mark.parametrize("baseline", ["random", "greedy", "strategic"])
def test_complete_matches_terminate_and_conserve_cards(baseline):
    e = MurlanEnv(3)
    e.reset()
    rng = np.random.default_rng(4)
    hands = 0
    for _ in range(7000):
        assert sum(map(len, e.hands)) + int(e.played_counts.sum()) == 54
        mask = e.legal_mask(e.current_player)
        assert mask.any()
        a = choose_action(e, baseline, rng)
        assert mask[a]
        _, _, rewards, done, info = e.step(a)
        if info.hand_over:
            hands += 1
            assert sorted(info.hand_finish_order) == [0, 1, 2, 3]
            np.testing.assert_allclose(rewards.sum(), 0, atol=1e-6)
            for place, seat in enumerate(info.hand_finish_order):
                assert rewards[seat] == pytest.approx(1 - 2 * place / 3)
        if done:
            with pytest.raises(ValueError):
                e.step(0)
            break
    else:
        pytest.fail("Match failed to terminate")
    assert hands >= 7


def test_minimum_partition_understands_combinations():
    counts = [0] * 15
    for r in range(5):
        counts[r] = 1
    assert minimum_plays(tuple(counts)) == 1
    counts[14] = 1
    assert minimum_plays(tuple(counts)) == 2


def test_passed_seat_is_skipped_until_trick_clears():
    e = MurlanEnv()
    e.reset()
    e.require_three_spades = False
    e.finished = [False] * 4
    e.passed = [False, True, False, False]
    e.current_combo = Combo("single", (0,))
    e.last_player = 0
    assert e._next_active(0) == 2
    e.passed = [False, False, False, False]
    assert e._next_active(0) == 1


def exact_hand(cards):
    e = MurlanEnv(0)
    e.reset()
    rest = [c for c in range(54) if c not in cards]
    e.hands = [list(cards), rest[::3], rest[1::3], rest[2::3]]
    e.current_player = 0
    e.require_three_spades = False
    return e


def test_manual_selection_plays_exact_suits():
    e = exact_hand([4, 5, 6, 7, 8])
    e.step(17, cards=[6, 7])  # pair of fours: choose hearts/spades, not clubs/diamonds
    assert e.hands[0] == [4, 5, 8]
    assert e.current_combo == Combo('pair', (1,))
    assert e.played_counts[0, 1] == 2


def test_manual_selection_rejects_bad_cards_without_mutation():
    e = exact_hand([0, 3, 4, 5, 6, 7, 8])
    before = deepcopy(e.__dict__)
    for cards in ([4, 4], [4, 8], [4, 50], [4], [4, True]):
        with pytest.raises(ValueError):
            e.step(17, cards=cards)
        assert e.hands == before['hands']
        assert e.last_actions == before['last_actions']
        np.testing.assert_array_equal(e.played_counts, before['played_counts'])
    e.require_three_spades = True
    with pytest.raises(ValueError, match='3S'):
        e.step(1, cards=[0])
    assert 0 in e.hands[0]
    e.step(1, cards=[3])
    assert 3 not in e.hands[0] and 0 in e.hands[0]


def test_manual_straight_preserves_unselected_suits():
    e = exact_hand([0, 3, 4, 5, 8, 10, 12, 15, 16, 18, 52])
    selected = [3, 5, 10, 15, 18]
    e.require_three_spades = True
    action = STRAIGHT_BASE + STRAIGHTS.index((0, 1, 2, 3, 4))
    e.step(action, cards=selected)
    assert e.hands[0] == [0, 4, 8, 12, 16, 52]
    assert e.current_combo == Combo('straight', (0, 1, 2, 3, 4))


def test_manual_exchange_and_lowest_card_fallback():
    for hand, chosen, action in [([4, 5, 6, 7, 8], 7, SWAP_BASE + 1), ([32, 33, 44], 32, SWAP_BASE)]:
        e = exact_hand(hand)
        e.phase = 'swap'
        e.swap_first, e.swap_last, e.pending_starter = 0, 1, 1
        if chosen == 32:
            with pytest.raises(ValueError, match='lowest'):
                e.step(action, cards=[33])
        e.step(action, cards=[chosen])
        assert chosen not in e.hands[0]
        assert chosen in e.hands[1]
        assert e.phase == 'play'
        assert sum(map(len, e.hands)) == 54
