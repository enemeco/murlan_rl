from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from app.game_manager import GameManager, MurlanBot
from app.replay import ReplayStore
from app.main import create_app
from app.online_learning import OnlineLearner
from murlan.baselines import choose_action

torch.set_num_threads(1)


@pytest.fixture
def manager(tmp_path):
    return GameManager(
        MurlanBot("checkpoints/murlan_policy.pt"),
        replay=ReplayStore(tmp_path / "replay.db"),
    )


def human_action(manager, gid):
    session = manager.games[gid]
    return choose_action(session.env, "strategic")


def finish_a_hand(manager, gid):
    for step in range(100):
        state = manager.get_state(gid)
        manager.human_act(
            gid, human_action(manager, gid), state["revision"], f"test-{step:06}"
        )
        if manager.replay.stats()["completed_hands"]:
            return
    pytest.fail("Hand did not finish")


def test_human_actions_persist_immediately_and_completed_hands_reward_all_bots(manager):
    gid = manager.new_game(seed=12)
    first = manager.get_state(gid)
    manager.human_act(
        gid, human_action(manager, gid), first["revision"], "request-0001"
    )
    assert ReplayStore(manager.replay.path).stats()["human_moves"] == 1
    assert manager.replay.demonstrations()[0]["weight"] > 0
    finish_a_hand(manager, gid)
    ids, rows = manager.replay.pending(0)
    assert ids and rows
    terminals = [r for r in rows if r["done"]]
    assert len(terminals) == 3
    assert all(abs(r["rew"]) >= 1 / 3 - 1e-6 for r in terminals)
    assert "hands" not in manager.get_state(gid)


def test_double_click_and_stale_turn_cannot_play_twice(manager):
    gid = manager.new_game(seed=1)
    state = manager.get_state(gid)
    a = human_action(manager, gid)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda _: manager.human_act(gid, a, state["revision"], "same-request"),
                range(2),
            )
        )
    assert results[0]["revision"] == results[1]["revision"]
    assert manager.replay.stats()["human_moves"] == 1
    with pytest.raises(ValueError):
        manager.human_act(gid, a, state["revision"], "new-request")
    with pytest.raises(ValueError):
        manager.human_act(gid, -1)


def test_opt_out_and_competitive_modes(manager):
    gid = manager.new_game(seed=9, learn=False)
    state = manager.get_state(gid)
    manager.human_act(gid, human_action(manager, gid), state["revision"], "no-learning")
    assert manager.replay.stats()["total_moves"] == 0
    gid = manager.new_game(seed=11, mode="competitive")
    finish_a_hand(manager, gid)
    ids, rows = manager.replay.pending(0)
    assert ids and not rows  # greedy actions never enter on-policy PPO
    assert manager.replay.demonstrations()


def test_snapshot_is_frozen_during_a_hand(manager):
    gid = manager.new_game(seed=1)
    session = manager.games[gid]
    before = {k: v.clone() for k, v in session.model.state_dict().items()}
    with torch.no_grad():
        for p in manager.bot.model.parameters():
            p.add_(0.1)
    for k, v in session.model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_online_rejected_candidate_does_not_change_champion(
    manager, monkeypatch, tmp_path
):
    gid = manager.new_game(seed=8)
    finish_a_hand(manager, gid)
    manager.bot.ckpt_path = str(tmp_path / "champion.pt")
    learner = OnlineLearner(
        manager.bot,
        manager.replay,
        min_hands=1,
        min_decisions=1,
        eval_deals=8,
        start=False,
    )

    def fixed_evaluate(model, seeds, **kwargs):
        return [
            dict(opponent="strategic", seed=s, points=1.5, win_rate=0.25) for s in seeds
        ]

    monkeypatch.setattr("app.online_learning.evaluate", fixed_evaluate)
    before = {k: v.clone() for k, v in manager.bot.model.state_dict().items()}
    details = learner.train_once()
    assert not details["gate"]["promote"]
    assert manager.bot.version == 0
    assert manager.replay.stats()["pending_hands"] == 0
    assert Path(tmp_path / "champion.candidate.pt").exists()
    assert not Path(manager.bot.ckpt_path).exists()
    for k, v in manager.bot.model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_api_errors_fallback_and_disabled_download(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DOWNLOAD_KEY", raising=False)
    app = create_app(
        model_path=str(tmp_path / "missing.pt"),
        replay_path=str(tmp_path / "replay.db"),
        online=False,
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/health").json()["ok"]
        assert client.get("/download-model").status_code == 401
        assert client.get("/api/state/missing").status_code == 404
        state = client.post("/api/new", json={}).json()
        assert state["bot_kind"] == "strategic"
        assert state["your_turn"]
        gid = state["game_id"]
        for bad in [-1, ACTION_DIM, True, "1"]:
            assert (
                client.post(
                    f"/api/act/{gid}",
                    json=dict(
                        action=bad,
                        revision=state["revision"],
                        request_id="api-test-000",
                    ),
                ).status_code
                == 422
            )
        result = client.post(
            f"/api/act/{gid}",
            json=dict(
                action=state["legal_actions"][0],
                revision=state["revision"],
                request_id="api-test-001",
            ),
        )
        assert result.status_code == 200
        assert client.get("/api/learning").json()["human_moves"] == 1
        assert client.delete(f"/api/game/{gid}").status_code == 200
        assert client.get(f"/api/state/{gid}").status_code == 404


from murlan.actions import ACTION_DIM


def test_online_promotion_is_atomic_and_survives_restart(
    manager, monkeypatch, tmp_path
):
    gid = manager.new_game(seed=18)
    finish_a_hand(manager, gid)
    manager.bot.ckpt_path = str(tmp_path / "champion.pt")
    learner = OnlineLearner(
        manager.bot,
        manager.replay,
        min_hands=1,
        min_decisions=1,
        eval_deals=8,
        start=False,
    )
    calls = []

    def scores(model, seeds, **kwargs):
        calls.append(1)
        return [
            dict(
                opponent="strategic",
                seed=s,
                points=2.0 if len(calls) == 1 else 1.5,
                win_rate=0.5,
            )
            for s in seeds
        ]

    monkeypatch.setattr("app.online_learning.evaluate", scores)
    details = learner.train_once()
    assert details["gate"]["promote"]
    assert manager.bot.version == 1
    restored = MurlanBot(manager.bot.ckpt_path)
    assert restored.version == 1
    assert manager.games[gid].version == 0  # in-flight hand keeps its old snapshot
    assert manager.replay.stats()["promotions"] == 1
    new = manager.new_game(seed=19)
    assert manager.games[new].version == 1
    for key, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, manager.bot.model.state_dict()[key])


def test_checkpoint_identity_prevents_off_policy_version_collisions(manager):
    gid = manager.new_game(seed=22)
    finish_a_hand(manager, gid)
    old_id = manager.bot.policy_id
    ids, rows = manager.replay.pending(0, policy_id=old_id)
    assert ids and rows
    manager.replay.mark_stale(0, "different-checkpoint-same-version")
    assert manager.replay.pending(0, policy_id=old_id) == ([], [])
    assert manager.replay.demonstrations()  # human data remains useful off-policy


@pytest.mark.parametrize("manual", [False, True])
def test_full_human_vs_three_bot_match_finishes(manager, manual):
    gid = manager.new_game(seed=33)
    for turn in range(1800):
        state = manager.get_state(gid)
        if state["done"]:
            assert state["winner"] in range(4)
            assert not state["legal_actions"]
            assert manager.replay.stats()["completed_hands"] >= 7
            return
        assert state["your_turn"]
        action = human_action(manager, gid)
        cards = None
        if manual:
            option = next(a for a in state['actions'] if a['id'] == action)
            sizes = {'pass': 0, 'single': 1, 'pair': 2, 'triple': 3, 'quad': 4, 'swap_give': 1}
            ranks = option['ranks'] if option['kind'] == 'straight' else option['ranks'] * sizes[option['kind']]
            available = list(reversed(state['your_hand']))
            cards = []
            for rank in ranks:
                if rank == '3' and state['require_three_spades'] and '3S' in available:
                    card = '3S'
                elif option['required_cards']:
                    card = option['required_cards'][0]
                else:
                    card = next(c for c in available if (c if c in ('BJ','RJ') else c[:-1]) == rank)
                cards.append(card)
                available.remove(card)
        manager.human_act(gid, action, state["revision"], f"full-match-{turn}", cards=cards)
    pytest.fail("Human-versus-three-bot match did not terminate")


def rig_human_hand(manager, gid, hand):
    e = manager.games[gid].env
    rest = [c for c in range(54) if c not in hand]
    e.hands = [list(hand), rest[::3], rest[1::3], rest[2::3]]
    e.current_player = 0
    e.require_three_spades = False
    e.phase = 'play'
    e.current_combo = None
    e.passed = [False] * 4
    e.finished = [False] * 4
    e.finish_order = []
    e.last_player = None
    return e


def test_physical_selection_logs_exact_cards_and_is_idempotent(manager, monkeypatch):
    gid = manager.new_game(seed=14)
    monkeypatch.setattr(manager, '_autoplay', lambda *args: None)
    rig_human_hand(manager, gid, [4, 5, 6, 7, 8])
    state = manager.get_state(gid)
    count = manager.replay.stats()['human_moves']
    with pytest.raises(ValueError):
        manager.human_act(gid, 17, state['revision'], 'wrong-cards', cards=['4C', '5C'])
    assert manager.replay.stats()['human_moves'] == count
    result = manager.human_act(gid, 17, state['revision'], 'physical-pair', cards=['4H', '4S'])
    assert result['events'][-1]['cards'] == ['4H', '4S']
    assert result['trick']['cards'] == ['4H', '4S']
    assert result['your_hand'] == ['4C', '4D', '5C']
    repeated = manager.human_act(gid, 17, state['revision'], 'physical-pair', cards=['4S', '4H'])
    assert repeated['revision'] == result['revision']
    assert manager.replay.stats()['human_moves'] == count + 1
    with pytest.raises(ValueError, match='different action'):
        manager.human_act(gid, 17, state['revision'], 'physical-pair', cards=['4C', '4D'])


def test_albanian_names_persist_and_new_match_changes_them(manager):
    from app.player_names import ALBANIAN_NAMES
    gid = manager.new_game(seed=23, human_seat=2)
    names = manager.get_state(gid)['player_names']
    assert names[2] == 'Ti'
    assert len(set(names)) == 4
    robots = set(names) - {'Ti'}
    assert robots <= set(ALBANIAN_NAMES)
    finish_a_hand(manager, gid)
    assert manager.get_state(gid)['player_names'] == names
    # Another concurrent game must not affect the new names for this player's next match.
    manager.new_game(seed=25)
    next_gid = manager.new_game(seed=24, human_seat=2, previous_game_id=gid)
    next_names = manager.get_state(next_gid)['player_names']
    assert next_names[2] == 'Ti'
    assert not robots.intersection(set(next_names) - {'Ti'})


def test_exact_cards_api_and_exchange_fallback(tmp_path, monkeypatch):
    app = create_app(model_path='checkpoints/murlan_policy.pt', replay_path=str(tmp_path/'cards.db'), online=False)
    with TestClient(app) as client:
        state = client.post('/api/new', json={}).json()
        gid = state['game_id']
        manager = app.state.manager
        monkeypatch.setattr(manager, '_autoplay', lambda *args: None)
        e = rig_human_hand(manager, gid, [32, 33, 44])
        e.phase, e.swap_first, e.swap_last, e.pending_starter = 'swap', 0, 1, 1
        state = client.get(f'/api/state/{gid}').json()
        option = state['actions'][0]
        assert option['ranks'] == ['J']
        assert option['required_cards'] == ['JC']
        payload = dict(action=option['id'], revision=state['revision'], request_id='exchange-test', cards=['JD'])
        assert client.post(f'/api/act/{gid}', json=payload).status_code == 409
        payload['cards'] = ['JC']
        result = client.post(f'/api/act/{gid}', json=payload)
        assert result.status_code == 200
        assert 'JC' not in result.json()['your_hand']
        assert 32 in e.hands[1]
