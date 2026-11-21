"""House-rule regressions: only the next Kolor can answer, including low 2."""
from copy import deepcopy
from collections import Counter
import hashlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.game_manager import GameManager, MurlanBot
from app.online_learning import OnlineLearner
from app.main import create_app
from app.replay import ReplayStore
from murlan.actions import ACTION_DIM, STRAIGHT_BASE, STRAIGHTS, SWAP_BASE, action_desc
from murlan.baselines import minimum_plays
from murlan.cards import Combo, RANK_TO_I, THREE_OF_SPADES_ID, straight_primary_rank
from murlan.checkpoint import load_model, save_checkpoint
from murlan.env import MurlanEnv
from murlan.features import ACTION_COUNTS


FIVE_CARD_CHAIN = [
    'A 2 3 4 5', '2 3 4 5 6', '3 4 5 6 7', '4 5 6 7 8',
    '5 6 7 8 9', '6 7 8 9 10', '7 8 9 10 J', '8 9 10 J Q',
    '9 10 J Q K', '10 J Q K A',
]

TWO_LOW_RUNS = [
    '2 3 4 5 6',
    '2 3 4 5 6 7',
    '2 3 4 5 6 7 8',
    '2 3 4 5 6 7 8 9',
    '2 3 4 5 6 7 8 9 10',
    '2 3 4 5 6 7 8 9 10 J',
    '2 3 4 5 6 7 8 9 10 J Q',
    '2 3 4 5 6 7 8 9 10 J Q K',
    '2 3 4 5 6 7 8 9 10 J Q K A',
]


ACE_LOW_RUNS = [' '.join(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'][:n])
                for n in range(5, 15)]


@pytest.fixture(params=[100, 101, 109])
def legacy_checkpoint(request, tmp_path):
    """Old schemas, even if GitHub has already replaced the bundled checkpoint."""
    model, _ = load_model('checkpoints/improved_policy.pt')
    ckpt = dict(model=model.state_dict(), obs_dim=model.obs_dim, hidden=model.hidden,
                schema_version={100: 2, 101: 3, 109: 4}[request.param],
                action_dim=request.param)
    for key in ('pi.weight', 'pi.bias'):
        source = ckpt['model'][key]
        ckpt['model'][key] = torch.cat([source[:request.param - 8], source[SWAP_BASE:]])
        if request.param > 100:
            ckpt['model'][key][92:request.param - 8] = 0.123
    path = tmp_path / f'legacy-{request.param}.pt'
    torch.save(ckpt, path)
    return path, ckpt


def ranks(text):
    return tuple(RANK_TO_I[r] for r in text.split())


def kolor_action(text):
    return STRAIGHT_BASE + STRAIGHTS.index(ranks(text))


def position(cards):
    env = MurlanEnv()
    env.reset()
    rest = [c for c in range(54) if c not in cards]
    env.hands = [sorted(cards), rest[::3], rest[1::3], rest[2::3]]
    env.current_player = 0
    env.require_three_spades = False
    return env


def legal_kolors(env):
    return {int(a) for a in np.flatnonzero(env.legal_mask(0))
            if action_desc(int(a)).kind == 'straight'}


@pytest.mark.parametrize('index,current', list(enumerate(FIVE_CARD_CHAIN)))
def test_only_the_immediate_successor_can_beat_each_five_card_kolor(index, current):
    # One card of every ordinary rank: all Kolors are available in this hand.
    env = position([4 * r for r in range(13)])
    env.current_combo = Combo('straight', tuple(sorted(ranks(current))))
    expected = {kolor_action(FIVE_CARD_CHAIN[index + 1])} if index < 9 else set()
    assert legal_kolors(env) == expected


def test_skipping_a_kolor_is_rejected_before_any_state_change():
    env = position([4 * r for r in range(13)])
    env.current_combo = Combo('straight', ranks('3 4 5 6 7'))
    before = deepcopy(env.__dict__)
    with pytest.raises(ValueError, match='Illegal action'):
        env.step(kolor_action('5 6 7 8 9'))
    assert env.hands == before['hands']
    assert env.current_combo == before['current_combo']
    assert env.last_actions == before['last_actions']
    env.step(kolor_action('4 5 6 7 8'))
    assert env.current_combo == Combo('straight', ranks('4 5 6 7 8'))


@pytest.mark.parametrize('run', TWO_LOW_RUNS)
@pytest.mark.parametrize('manual', [False, True])
def test_every_two_low_run_is_legal_as_an_opening_and_consumes_exact_cards(manual, run):
    seq = ranks(run)
    selected = [THREE_OF_SPADES_ID if r == 0 else r * 4 for r in seq]
    env = position(selected + [52])
    env.require_three_spades = True
    action = kolor_action(run)
    assert env.legal_mask(0)[action]
    env.step(action, cards=selected if manual else None)
    assert env.hands[0] == [52]
    assert env.current_combo == Combo('straight', seq)
    assert env.current_combo.primary_rank == RANK_TO_I[run.split()[-1]]
    assert not env.require_three_spades
    assert int(env.played_counts.sum()) == len(seq)
    assert sum(map(len, env.hands)) + int(env.played_counts.sum()) == 54


@pytest.mark.parametrize('run', TWO_LOW_RUNS)
def test_every_two_low_run_has_consistent_features_and_baseline_partition(run):
    seq = ranks(run)
    end = RANK_TO_I[run.split()[-1]]
    for ordering in (seq, tuple(sorted(seq)), tuple(reversed(seq))):
        assert straight_primary_rank(ordering) == end
        assert Combo('straight', ordering).primary_rank == end
    counts = tuple(1 if r in seq else 0 for r in range(15))
    assert minimum_plays(counts) == 1
    np.testing.assert_array_equal(ACTION_COUNTS[kolor_action(run)], counts)


@pytest.mark.parametrize('current,successor', [
    ('2 3 4 5 6 7', '3 4 5 6 7 8'),
    ('2 3 4 5 6 7 8', '3 4 5 6 7 8 9'),
    ('2 3 4 5 6 7 8 9', '3 4 5 6 7 8 9 10'),
    ('2 3 4 5 6 7 8 9 10', '3 4 5 6 7 8 9 10 J'),
    ('2 3 4 5 6 7 8 9 10 J', '3 4 5 6 7 8 9 10 J Q'),
    ('2 3 4 5 6 7 8 9 10 J Q', '3 4 5 6 7 8 9 10 J Q K'),
    ('2 3 4 5 6 7 8 9 10 J Q K', '3 4 5 6 7 8 9 10 J Q K A'),
    ('2 3 4 5 6 7 8 9 10 J Q K A', None),
])
def test_long_two_low_runs_only_accept_the_next_equal_length_run(current, successor):
    env = position([4 * r for r in range(13)])
    env.current_combo = Combo('straight', tuple(sorted(ranks(current))))
    assert legal_kolors(env) == ({kolor_action(successor)} if successor else set())


def test_no_duplicate_patterns_jokers_or_high_two_wraparound_straights():
    assert len(STRAIGHTS) == len(set(STRAIGHTS)) == 55
    for seq in STRAIGHTS:
        assert 5 <= len(seq) <= 14
        repeated = {r: n for r, n in Counter(seq).items() if n > 1}
        assert repeated == ({RANK_TO_I['A']: 2} if len(seq) == 14 else {})
        assert not set(seq) & {RANK_TO_I['BJ'], RANK_TO_I['RJ']}
        if RANK_TO_I['2'] in seq:
            assert seq in [ranks(run) for run in TWO_LOW_RUNS + ACE_LOW_RUNS]


def test_every_ace_ending_run_has_no_equal_length_kolor_response():
    all_ranks = TWO_LOW_RUNS[-1].split()
    env = position([4 * r for r in range(13)])
    for start in range(9):  # 2-through-A (13 cards) down to 10-through-A (5).
        seq = ranks(' '.join(all_ranks[start:]))
        env.current_combo = Combo('straight', tuple(sorted(seq)))
        assert not legal_kolors(env)


def test_longer_kolors_keep_equal_length_and_cannot_skip_sequences():
    env = position([4 * r for r in range(13)])
    env.current_combo = Combo('straight', ranks('3 4 5 6 7 8'))
    assert legal_kolors(env) == {kolor_action('4 5 6 7 8 9')}


def test_top_kolor_cannot_wrap_around_but_existing_quad_rule_still_applies():
    env = position([0, 1, 2, 3, 4, 8, 12, 44, 48])
    env.current_combo = Combo('straight', ranks('10 J Q K A'))
    assert set(np.flatnonzero(env.legal_mask(0))) == {0, 42}


def test_api_exposes_new_kolor_and_accepts_manual_cards(tmp_path):
    manager = GameManager(MurlanBot('checkpoints/improved_policy.pt'),
                          replay=ReplayStore(tmp_path / 'replay.db'))
    gid = manager.new_game(seed=1, learn=False)
    session = manager.games[gid]
    session.env = position([48, 3, 4, 8, 12, 52])
    session.env.current_combo = Combo('straight', tuple(sorted(ranks('A 2 3 4 5'))))
    session.env.last_player = 1
    state = manager.get_state(gid)
    option = next(a for a in state['actions'] if a['kind'] == 'straight')
    assert option['ranks'] == ['2', '3', '4', '5', '6']
    expected_event = state['revision'] + 1
    state = manager.human_act(gid, option['id'], state['revision'], 'kolor-test-001',
                              cards=['2C', '3S', '4C', '5C', '6C'])
    played = next(e for e in state['events'] if e['id'] == expected_event)
    assert played['kind'] == 'straight' and played['player'] == 0
    assert played['cards'] == ['2C', '3S', '4C', '5C', '6C']


def test_http_api_accepts_full_thirteen_card_kolor(tmp_path):
    app = create_app(model_path='checkpoints/improved_policy.pt',
                     replay_path=str(tmp_path / 'replay.db'), online=False)
    with TestClient(app) as client:
        state = client.post('/api/new', json={}).json()
        gid = state['game_id']
        session = app.state.manager.games[gid]
        seq = ranks(TWO_LOW_RUNS[-1])
        selected = [THREE_OF_SPADES_ID if r == 0 else r * 4 for r in seq]
        session.env = position(selected + [52])
        session.env.require_three_spades = True
        state = client.get(f'/api/state/{gid}').json()
        action = kolor_action(TWO_LOW_RUNS[-1])
        option = next(a for a in state['actions'] if a['id'] == action)
        assert option['ranks'] == TWO_LOW_RUNS[-1].split()
        assert action in state['legal_actions']
        cards = [r + ('S' if r == '3' else 'C') for r in option['ranks']]
        result = client.post(f'/api/act/{gid}', json=dict(
            action=action, revision=state['revision'], request_id='kolor-13-test', cards=cards))
        assert result.status_code == 200
        body = result.json()
        assert body['your_hand'] == ['BJ']
        played = next(e for e in body['events'] if e['id'] == state['revision'] + 1)
        assert played['cards'] == cards
        assert client.get('/api/learning').json()['human_moves'] == 1


def test_improved_checkpoint_migration_preserves_old_action_outputs(tmp_path, legacy_checkpoint):
    path, _ = legacy_checkpoint
    model, old = load_model(path)
    old_swap_base = old['model']['pi.weight'].shape[0] - 8
    for key in ('pi.weight', 'pi.bias'):
        migrated = model.state_dict()[key]
        torch.testing.assert_close(migrated[:old_swap_base], old['model'][key][:old_swap_base])
        torch.testing.assert_close(migrated[SWAP_BASE:], old['model'][key][old_swap_base:])
    # Every old logit, including the action-semantic head, retains its meaning.
    x = torch.randn(3, 249)
    with torch.no_grad():
        hidden = model.backbone(x)
        preserved_features = torch.cat([model.action_features[:old_swap_base], model.action_features[SWAP_BASE:]])
        expected = torch.nn.functional.linear(hidden, old['model']['pi.weight'], old['model']['pi.bias'])
        expected += model.action_query(hidden) @ preserved_features.T
        actual = model(x)[0]
    torch.testing.assert_close(torch.cat([actual[:, :old_swap_base], actual[:, SWAP_BASE:]], dim=1), expected)
    assert torch.isfinite(actual).all()
    assert actual.shape == (3, ACTION_DIM)
    for rank in range(8):
        assert action_desc(SWAP_BASE + rank).payload == (rank,)
    saved = tmp_path / 'migrated.pt'
    save_checkpoint(saved, model, version=4)
    restored, metadata = load_model(saved)
    assert metadata['schema_version'] == 5 and metadata['action_dim'] == 118
    assert metadata['version'] == 4
    torch.testing.assert_close(restored(x)[0], model(x)[0])


def test_legacy_replay_is_retained_but_not_trained_under_new_rules(tmp_path, legacy_checkpoint):
    path, ckpt = legacy_checkpoint
    bot = MurlanBot(path)
    old = ckpt['model']
    old_action_dim = old['pi.weight'].shape[0]
    digest = hashlib.sha256(str(ckpt['schema_version']).encode())
    for key, value in sorted(old.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    legacy_policy_id = digest.hexdigest()
    assert bot.policy_id != legacy_policy_id
    replay = ReplayStore(tmp_path / 'replay.db')
    for hand in range(4):
        replay.record_action('old-game', hand, hand, 0, True, 0, np.zeros(249),
                             np.ones(old_action_dim, dtype=bool), old_action_dim - 8)
        replay.complete_hand('old-game', hand, 0, [{'old-schema': True}], [0, 1, 2, 3],
                             policy_id=legacy_policy_id)
    assert replay.demonstrations() == []
    learner = OnlineLearner(bot, replay, start=False)
    assert learner.train_once() is None
    assert replay.stats()['completed_hands'] == 4
    assert replay.stats()['pending_hands'] == 0
    replay.record_action('new-game', 0, 0, 0, True, 0, np.zeros(249),
                         np.ones(ACTION_DIM, dtype=bool), kolor_action('2 3 4 5 6'))
    demos = replay.demonstrations()
    assert len(demos) == 1 and demos[0]['act'] == kolor_action('2 3 4 5 6')
    assert replay.stats()['human_moves'] == 5


def test_old_optimizer_resume_requests_weight_migration(tmp_path, monkeypatch, capsys):
    import train_selfplay
    monkeypatch.setattr('sys.argv', ['train_selfplay.py', '--resume',
                        'checkpoints/training_resume.pt', '--output-dir', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        train_selfplay.main()
    assert error.value.code == 2
    assert 'Use --init instead of --resume' in capsys.readouterr().err


def physical_run(seq):
    used = Counter()
    cards = []
    for rank in seq:
        cards.append(THREE_OF_SPADES_ID if rank == 0 else 4 * rank + used[rank])
        used[rank] += 1
    return cards


@pytest.mark.parametrize('run', ACE_LOW_RUNS)
@pytest.mark.parametrize('manual', [False, True])
def test_all_ace_low_runs_consume_exact_cards_and_preserve_comparison(manual, run):
    seq = ranks(run)
    selected = physical_run(seq)
    # Keep physical hands within the maximum deal size (14).
    env = position(selected + ([52] if len(selected) < 14 else []))
    env.require_three_spades = True
    action = kolor_action(run)
    assert env.legal_mask(0)[action]
    _, _, _, _, info = env.step(action, cards=list(reversed(selected)) if manual else None)
    assert env.hands[0] == ([52] if len(selected) < 14 else [])
    assert info.played.ranks == seq
    assert info.played.primary_rank == RANK_TO_I[run.split()[-1]]
    assert int(env.played_counts.sum()) == len(seq)
    assert sum(map(len, env.hands)) + int(env.played_counts.sum()) == 54
    counts = tuple(Counter(seq)[r] for r in range(15))
    np.testing.assert_array_equal(ACTION_COUNTS[action], counts)
    assert minimum_plays(counts) == 1


@pytest.mark.parametrize('run', ACE_LOW_RUNS)
def test_each_ace_low_run_accepts_only_the_next_equal_length_run(run):
    env = position([4 * r for r in range(13)] + [45])
    env.current_combo = Combo('straight', ranks(run))
    length = len(run.split())
    expected = {kolor_action(TWO_LOW_RUNS[length - 5])} if length <= 13 else set()
    assert legal_kolors(env) == expected


def test_full_ace_to_ace_requires_two_distinct_physical_aces():
    action = kolor_action(ACE_LOW_RUNS[-1])
    seq = ranks(ACE_LOW_RUNS[-1])
    one_ace = physical_run(seq[:-1])
    env = position(one_ace + [52])
    before = deepcopy(env.__dict__)
    assert not env.legal_mask(0)[action]
    with pytest.raises(ValueError, match='Illegal action'):
        env.step(action)
    assert env.hands == before['hands'] and env.current_combo == before['current_combo']
    env = position(physical_run(seq))
    duplicate = one_ace + [44]
    before = deepcopy(env.__dict__)
    with pytest.raises(ValueError, match='selected twice'):
        env.step(action, cards=duplicate)
    assert env.hands == before['hands']
    assert env.played_counts.sum() == 0


def test_thirteen_card_ace_low_and_ace_high_remain_distinguishable():
    low, high = ranks(ACE_LOW_RUNS[-2]), ranks(TWO_LOW_RUNS[-1])
    assert Counter(low) == Counter(high)
    assert Combo('straight', low).primary_rank == RANK_TO_I['K']
    assert Combo('straight', high).primary_rank == RANK_TO_I['A']
    for seq, end in [(low, 'K'), (high, 'A')]:
        for manual in (False, True):
            cards = physical_run(seq)
            env = position(cards + [52])
            env.step(STRAIGHT_BASE + STRAIGHTS.index(seq), cards=list(reversed(cards)) if manual else None)
            assert env.current_combo.primary_rank == RANK_TO_I[end]


def test_http_api_plays_fourteen_cards_with_two_aces_and_records_one_move(tmp_path, monkeypatch):
    app = create_app(model_path='checkpoints/improved_policy.pt',
                     replay_path=str(tmp_path / 'replay.db'), online=False)
    with TestClient(app) as client:
        state = client.post('/api/new', json={}).json()
        gid = state['game_id']
        manager = app.state.manager
        monkeypatch.setattr(manager, '_autoplay', lambda *args: None)
        session = manager.games[gid]
        seq = ranks(ACE_LOW_RUNS[-1])
        selected = physical_run(seq)
        session.env = position(selected)
        session.env.require_three_spades = True
        state = client.get(f'/api/state/{gid}').json()
        action = kolor_action(ACE_LOW_RUNS[-1])
        option = next(a for a in state['actions'] if a['id'] == action)
        assert option['ranks'].count('A') == 2 and len(option['ranks']) == 14
        from app.game_manager import card_to_str
        cards = [card_to_str(c) for c in reversed(selected)]
        result = client.post(f'/api/act/{gid}', json=dict(
            action=action, revision=state['revision'], request_id='ace-to-ace-14', cards=cards))
        assert result.status_code == 200
        body = result.json()
        assert body['your_hand'] == [] and body['finished'][0]
        played = next(e for e in body['events'] if e['id'] == state['revision'] + 1)
        assert played['cards'] == cards and len(set(cards)) == 14
        assert body['trick']['size'] == 14
        assert body['trick']['primary_rank'] == RANK_TO_I['A']
        assert client.get('/api/learning').json()['human_moves'] == 1
