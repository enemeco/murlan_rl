import numpy as np
import pytest
import torch
from murlan.ppo import compute_gae, PPOTrainer, PPOConfig
from murlan.trajectory import HandTrajectory, rows_to_batch
from murlan.model import PolicyValueNet, masked_categorical
from murlan.env import MurlanEnv
from murlan.checkpoint import load_model, save_checkpoint
from murlan.training import LeagueCollector
from murlan.evaluation import promotion_decision
from murlan.actions import ACTION_DIM

torch.set_num_threads(1)


def test_terminal_reward_propagates_without_crossing_episodes():
    adv, ret = compute_gae(
        np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([0, 1, 1]), 1.0, 1.0
    )
    np.testing.assert_allclose(adv, [3, 2, 3])
    np.testing.assert_allclose(ret, adv)
    adv, ret = compute_gae(
        np.array([1.0]), np.array([2.0]), np.array([0]), 0.9, 0.95, bootstrap_value=3.0
    )
    np.testing.assert_allclose(ret, [3.7])


def test_seat_credit_receives_rewards_from_others_terminal_turn():
    t = HandTrajectory()
    obs = np.zeros(2)
    legal = np.ones(ACTION_DIM, dtype=bool)
    for s in (0, 1, 2):
        t.record(s, obs + s, legal, 1, 0, 0)
    t.record(0, obs, legal, 2, 0, 0)
    t.reward(np.array([1.0, 1 / 3, -1 / 3, -1]))
    rows = t.finish(gamma=1, lam=1)
    assert [r["ret"] for r in rows] == pytest.approx([1, 1, 1 / 3, -1 / 3])
    assert [r["done"] for r in rows] == [0, 1, 1, 1]


def test_exploration_is_masked_and_reconstructs_behavior_probability():
    logits = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    legal = torch.tensor([[True, False, True, False]])
    d = masked_categorical(logits, legal, temperature=1.2, epsilon=0.2)
    expected = 0.8 * torch.softmax(torch.tensor([2.0, 5.0]) / 1.2, 0) + 0.1
    torch.testing.assert_close(d.probs[0, [0, 2]], expected)
    assert set(d.sample((1000,)).flatten().tolist()) == {0, 2}
    with pytest.raises(ValueError):
        masked_categorical(logits, torch.zeros_like(legal))


def test_legacy_checkpoint_migration_preserves_policy_exactly():
    model, ckpt = load_model("checkpoints/murlan_policy.pt")
    if ckpt["obs_dim"] != 43:
        pytest.skip("Requires original repository checkpoint")
    env = MurlanEnv(4)
    env.reset()
    obs = torch.from_numpy(env.observe(env.current_player))[None]
    weights = ckpt["model"]
    x = obs[:, :43]
    for i in (0, 2):
        x = torch.tanh(
            torch.nn.functional.linear(
                x, weights[f"backbone.{i}.weight"], weights[f"backbone.{i}.bias"]
            )
        )
    expected = torch.nn.functional.linear(x, weights["pi.weight"], weights["pi.bias"])
    torch.testing.assert_close(model(obs)[0], expected, atol=1e-5, rtol=1e-5)


def test_rollouts_and_ppo_update_are_finite_and_checkpoint_roundtrips(tmp_path):
    model = PolicyValueNet(249, hidden=32)
    collector = LeagueCollector(seed=0, n_env=2)
    collector.add_snapshot(model)
    batch = collector.collect(model, min_decisions=64, temperature=1.1, epsilon=0.07)
    assert np.isfinite(batch["adv"]).all()
    stats = PPOTrainer(model, PPOConfig(epochs=2, minibatch_size=32)).update(batch)
    assert stats["minibatches"] > 0
    assert all(np.isfinite(v) for v in stats.values())
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(path, model, version=7)
    restored, ckpt = load_model(path)
    assert ckpt["version"] == 7
    x = torch.from_numpy(batch["obs"][:2])
    torch.testing.assert_close(restored(x)[0], model(x)[0])
    with pytest.raises(ValueError):
        PPOTrainer(model, PPOConfig()).update({"act": [1]})


def test_promotion_requires_positive_evidence_and_no_opponent_regression():
    old = [
        dict(opponent=o, seed=s, points=1.0)
        for o in ("greedy", "strategic")
        for s in range(12)
    ]
    new = [dict(r, points=1.3) for r in old]
    assert promotion_decision(new, old)["promote"]
    assert not promotion_decision(old, old)["promote"]
    new = [dict(r, points=0.8 if r["opponent"] == "strategic" else 2.5) for r in old]
    assert not promotion_decision(new, old)["promote"]
    assert not promotion_decision(old[:3], old[:3])["promote"]
