"""Complete-hand collection, opponent diversity, and demonstration pretraining."""

from copy import deepcopy
import numpy as np
import torch
from .baselines import choose_action
from .env import MurlanEnv
from .evaluation import policy_action
from .trajectory import HandTrajectory, rows_to_batch
from .model import masked_categorical


class LeagueCollector:
    def __init__(self, seed=0, n_env=8, pool_size=8):
        if n_env < 1:
            raise ValueError("n_env must be positive")
        self.rng = np.random.default_rng(seed)
        self.envs = [MurlanEnv(seed + i) for i in range(n_env)]
        for env in self.envs:
            env.reset()
        self.pool, self.pool_size = [], pool_size
        self.steps = 0

    def add_snapshot(self, model):
        snapshot = deepcopy(model).eval()
        for p in snapshot.parameters():
            p.requires_grad_(False)
        self.pool.append(snapshot)
        self.pool = self.pool[-self.pool_size :]

    def collect(self, model, min_decisions=2048, temperature=1.0, epsilon=0.03):
        rows = []
        while len(rows) < min_decisions:
            for env in self.envs:
                if env.match_over:
                    env.reset()
                # Mix all-current self-play with one learner against diverse rivals.
                all_current = self.rng.random() < 0.4
                learner_seat = int(self.rng.integers(4))
                rivals = []
                for _ in range(4):
                    if self.pool and self.rng.random() < 0.5:
                        rivals.append(self.pool[int(self.rng.integers(len(self.pool)))])
                    else:
                        rivals.append(
                            str(
                                self.rng.choice(
                                    ["random", "greedy", "strategic"], p=[0.1, 0.3, 0.6]
                                )
                            )
                        )
                trajectory = HandTrajectory()
                for _ in range(2000):
                    actor = env.current_player
                    learn = all_current or actor == learner_seat
                    if learn:
                        obs, legal = env.observe(actor), env.legal_mask(actor)
                        action, logp, value = policy_action(
                            model,
                            env,
                            greedy=False,
                            temperature=temperature,
                            epsilon=epsilon,
                        )
                        trajectory.record(
                            actor, obs, legal, action, logp, value, temperature, epsilon
                        )
                    elif isinstance(rivals[actor], str):
                        action = choose_action(env, rivals[actor], self.rng)
                    else:
                        action = policy_action(rivals[actor], env, greedy=False)[0]
                    _, _, rewards, _, info = env.step(action)
                    trajectory.reward(rewards)
                    self.steps += 1
                    if info.hand_over:
                        rows.extend(trajectory.finish())
                        break
                else:
                    raise RuntimeError("Training hand did not terminate")
                if len(rows) >= min_decisions:
                    break
        return rows_to_batch(rows)


def teacher_examples(hands=64, seed=0):
    rows = []
    env = MurlanEnv(seed)
    env.reset()
    for _ in range(hands):
        if env.match_over:
            env.reset()
        for _ in range(2000):
            p = env.current_player
            action = choose_action(env, "strategic")
            rows.append(
                dict(
                    obs=env.observe(p), legal=env.legal_mask(p), act=action, weight=1.0
                )
            )
            if env.step(action)[4].hand_over:
                break
        else:
            raise RuntimeError("Teacher hand did not terminate")
    return rows


def imitate(model, rows, epochs=2, lr=1e-4, batch_size=256, seed=0):
    """Supervised learning only; human/teacher actions never enter PPO ratios."""
    if not rows or epochs <= 0:
        return 0.0
    device = next(model.parameters()).device
    obs = torch.tensor(
        np.asarray([r["obs"] for r in rows]), dtype=torch.float32, device=device
    )
    legal = torch.tensor(
        np.asarray([r["legal"] for r in rows]), dtype=torch.bool, device=device
    )
    act = torch.tensor([r["act"] for r in rows], dtype=torch.long, device=device)
    weights = torch.tensor(
        [r.get("weight", 1.0) for r in rows], dtype=torch.float32, device=device
    )
    if not bool(legal.gather(1, act[:, None]).all()):
        raise ValueError("Illegal demonstration")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    total, count = 0.0, 0
    model.train()
    for _ in range(epochs):
        for mb in np.array_split(
            rng.permutation(len(rows)), max(1, int(np.ceil(len(rows) / batch_size)))
        ):
            logits, _ = model(obs[mb])
            loss = -(
                masked_categorical(logits, legal[mb]).log_prob(act[mb]) * weights[mb]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            count += 1
    model.eval()
    return total / max(count, 1)
