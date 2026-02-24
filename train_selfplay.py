"""
Self-play PPO training for Murlan.

1) Simulate many matches between 4 agents that share the same policy
2) Save the trained model

Run:
  python train_selfplay.py --steps 500000 --n_env 16

Output:
  checkpoints/murlan_policy.pt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

from murlan.env import MurlanEnv
from murlan.model import PolicyValueNet, masked_categorical
from murlan.ppo import PPOConfig, PPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000, help="environment steps (approx)")
    ap.add_argument("--n_env", type=int, default=16, help="number of parallel env instances")
    ap.add_argument("--rollout", type=int, default=8192, help="transitions per PPO update")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = [MurlanEnv(seed=args.seed + i) for i in range(args.n_env)]
    obs0, mask0 = envs[0].reset(seed=args.seed)
    obs_dim = int(obs0.shape[0])

    model = PolicyValueNet(obs_dim=obs_dim, hidden=256)
    cfg = PPOConfig(lr=3e-4, epochs=4, minibatch_size=512, gamma=0.995, gae_lambda=0.95, ent_coef=0.01)
    trainer = PPOTrainer(model, cfg, device=args.device)

    os.makedirs("checkpoints", exist_ok=True)

    states = []
    for i, env in enumerate(envs):
        obs, mask = env.reset(seed=args.seed + 1000 + i)
        states.append((obs, mask))

    total_steps = 0
    update_i = 0

    while total_steps < args.steps:
        batch = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": [], "legal": []}

        while len(batch["act"]) < args.rollout and total_steps < args.steps:
            for ei, env in enumerate(envs):
                obs, mask = states[ei]

                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
                legal_t = torch.from_numpy(mask.astype(np.bool_)).unsqueeze(0).to(args.device)

                with torch.no_grad():
                    logits, v = model(obs_t)
                    dist = masked_categorical(logits, legal_t)
                    a = dist.sample()
                    logp = dist.log_prob(a)

                act = int(a.item())
                obs2, mask2, reward_vec, done, info = env.step(act)
                r = float(reward_vec[info.last_actor])

                batch["obs"].append(obs)
                batch["legal"].append(mask)
                batch["act"].append(act)
                batch["logp"].append(float(logp.item()))
                batch["val"].append(float(v.item()))
                batch["rew"].append(r)
                batch["done"].append(1.0 if done else 0.0)

                total_steps += 1

                if done:
                    obs2, mask2 = env.reset(seed=None)

                states[ei] = (obs2, mask2)

                if len(batch["act"]) >= args.rollout or total_steps >= args.steps:
                    break

        update_i += 1
        stats = trainer.update({k: np.asarray(v) for k, v in batch.items()})
        print(
            f"update {update_i:04d} | steps {total_steps} | "
            f"loss {stats['loss']:.4f} | pi {stats['pi_loss']:.4f} | vf {stats['vf_loss']:.4f} | ent {stats['ent']:.3f}"
        )

        if update_i % args.save_every == 0:
            ckpt = {"model": model.state_dict(), "obs_dim": obs_dim, "ppo": asdict(cfg), "steps": total_steps}
            torch.save(ckpt, "checkpoints/murlan_policy.pt")

    ckpt = {"model": model.state_dict(), "obs_dim": obs_dim, "ppo": asdict(cfg), "steps": total_steps}
    torch.save(ckpt, "checkpoints/murlan_policy.pt")
    print("Saved checkpoints/murlan_policy.pt")


if __name__ == "__main__":
    main()