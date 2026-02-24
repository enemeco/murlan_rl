from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model import masked_categorical


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 512


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
        if dones[t] == 1.0:
            next_value = 0.0
            last_gae = 0.0

    returns = adv + values
    return adv, returns


class PPOTrainer:
    def __init__(self, model: nn.Module, config: PPOConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = batch["obs"].astype(np.float32)
        act = batch["act"].astype(np.int64)
        old_logp = batch["logp"].astype(np.float32)
        old_val = batch["val"].astype(np.float32)
        rew = batch["rew"].astype(np.float32)
        done = batch["done"].astype(np.float32)
        legal = batch["legal"].astype(np.bool_)

        adv, ret = compute_gae(rew, old_val, done, self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(obs).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        old_logp_t = torch.from_numpy(old_logp).to(self.device)
        adv_t = torch.from_numpy(adv).to(self.device)
        ret_t = torch.from_numpy(ret).to(self.device)
        legal_t = torch.from_numpy(legal).to(self.device)

        N = obs.shape[0]
        idx = np.arange(N)

        stats = {"loss": 0.0, "pi_loss": 0.0, "vf_loss": 0.0, "ent": 0.0}

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                logits, v = self.model(obs_t[mb])
                dist = masked_categorical(logits, legal_t[mb])

                logp = dist.log_prob(act_t[mb])
                ent = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                vf_loss = 0.5 * (ret_t[mb] - v).pow(2).mean()
                loss = pi_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                stats["loss"] += float(loss.item())
                stats["pi_loss"] += float(pi_loss.item())
                stats["vf_loss"] += float(vf_loss.item())
                stats["ent"] += float(ent.item())

        denom = self.cfg.epochs * max(1, (N + self.cfg.minibatch_size - 1) // self.cfg.minibatch_size)
        for k in stats:
            stats[k] /= denom
        return stats