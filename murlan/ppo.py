from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .model import masked_categorical


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.02
    value_clip: float = 0.2


def compute_gae(rewards, values, dones, gamma, lam, bootstrap_value=0.0):
    """Single-seat sequence; dones cut *future* credit, not terminal rewards.

    For an unfinished sequence, supply the value of that same seat's next
    decision. The collectors use complete hands, so bootstrap is zero.
    """
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("GAE inputs must have equal lengths")
    adv = np.zeros(len(rewards), dtype=np.float32)
    next_value, gae = float(bootstrap_value), 0.0
    for t in reversed(range(len(rewards))):
        alive = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * alive - values[t]
        gae = delta + gamma * lam * alive * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values


class PPOTrainer:
    def __init__(self, model, config, device="cpu"):
        self.model, self.cfg, self.device = model.to(device), config, device
        self.opt = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    def update(self, batch):
        # Never infer temporal adjacency from shuffled/multi-player rows.
        if "adv" not in batch or "ret" not in batch:
            raise ValueError("Compute per-seat advantages before merging trajectories")
        n = len(batch["act"])
        if not n:
            raise ValueError("Empty rollout")
        tensors = {}
        for key in [
            "obs",
            "act",
            "logp",
            "val",
            "legal",
            "adv",
            "ret",
            "temperature",
            "epsilon",
        ]:
            value = batch.get(key, np.full(n, 1.0 if key == "temperature" else 0.0))
            dtype = (
                torch.long
                if key == "act"
                else torch.bool if key == "legal" else torch.float32
            )
            tensors[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        t = tensors
        if not bool(t["legal"].gather(1, t["act"][:, None]).all()):
            raise ValueError("Rollout contains illegal actions")
        if not all(
            bool(torch.isfinite(t[k]).all())
            for k in ["obs", "logp", "val", "adv", "ret"]
        ):
            raise ValueError("Non-finite rollout")
        adv = t["adv"]
        t["adv"] = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        stats = dict(
            loss=0.0,
            pi_loss=0.0,
            vf_loss=0.0,
            ent=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
        )
        updates = 0
        stopped = False
        self.model.train()
        for _ in range(self.cfg.epochs):
            for mb in torch.randperm(n, device=self.device).split(
                self.cfg.minibatch_size
            ):
                logits, val = self.model(t["obs"][mb])
                dist = masked_categorical(
                    logits, t["legal"][mb], t["temperature"][mb], t["epsilon"][mb]
                )
                logratio = dist.log_prob(t["act"][mb]) - t["logp"][mb]
                ratio = logratio.exp()
                kl = ((ratio - 1) - logratio).mean()
                if self.cfg.target_kl and float(kl.detach()) > self.cfg.target_kl * 1.5:
                    stopped = True
                    break
                policy_loss = -torch.minimum(
                    ratio * t["adv"][mb],
                    ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                    * t["adv"][mb],
                ).mean()
                clipped_val = t["val"][mb] + (val - t["val"][mb]).clamp(
                    -self.cfg.value_clip, self.cfg.value_clip
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (val - t["ret"][mb]).square(),
                        (clipped_val - t["ret"][mb]).square(),
                    ).mean()
                )
                # Normalize by legal-action count so many-action leads do not dominate.
                entropy = dist.entropy() / t["legal"][mb].sum(
                    -1
                ).float().log().clamp_min(1.0)
                ent = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - self.cfg.ent_coef * ent
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite PPO loss")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.opt.step()
                values = [
                    loss,
                    policy_loss,
                    value_loss,
                    ent,
                    kl,
                    ((ratio - 1).abs() > self.cfg.clip_eps).float().mean(),
                ]
                for key, value in zip(stats, values):
                    stats[key] += float(value.detach())
                updates += 1
            if stopped:
                break
        stats = {k: v / max(updates, 1) for k, v in stats.items()}
        variance = float(t["ret"].var(unbiased=False))
        stats.update(
            early_stop=stopped,
            minibatches=updates,
            explained_variance=(
                1 - float((t["ret"] - t["val"]).var(unbiased=False)) / variance
                if variance > 1e-8
                else 0.0
            ),
        )
        self.model.eval()
        return stats
