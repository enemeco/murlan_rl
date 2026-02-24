from __future__ import annotations

import torch
import torch.nn as nn

from .actions import ACTION_DIM


class PolicyValueNet(nn.Module):
    """Simple MLP policy + value network for Murlan."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, ACTION_DIM)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


def masked_categorical(logits: torch.Tensor, legal_mask: torch.Tensor):
    """torch.distributions.Categorical with illegal actions masked out."""
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(legal_mask, logits, neg_inf)
    return torch.distributions.Categorical(logits=masked_logits)