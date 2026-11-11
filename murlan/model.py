from __future__ import annotations
import torch
import torch.nn as nn
from .actions import ACTION_DIM
from .features import ACTION_FEATURES


class PolicyValueNet(nn.Module):
    """Masked policy with a shared action-semantic head and state-value baseline."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.hidden = obs_dim, hidden
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi = nn.Linear(hidden, ACTION_DIM)
        self.v = nn.Linear(hidden, 1)
        self.action_query = nn.Linear(hidden, ACTION_FEATURES.shape[1], bias=False)
        self.register_buffer(
            "action_features",
            torch.from_numpy(ACTION_FEATURES.copy()),
            persistent=False,
        )
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1)
        nn.init.zeros_(self.v.bias)
        nn.init.zeros_(self.action_query.weight)

    def forward(self, obs):
        x = self.backbone(obs)
        return self.pi(x) + self.action_query(x) @ self.action_features.T, self.v(
            x
        ).squeeze(-1)


def masked_categorical(logits, legal_mask, temperature=1.0, epsilon=0.0):
    """Exact behavior distribution, including legal-only epsilon exploration.

    Keep temperature/epsilon in each rollout and reuse during PPO updates.
    """
    legal_mask = legal_mask.bool()
    if not bool(legal_mask.any(dim=-1).all()):
        raise ValueError("Every state must have at least one legal action")
    temperature = torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype)
    epsilon = torch.as_tensor(epsilon, device=logits.device, dtype=logits.dtype)
    if bool((temperature <= 0).any()) or bool(((epsilon < 0) | (epsilon > 1)).any()):
        raise ValueError("Invalid exploration parameters")
    if temperature.ndim == 1:
        temperature = temperature.unsqueeze(-1)
    if epsilon.ndim == 1:
        epsilon = epsilon.unsqueeze(-1)
    masked = (logits / temperature).masked_fill(
        ~legal_mask, torch.finfo(logits.dtype).min
    )
    probs = torch.softmax(masked, dim=-1)
    uniform = legal_mask.to(logits.dtype) / legal_mask.sum(-1, keepdim=True)
    return torch.distributions.Categorical(
        probs=(1 - epsilon) * probs + epsilon * uniform
    )
