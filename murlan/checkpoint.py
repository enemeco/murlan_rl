"""Versioned, atomic checkpoints; explicit migration of the original 43 inputs."""

from pathlib import Path
import os
import tempfile
import hashlib
import torch
from .env import MurlanEnv
from .model import PolicyValueNet

SCHEMA_VERSION = 2


def policy_fingerprint(model):
    """Detect manual checkpoint replacements even when version numbers collide."""
    digest = hashlib.sha256(str(SCHEMA_VERSION).encode())
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    old = ckpt["model"]
    old_dim = int(ckpt["obs_dim"])
    dim = len(MurlanEnv().reset()[0])
    if old_dim not in (43, dim):
        raise ValueError(f"Unsupported observation schema: {old_dim}")
    hidden = old["backbone.0.weight"].shape[0]
    model = PolicyValueNet(dim, hidden).to(device)
    if old_dim == 43 and dim != old_dim:
        weight = torch.zeros(hidden, dim, device=device)
        weight[:, :43] = old["backbone.0.weight"]
        old = dict(old, **{"backbone.0.weight": weight})
    missing, unexpected = model.load_state_dict(old, strict=False)
    if set(missing) - {"action_query.weight"} or unexpected:
        raise ValueError(f"Incompatible checkpoint: {missing}, {unexpected}")
    model.eval()
    return model, ckpt


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        obs_dim=model.obs_dim,
        hidden=model.hidden,
        schema_version=SCHEMA_VERSION,
        **metadata,
    )
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
