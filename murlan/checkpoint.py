"""Atomic checkpoints with observation and Kolor action-space migration."""

from pathlib import Path
import os
import tempfile
import hashlib
import torch
from .env import MurlanEnv
from .model import PolicyValueNet
from .actions import ACTION_DIM, STRAIGHT_BASE, STRAIGHTS, SWAP_BASE, SWAP_N, action_desc

SCHEMA_VERSION = 5


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
    old_action_dim = old["pi.weight"].shape[0]
    if old_action_dim != ACTION_DIM:
        if old_action_dim not in (100, 101, 109) or ACTION_DIM != 118:
            raise ValueError(f"Unsupported action schema: {old_action_dim}")
        # Accept original weights and both previous Kolor fixes.
        # Existing straight IDs stay put; the eight swap outputs move to 110..117.
        # Warm-start new outputs from related runs; rebuild shared action features.
        old = dict(old)
        old_swap_base = old_action_dim - SWAP_N
        for key in ("pi.weight", "pi.bias"):
            source = old[key]
            expanded = source.new_empty((ACTION_DIM, *source.shape[1:]))
            expanded[:old_swap_base] = source[:old_swap_base]
            expanded[SWAP_BASE:] = source[old_swap_base:]
            for action in range(old_swap_base, SWAP_BASE):
                seq = action_desc(action).payload
                length = len(seq)
                if seq[:2] == (11, 12):
                    # New A-low actions follow every 2-low action. These peers
                    # have already been copied/migrated even for 100-action files.
                    peer = STRAIGHT_BASE + STRAIGHTS.index((12, *range(min(length, 13) - 1)))
                    expanded[action] = expanded[peer]
                elif length == 5:
                    low_ace = STRAIGHT_BASE + STRAIGHTS.index((11, 12, 0, 1, 2))
                    expanded[action] = (source[STRAIGHT_BASE] + source[low_ace]) / 2
                else:
                    # Use the 3-low run of equal length, or the 12-card run
                    # for the new 13-card action, which had no equal-length peer.
                    peer = STRAIGHT_BASE + STRAIGHTS.index(tuple(range(min(length, 12))))
                    expanded[action] = source[peer]
            old[key] = expanded
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
        action_dim=ACTION_DIM,
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
