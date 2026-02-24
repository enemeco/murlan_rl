from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch

from murlan.ppo import PPOConfig, PPOTrainer


class OnlineLearner:
    """Background PPO updates from completed human-vs-bot games.

    - single background thread
    - small PPO updates on bot-only trajectories
    - uses a shared lock with the bot so serving & training don't race
    """

    def __init__(self, model, device: str = "cpu", lock=None, ckpt_path: str = "checkpoints/murlan_policy.pt"):
        self.model = model
        self.device = device
        self.model_lock = lock
        self.ckpt_path = ckpt_path

        self.cfg = PPOConfig(lr=1e-4, epochs=2, minibatch_size=256, ent_coef=0.005)
        self.trainer = PPOTrainer(self.model, self.cfg, device=device)

        self._lock = threading.Lock()
        self._queue: List[Dict[str, np.ndarray]] = []
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        self._updates = 0

    def submit_game(self, bot_traj: List[dict]):
        if not bot_traj:
            return
        batch = {k: np.asarray([t[k] for t in bot_traj]) for k in bot_traj[0].keys()}
        with self._lock:
            self._queue.append(batch)

    def _worker(self):
        while not self._stop:
            batch = None
            with self._lock:
                if self._queue:
                    batch = self._queue.pop(0)
            if batch is None:
                time.sleep(0.2)
                continue

            if self.model_lock is None:
                self.trainer.update(batch)
            else:
                with self.model_lock:
                    self.trainer.update(batch)

            self._updates += 1

            if self._updates % 2 == 0:
                if self.model_lock is None:
                    ckpt = {
                        "model": self.model.state_dict(),
                        "obs_dim": int(self.model.backbone[0].in_features) if hasattr(self.model, "backbone") else 0,
                        "ppo": asdict(self.cfg),
                        "updates": self._updates,
                    }
                    torch.save(ckpt, self.ckpt_path)
                else:
                    with self.model_lock:
                        ckpt = {
                            "model": self.model.state_dict(),
                            "obs_dim": int(self.model.backbone[0].in_features) if hasattr(self.model, "backbone") else 0,
                            "ppo": asdict(self.cfg),
                            "updates": self._updates,
                        }
                        torch.save(ckpt, self.ckpt_path)