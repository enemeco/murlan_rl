from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from murlan.env import MurlanEnv
from murlan.actions import pretty_action
from murlan.cards import (
    BLACK_JOKER_ID,
    RED_JOKER_ID,
    SUITS,
    RANKS,
    card_rank_i,
    card_suit_i,
)
from murlan.model import PolicyValueNet, masked_categorical


def card_to_str(card_id: int) -> str:
    if card_id == BLACK_JOKER_ID:
        return "BJ"
    if card_id == RED_JOKER_ID:
        return "RJ"
    r = RANKS[card_rank_i(card_id)]
    s = SUITS[card_suit_i(card_id)]
    return f"{r}{s}"


@dataclass
class GameSession:
    env: MurlanEnv
    human_seat: int
    done: bool
    bot_traj: List[dict]


class MurlanBot:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.obs_dim = int(ckpt["obs_dim"])
        self.model = PolicyValueNet(obs_dim=self.obs_dim, hidden=256).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.lock = threading.RLock()

    @torch.no_grad()
    def act(self, obs: np.ndarray, legal_mask: np.ndarray, greedy: bool = True) -> int:
        with self.lock:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            legal_t = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0).to(self.device)
            logits, _ = self.model(obs_t)
            dist = masked_categorical(logits, legal_t)
            if greedy:
                return int(dist.probs.argmax(dim=-1).item())
            return int(dist.sample().item())

    @torch.no_grad()
    def act_with_stats(self, obs: np.ndarray, legal_mask: np.ndarray, greedy: bool = True):
        """Return (action, logp, value) under current policy."""
        with self.lock:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            legal_t = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0).to(self.device)
            logits, v = self.model(obs_t)
            dist = masked_categorical(logits, legal_t)
            a_t = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logp = dist.log_prob(a_t)
            return int(a_t.item()), float(logp.item()), float(v.item())


class GameManager:
    def __init__(self, bot: MurlanBot, learner=None):
        self.bot = bot
        self.learner = learner
        self.games: Dict[str, GameSession] = {}

    def new_game(self, seed: Optional[int] = None, human_seat: int = 0) -> str:
        env = MurlanEnv(seed=seed or 0)
        env.reset(seed=seed)
        gid = secrets.token_urlsafe(16)
        self.games[gid] = GameSession(env=env, human_seat=human_seat, done=False, bot_traj=[])
        self._autoplay_until_human(gid)
        return gid

    def get_state(self, gid: str) -> Dict:
        sess = self.games[gid]
        env = sess.env
        human = sess.human_seat

        mask = env.legal_mask(human) if env.current_player == human else np.zeros_like(env.legal_mask(env.current_player))
        state = {
            "game_id": gid,
            "phase": env.phase,
            "current_player": env.current_player,
            "human_seat": human,
            "scores": env.scores,
            "goal_target": env.goal_target,
            "hand_sizes": [len(h) for h in env.hands],
            "your_hand": [card_to_str(c) for c in env.hands[human]],
            "trick": None if env.current_combo is None else {
                "kind": env.current_combo.kind,
                "size": env.current_combo.size,
                "primary_rank": int(env.current_combo.primary_rank),
            },
            "passed": env.passed,
            "done": sess.done,
            "your_turn": (env.current_player == human and not sess.done),
            "legal_actions": [i for i, ok in enumerate(mask.tolist()) if ok],
            "legal_action_names": [pretty_action(i) for i, ok in enumerate(mask.tolist()) if ok],
        }
        return state

    def human_act(self, gid: str, action: int) -> Dict:
        sess = self.games[gid]
        env = sess.env
        human = sess.human_seat

        if sess.done:
            return self.get_state(gid)

        if env.current_player != human:
            self._autoplay_until_human(gid)
            return self.get_state(gid)

        _, _, _, done, _ = env.step(int(action))
        if done:
            sess.done = True
            if self.learner is not None:
                self.learner.submit_game(sess.bot_traj)
            return self.get_state(gid)

        self._autoplay_until_human(gid)
        return self.get_state(gid)

    def _autoplay_until_human(self, gid: str, max_steps: int = 5000):
        sess = self.games[gid]
        env = sess.env
        human = sess.human_seat

        steps = 0
        while (not sess.done) and env.current_player != human:
            actor = env.current_player
            obs = env.observe(actor)
            mask = env.legal_mask(actor)

            a, logp, val = self.bot.act_with_stats(obs, mask, greedy=True)
            _, _, reward_vec, done, _ = env.step(a)

            sess.bot_traj.append({
                "obs": obs.astype(np.float32),
                "legal": mask.astype(np.bool_),
                "act": int(a),
                "logp": float(logp),
                "val": float(val),
                "rew": float(reward_vec[actor]),
                "done": 1.0 if done else 0.0,
            })

            if done:
                sess.done = True
                if self.learner is not None:
                    self.learner.submit_game(sess.bot_traj)
                break

            steps += 1
            if steps > max_steps:
                sess.done = True
                break