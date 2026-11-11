from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import secrets
import threading
import time
import numpy as np
import torch
from murlan.env import MurlanEnv
from murlan.actions import ACTION_DIM, action_desc, pretty_action
from murlan.cards import SUITS, RANKS, card_rank_i, card_suit_i, sort_cards_by_rank
from murlan.model import PolicyValueNet
from murlan.checkpoint import load_model, policy_fingerprint
from murlan.evaluation import policy_action
from murlan.baselines import choose_action
from murlan.trajectory import HandTrajectory
from .replay import ReplayStore
from .player_names import new_player_names


def card_to_str(card):
    if card >= 52:
        return "BJ" if card == 52 else "RJ"
    return RANKS[card_rank_i(card)] + SUITS[card_suit_i(card)]


CARD_IDS = {card_to_str(card): card for card in range(54)}


class MurlanBot:
    def __init__(self, ckpt_path, device="cpu"):
        self.device, self.ckpt_path = device, str(ckpt_path)
        self.lock = threading.RLock()
        self.version = 0
        self.fallback = not Path(ckpt_path).is_file()
        if not self.fallback:
            self.model, ckpt = load_model(ckpt_path, device)
            self.version = int(ckpt.get("version", 0))
        else:
            self.model = PolicyValueNet(len(MurlanEnv().reset()[0])).to(device).eval()
        self.policy_id = (
            "strategic" if self.fallback else policy_fingerprint(self.model)
        )

    def snapshot(self):
        with self.lock:
            return (
                deepcopy(self.model).eval(),
                self.version,
                self.fallback,
                self.policy_id,
            )

    def promote(self, model, version):
        with self.lock:
            self.model.load_state_dict(model.state_dict())
            self.model.eval()
            self.version = version
            self.fallback = False
            self.policy_id = policy_fingerprint(self.model)


@dataclass
class GameSession:
    env: MurlanEnv
    human_seat: int
    model: object
    version: int
    fallback: bool
    mode: str
    learn: bool
    policy_id: str = ""
    player_names: list = field(default_factory=list)
    done: bool = False
    step: int = 0
    trajectory: HandTrajectory = field(default_factory=HandTrajectory)
    events: list = field(default_factory=list)
    trick_cards: list = field(default_factory=list)
    lock: object = field(default_factory=threading.RLock)
    touched: float = field(default_factory=time.monotonic)
    requests: dict = field(default_factory=dict)


class GameManager:
    def __init__(self, bot, learner=None, replay=None, max_games=128, ttl=7200):
        self.bot, self.learner = bot, learner
        self.replay = replay or ReplayStore()
        self.games = {}
        self.lock = threading.RLock()
        self.max_games, self.ttl = max_games, ttl
        self.last_names = []

    def _session(self, gid):
        with self.lock:
            session = self.games.get(gid)
            if session is None or time.monotonic() - session.touched > self.ttl:
                if session is not None:
                    del self.games[gid]
                raise KeyError("Game not found or expired")
            session.touched = time.monotonic()
            return session

    def new_game(self, seed=None, human_seat=0, mode="learn", learn=True, previous_game_id=None):
        if human_seat not in range(4) or mode not in ("learn", "competitive"):
            raise ValueError("Invalid seat or mode")
        env = MurlanEnv(secrets.randbits(32) if seed is None else seed)
        env.reset()
        model, version, fallback, policy_id = self.bot.snapshot()
        gid = secrets.token_urlsafe(24)
        session = GameSession(
            env, human_seat, model, version, fallback, mode, learn, policy_id
        )
        with self.lock:
            expired = [
                key
                for key, s in self.games.items()
                if time.monotonic() - s.touched > self.ttl
            ]
            for key in expired:
                del self.games[key]
            if len(self.games) >= self.max_games:
                raise RuntimeError("Server is at game capacity; try again later")
            previous = self.games.get(previous_game_id)
            session.player_names = new_player_names(
                human_seat, previous.player_names if previous else self.last_names
            )
            self.last_names = list(session.player_names)
            self.games[gid] = session
        with session.lock:
            self._autoplay(gid, session)
        return gid

    def get_state(self, gid):
        session = self._session(gid)
        with session.lock:
            return self._state(gid, session)

    def _state(self, gid, s):
        e, h = s.env, s.human_seat
        turn = e.current_player == h and not s.done
        legal = np.flatnonzero(e.legal_mask(h)).tolist() if turn else []
        combo = e.current_combo
        options = []
        for a in legal:
            d = action_desc(a)
            ranks = [RANKS[r] for r in d.payload]
            required_cards = []
            if d.kind == "swap_give" and not any(card_rank_i(c) == d.payload[0] for c in e.hands[h]):
                ranks = [RANKS[card_rank_i(e.hands[h][0])]]
                required_cards = [card_to_str(e.hands[h][0])]
            options.append(
                dict(
                    id=a,
                    kind=d.kind,
                    ranks=ranks,
                    required_cards=required_cards,
                    label=pretty_action(a),
                )
            )
        return dict(
            game_id=gid,
            revision=s.step,
            phase=e.phase,
            hand_number=e.hand_index + 1,
            current_player=e.current_player,
            human_seat=h,
            player_names=list(s.player_names),
            require_three_spades=e.require_three_spades,
            scores=list(e.scores),
            goal_target=e.goal_target,
            hand_sizes=[len(x) for x in e.hands],
            your_hand=[card_to_str(c) for c in sort_cards_by_rank(e.hands[h])],
            trick=(
                None
                if combo is None
                else dict(
                    kind=combo.kind,
                    size=combo.size,
                    primary_rank=combo.primary_rank,
                    ranks=[RANKS[r] for r in combo.ranks],
                    cards=list(s.trick_cards),
                )
            ),
            passed=list(e.passed),
            finished=list(e.finished),
            done=s.done,
            your_turn=turn,
            winner=int(np.argmax(e.scores)) if s.done else None,
            legal_actions=legal,
            legal_action_names=[pretty_action(a) for a in legal],
            actions=options,
            events=list(s.events[-32:]),
            bot_version=s.version,
            bot_kind="strategic" if s.fallback else "policy",
            mode=s.mode,
            learning_enabled=s.learn,
        )

    def human_act(self, gid, action, revision=None, request_id=None, cards=None):
        s = self._session(gid)
        with s.lock:
            if cards is not None and (not isinstance(cards, list) or any(not isinstance(c, str) or c not in CARD_IDS for c in cards)):
                raise ValueError("Unknown selected card")
            request = (action, revision, None if cards is None else tuple(sorted(cards)))
            if request_id and request_id in s.requests:
                previous = s.requests[request_id]
                if previous != request:
                    raise ValueError("Request ID already used for a different action")
                return self._state(gid, s)
            if revision is not None and revision != s.step:
                raise ValueError("Stale turn; refresh the game")
            if s.done or s.env.current_player != s.human_seat:
                raise ValueError("It is not your turn")
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < ACTION_DIM
                or not s.env.legal_mask(s.human_seat)[action]
            ):
                raise ValueError("Illegal action")
            card_ids = None if cards is None else [CARD_IDS[c] for c in cards]
            if card_ids is not None:
                s.env.validate_cards(action, card_ids)
            self._step(gid, s, action, human=True, cards=card_ids)
            self._autoplay(gid, s)
            if request_id:
                s.requests[request_id] = request
                if len(s.requests) > 128:
                    del s.requests[next(iter(s.requests))]
            return self._state(gid, s)

    def _step(self, gid, s, action, human=False, stats=None, cards=None):
        env = s.env
        p, hand = env.current_player, env.hand_index
        obs, legal = env.observe(p), env.legal_mask(p)
        if s.learn:
            # Commit each human move immediately; unfinished games retain their moves.
            self.replay.record_action(
                gid, hand, s.step, p, human, s.version, obs, legal, action
            )
        if stats is not None and s.learn:
            logp, value = stats
            s.trajectory.record(
                p, obs, legal, action, logp, value, temperature=1.0, epsilon=0.03
            )
        before = list(env.hands[p])
        exact_cards = cards
        _, _, rewards, done, info = env.step(action, cards=exact_cards)
        if s.learn:
            s.trajectory.reward(rewards)
        s.step += 1
        desc = action_desc(action)
        # Exact public action history, including every bot move between human turns.
        cards = []
        if info.played and exact_cards is not None:
            cards = [card_to_str(c) for c in exact_cards]
        elif info.played and not info.hand_over:
            cards = [card_to_str(c) for c in before if c not in env.hands[p]]
        elif info.played:
            need = (
                list(desc.payload)
                if desc.kind == "straight"
                else list(desc.payload) * info.played.size
            )
            for rank in need:
                c = next(c for c in before if card_rank_i(c) == rank)
                before.remove(c)
                cards.append(card_to_str(c))
        event = dict(
            id=s.step,
            player=p,
            kind=desc.kind,
            label=pretty_action(action),
            cards=cards,
            hand=hand + 1,
        )
        if env.current_combo is None:
            s.trick_cards = []
        elif info.played:
            s.trick_cards = list(cards)
        if info.hand_over:
            event["finish_order"] = info.hand_finish_order
        s.events.append(event)
        s.events = s.events[-64:]
        s.done = done
        if info.hand_over:
            if s.learn:
                self.replay.complete_hand(
                    gid,
                    hand,
                    s.version,
                    s.trajectory.finish(),
                    info.hand_finish_order,
                    policy_id=s.policy_id,
                )
                if self.learner:
                    self.learner.wake()
            s.trajectory = HandTrajectory()
            # Freeze a serving policy for a whole hand; never change it mid-trajectory.
            s.model, s.version, s.fallback, s.policy_id = self.bot.snapshot()

    def _autoplay(self, gid, s):
        for _ in range(5000):
            if s.done or s.env.current_player == s.human_seat:
                return
            if s.fallback:
                action = choose_action(s.env, "strategic")
                stats = None
            else:
                sampled = s.mode == "learn"
                action, logp, value = policy_action(
                    s.model, s.env, greedy=not sampled, epsilon=0.03 if sampled else 0.0
                )
                # Greedy competitive play is not on-policy PPO experience.
                stats = (logp, value) if sampled else None
            self._step(gid, s, action, stats=stats)
        raise RuntimeError("Autoplay exceeded its safety limit")
