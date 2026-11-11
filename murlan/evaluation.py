"""Paired deals and all four seats, with uncertainty across deals."""

import numpy as np
import torch
from .env import MurlanEnv
from .baselines import choose_action
from .model import masked_categorical


@torch.inference_mode()
def policy_action(
    model, env, greedy=True, temperature=1.0, epsilon=0.0, generator=None
):
    device = next(model.parameters()).device
    obs = env.observe(env.current_player)
    legal = env.legal_mask(env.current_player)
    logits, value = model(torch.as_tensor(obs, device=device).unsqueeze(0))
    dist = masked_categorical(
        logits, torch.as_tensor(legal, device=device).unsqueeze(0), temperature, epsilon
    )
    action = (
        dist.probs.argmax(-1)
        if greedy
        else torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
    )
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def evaluate(policy, seeds, opponents=("random", "greedy", "strategic"), mode="hand"):
    """Return points (0..3) and wins per deal; one hand per rotated seat.

    The same seed/deal/opponent RNG is reused for candidate and champion.
    """
    if mode not in ("hand", "match"):
        raise ValueError("Evaluation mode must be hand or match")
    records = []
    for opponent in opponents:
        for seed in seeds:
            points, wins, match_wins = [], [], []
            for seat in range(4):
                env = MurlanEnv(seed)
                env.reset(seed)
                rng = np.random.default_rng(seed * 7 + seat)
                seat_points, seat_wins = [], []
                for _ in range(2000 if mode == "hand" else 20000):
                    if env.current_player == seat:
                        action = (
                            choose_action(env, policy, rng)
                            if isinstance(policy, str)
                            else policy_action(policy, env)[0]
                        )
                    else:
                        action = choose_action(env, opponent, rng)
                    _, _, _, done, info = env.step(action)
                    if info.hand_over:
                        place = info.hand_finish_order.index(seat)
                        seat_points.append(3 - place)
                        seat_wins.append(float(place == 0))
                        if mode == "hand" or done:
                            points.append(float(np.mean(seat_points)))
                            wins.append(float(np.mean(seat_wins)))
                            if done:
                                match_wins.append(
                                    float(int(np.argmax(env.scores)) == seat)
                                )
                            break
                else:
                    raise RuntimeError("Evaluation hand did not terminate")
            record = dict(
                opponent=opponent,
                seed=int(seed),
                points=float(np.mean(points)),
                win_rate=float(np.mean(wins)),
            )
            if mode == "match":
                record["match_win_rate"] = float(np.mean(match_wins))
            records.append(record)
    return records


def summarize(records):
    result = {}
    for opponent in sorted({r["opponent"] for r in records}):
        rows = [r for r in records if r["opponent"] == opponent]
        points = np.array([r["points"] for r in rows])
        result[opponent] = dict(
            mean_points=float(points.mean()),
            win_rate=float(np.mean([r["win_rate"] for r in rows])),
            points_ci95=(
                1.96 * float(points.std(ddof=1)) / len(points) ** 0.5
                if len(points) > 1
                else None
            ),
            deals=len(rows),
            hands=4 * len(rows),
        )
        if "match_win_rate" in rows[0]:
            result[opponent]["matches"] = result[opponent].pop("hands")
            result[opponent]["match_win_rate"] = float(
                np.mean([r["match_win_rate"] for r in rows])
            )
    return result


def promotion_decision(candidate, incumbent, min_deals=8, margin=0.0):
    a = {(r["opponent"], r["seed"]): r["points"] for r in candidate}
    b = {(r["opponent"], r["seed"]): r["points"] for r in incumbent}
    if a.keys() != b.keys() or not a:
        raise ValueError("Evaluation results must use the same opponents and seeds")
    # Cluster by deal: the rotated seats and opponent types are correlated.
    seeds = sorted({k[1] for k in a})
    delta = np.array([np.mean([a[k] - b[k] for k in a if k[1] == s]) for s in seeds])
    mean = float(delta.mean())
    lower = (
        mean - 1.96 * float(delta.std(ddof=1)) / len(delta) ** 0.5
        if len(delta) > 1
        else -float("inf")
    )
    per_opponent = {
        o: float(np.mean([a[k] - b[k] for k in a if k[0] == o]))
        for o in {k[0] for k in a}
    }
    match_deltas = {}
    if "match_win_rate" in candidate[0] and "match_win_rate" in incumbent[0]:
        for opponent in per_opponent:
            match_deltas[opponent] = float(
                np.mean(
                    [
                        r["match_win_rate"]
                        for r in candidate
                        if r["opponent"] == opponent
                    ]
                )
                - np.mean(
                    [
                        r["match_win_rate"]
                        for r in incumbent
                        if r["opponent"] == opponent
                    ]
                )
            )
    return dict(
        promote=len(seeds) >= min_deals
        and lower > margin
        and min(per_opponent.values()) >= -0.1
        and (not match_deltas or min(match_deltas.values()) >= -0.05),
        mean_delta=mean,
        lower95=lower if np.isfinite(lower) else None,
        per_opponent_delta=per_opponent,
        per_opponent_match_win_delta=match_deltas,
        deals=len(seeds),
    )
