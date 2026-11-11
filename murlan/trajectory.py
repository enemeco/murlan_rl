"""Hand episodes separated by seat: terminal rewards reach every participant."""

import numpy as np
from .ppo import compute_gae


class HandTrajectory:
    def __init__(self):
        self.seats = [[] for _ in range(4)]

    def record(
        self, seat, obs, legal, action, logp, value, temperature=1.0, epsilon=0.0
    ):
        self.seats[seat].append(
            dict(
                obs=obs.copy(),
                legal=legal.copy(),
                act=int(action),
                logp=float(logp),
                val=float(value),
                rew=0.0,
                done=0.0,
                temperature=float(temperature),
                epsilon=float(epsilon),
            )
        )

    def reward(self, rewards):
        # Rewards produced by another seat's move still belong to this seat.
        for seat, rows in enumerate(self.seats):
            if rows:
                rows[-1]["rew"] += float(rewards[seat])

    def finish(self, gamma=0.995, lam=0.95):
        result = []
        for rows in self.seats:
            if not rows:
                continue
            rows[-1]["done"] = 1.0
            adv, ret = compute_gae(
                np.array([r["rew"] for r in rows]),
                np.array([r["val"] for r in rows]),
                np.array([r["done"] for r in rows]),
                gamma,
                lam,
            )
            for row, a, r in zip(rows, adv, ret):
                result.append(dict(row, adv=float(a), ret=float(r)))
        return result


def rows_to_batch(rows):
    if not rows:
        raise ValueError("Empty rollout")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
