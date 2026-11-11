"""Durable per-action feedback and a recoverable completed-hand training queue."""

from pathlib import Path
import json
import sqlite3
import numpy as np


class ReplayStore:
    def __init__(self, path="data/replay.sqlite3"):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, hand INTEGER NOT NULL,
                    step INTEGER NOT NULL, seat INTEGER NOT NULL, human INTEGER NOT NULL,
                    version INTEGER NOT NULL, obs TEXT NOT NULL, legal TEXT NOT NULL,
                    action INTEGER NOT NULL, points REAL,
                    UNIQUE(game_id,step));
                CREATE TABLE IF NOT EXISTS hands (
                    id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, hand INTEGER NOT NULL,
                    version INTEGER NOT NULL, rows TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(game_id,hand));
                CREATE TABLE IF NOT EXISTS updates (
                    id INTEGER PRIMARY KEY, version INTEGER NOT NULL, status TEXT NOT NULL,
                    details TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS pending_hands ON hands(processed,version);
                CREATE INDEX IF NOT EXISTS human_actions ON actions(human,id);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(hands)")}
            if "policy_id" not in columns:
                db.execute(
                    "ALTER TABLE hands ADD COLUMN policy_id TEXT NOT NULL DEFAULT ''"
                )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def record_action(
        self, game_id, hand, step, seat, human, version, obs, legal, action
    ):
        with self.connect() as db:
            db.execute(
                "INSERT INTO actions(game_id,hand,step,seat,human,version,obs,legal,action) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    game_id,
                    hand,
                    step,
                    seat,
                    int(human),
                    version,
                    json.dumps(obs.tolist()),
                    json.dumps(legal.astype(int).tolist()),
                    int(action),
                ),
            )

    def complete_hand(self, game_id, hand, version, rows, finish_order, policy_id=""):
        serial = [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
            for row in rows
        ]
        with self.connect() as db:
            for place, seat in enumerate(finish_order):
                db.execute(
                    "UPDATE actions SET points=? WHERE game_id=? AND hand=? AND seat=?",
                    (3 - place, game_id, hand, seat),
                )
            db.execute(
                "INSERT INTO hands(game_id,hand,version,rows,policy_id) VALUES(?,?,?,?,?)",
                (game_id, hand, version, json.dumps(serial), policy_id),
            )

    def pending(self, version, limit=32, policy_id=None):
        with self.connect() as db:
            entries = db.execute(
                "SELECT id,rows FROM hands WHERE processed=0 AND version=? AND (? IS NULL OR policy_id=?) ORDER BY id LIMIT ?",
                (version, policy_id, policy_id, limit),
            ).fetchall()
        return [e[0] for e in entries], [
            row for _, payload in entries for row in json.loads(payload)
        ]

    def mark_processed(self, ids):
        with self.connect() as db:
            db.executemany(
                "UPDATE hands SET processed=1 WHERE id=?", [(i,) for i in ids]
            )

    def mark_stale(self, version, policy_id=None):
        with self.connect() as db:
            db.execute(
                "UPDATE hands SET processed=2 WHERE processed=0 AND (version<>? OR (? IS NOT NULL AND policy_id<>?))",
                (version, policy_id, policy_id),
            )

    def demonstrations(self, limit=512, seed=0):
        # Bounded random sample across the full history, including unfinished hands.
        rng = np.random.default_rng(seed)
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM actions WHERE human=1")]
            if not ids:
                return []
            chosen = rng.choice(ids, min(limit, len(ids)), replace=False).tolist()
            query = (
                "SELECT obs,legal,action,points FROM actions WHERE id IN ("
                + ",".join("?" for _ in chosen)
                + ")"
            )
            rows = db.execute(query, chosen).fetchall()
        return [
            dict(
                obs=json.loads(o),
                legal=json.loads(l),
                act=a,
                weight=0.15 if points is None else 0.25 + float(points) / 3.0,
            )
            for o, l, a, points in rows
        ]

    def update_log(self, version, status, details):
        with self.connect() as db:
            db.execute(
                "INSERT INTO updates(version,status,details) VALUES(?,?,?)",
                (version, status, json.dumps(details, allow_nan=False)),
            )

    def stats(self):
        with self.connect() as db:
            human, all_actions = db.execute(
                "SELECT coalesce(sum(human),0),count(*) FROM actions"
            ).fetchone()
            hands = db.execute("SELECT count(*) FROM hands").fetchone()[0]
            pending = db.execute(
                "SELECT count(*) FROM hands WHERE processed=0"
            ).fetchone()[0]
            updates = db.execute("SELECT count(*) FROM updates").fetchone()[0]
            promotions = db.execute(
                "SELECT count(*) FROM updates WHERE status='promoted'"
            ).fetchone()[0]
            last = db.execute(
                "SELECT status,details,created FROM updates ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(
            human_moves=human,
            total_moves=all_actions,
            completed_hands=hands,
            pending_hands=pending,
            updates=updates,
            promotions=promotions,
            last_update=(
                None
                if not last
                else dict(status=last[0], details=json.loads(last[1]), created=last[2])
            ),
        )
