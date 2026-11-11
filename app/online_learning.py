"""Train a separate candidate; only promote after held-out paired evaluation."""

from pathlib import Path
import logging
import threading
import torch
from murlan.ppo import PPOConfig, PPOTrainer
from murlan.trajectory import rows_to_batch
from murlan.training import imitate, teacher_examples
from murlan.evaluation import evaluate, promotion_decision, summarize
from murlan.checkpoint import save_checkpoint

log = logging.getLogger(__name__)


class OnlineLearner:
    def __init__(
        self, bot, replay, min_hands=4, min_decisions=256, eval_deals=12, start=True
    ):
        self.bot, self.replay = bot, replay
        self.min_hands, self.min_decisions, self.eval_deals = (
            min_hands,
            min_decisions,
            eval_deals,
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._training = threading.Lock()
        self.status = "waiting"
        self.error = None
        self._thread = None
        if start:
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="murlan-learner"
            )
            self._thread.start()

    def wake(self):
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _worker(self):
        while not self._stop.is_set():
            try:
                self.train_once()
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                log.exception(
                    "Online candidate update failed; serving policy is unchanged"
                )
            self._wake.wait(timeout=5)
            self._wake.clear()

    def train_once(self):
        if not self._training.acquire(blocking=False):
            return None
        try:
            model, version, fallback, policy_id = self.bot.snapshot()
            self.replay.mark_stale(version, policy_id)
            ids, rows = self.replay.pending(version, policy_id=policy_id)
            if len(ids) < self.min_hands or (
                bool(rows) and len(rows) < self.min_decisions and not fallback
            ):
                self.status = "waiting"
                return None
            self.status = "training"
            # All PPO rows came from the exact frozen champion and are used once.
            stats = {}
            if rows:
                trainer = PPOTrainer(
                    model,
                    PPOConfig(
                        lr=5e-5,
                        epochs=2,
                        minibatch_size=128,
                        ent_coef=0.01,
                        target_kl=0.01,
                    ),
                )
                stats = trainer.update(rows_to_batch(rows))
            if fallback:
                stats["bootstrap_loss"] = imitate(
                    model, teacher_examples(32, version + 40000), epochs=4, lr=5e-4
                )
            demos = self.replay.demonstrations(limit=1024, seed=version + len(ids))
            stats["human_moves"] = len(demos)
            stats["imitation_loss"] = imitate(model, demos, epochs=1, lr=1e-5)
            self.status = "evaluating"
            # Fresh validation deals per attempt; reserved offline test seeds stay untouched.
            attempts = self.replay.stats()["updates"]
            seeds = range(
                200000 + attempts * 100, 200000 + attempts * 100 + self.eval_deals
            )
            champion, champion_version, _, champion_id = self.bot.snapshot()
            if champion_version != version or champion_id != policy_id:
                raise RuntimeError("Champion changed while candidate was training")
            candidate_results = evaluate(model, seeds, mode="match")
            champion_results = evaluate(
                "strategic" if fallback else champion, seeds, mode="match"
            )
            gate = promotion_decision(candidate_results, champion_results)
            details = dict(
                training=stats,
                gate=gate,
                evaluation=summarize(candidate_results),
                source_policy_id=policy_id,
            )
            ckpt = Path(self.bot.ckpt_path)
            candidate_path = ckpt.with_name(ckpt.stem + ".candidate.pt")
            save_checkpoint(candidate_path, model, version=version, validation=details)
            if gate["promote"]:
                # Save first so a failed disk write never silently changes the live bots.
                save_checkpoint(ckpt, model, version=version + 1, validation=details)
                self.bot.promote(model, version + 1)
                self.status = "promoted"
            else:
                self.status = "retained"
            self.replay.update_log(version, self.status, details)
            self.replay.mark_processed(ids)
            self.error = None
            return details
        finally:
            self._training.release()
