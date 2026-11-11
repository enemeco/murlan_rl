"""League PPO with complete per-seat hand trajectories and evaluated promotion."""

from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from murlan.checkpoint import load_model, save_checkpoint
from murlan.env import MurlanEnv
from murlan.model import PolicyValueNet
from murlan.ppo import PPOConfig, PPOTrainer
from murlan.training import LeagueCollector, teacher_examples, imitate
from murlan.evaluation import evaluate, summarize, promotion_decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n_env", "--n-env", type=int, default=8)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", "--save-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-deals", type=int, default=16)
    p.add_argument("--eval-mode", choices=["hand", "match"], default="match")
    p.add_argument("--warmstart-hands", type=int, default=64)
    p.add_argument("--warmstart-epochs", type=int, default=4)
    p.add_argument("--init", help="Initialize/migrate a policy; reset optimizer")
    p.add_argument(
        "--resume", help="Resume a v2 training checkpoint including optimizer"
    )
    p.add_argument("--human-db", help="Optional SQLite replay collected by the web app")
    p.add_argument("--output-dir", default="runs/league")
    args = p.parse_args()
    if (
        min(
            args.steps,
            args.n_env,
            args.rollout,
            args.save_every,
            args.eval_every,
            args.eval_deals,
        )
        < 1
    ):
        p.error(
            "Steps, environments, rollout, save/eval intervals and deals must be positive"
        )
    if args.resume and args.init:
        p.error("Use either --resume or --init")
    if args.warmstart_hands < 0 or args.warmstart_epochs < 0:
        p.error("Warmstart values cannot be negative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig()
    ckpt = {}
    if args.resume or args.init:
        model, ckpt = load_model(args.resume or args.init, args.device)
    else:
        model = PolicyValueNet(len(MurlanEnv().reset()[0])).to(args.device)
    if args.resume:
        if "optimizer" not in ckpt:
            p.error(
                "Resume requires a training checkpoint; use --init for legacy/serving models"
            )
        cfg = PPOConfig(**ckpt["ppo"])
    if not args.resume and args.warmstart_hands:
        examples = teacher_examples(args.warmstart_hands, args.seed + 10000)
        loss = imitate(
            model, examples, epochs=args.warmstart_epochs, lr=5e-4, seed=args.seed
        )
        print(
            json.dumps(dict(warmstart_examples=len(examples), imitation_loss=loss)),
            flush=True,
        )
    replay = None
    if args.human_db:
        from app.replay import ReplayStore

        replay = ReplayStore(args.human_db)
    trainer = PPOTrainer(model, cfg, args.device)
    total = int(ckpt.get("steps", 0)) if args.resume else 0
    update = int(ckpt.get("updates", 0)) if args.resume else 0
    if args.resume:
        trainer.opt.load_state_dict(ckpt["optimizer"])
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
    # Resume starts fresh environments; optimizer/steps resume, trajectories do not.
    collector = LeagueCollector(args.seed + total, args.n_env)
    collector.add_snapshot(model)
    champion = deepcopy(model)
    champion_path = out / "champion.pt"
    if args.resume and champion_path.exists():
        champion, _ = load_model(champion_path, args.device)
    else:
        save_checkpoint(
            champion_path,
            champion,
            steps=total,
            updates=update,
            provenance="initial policy; not yet validated",
        )

    def save():
        save_checkpoint(
            out / "latest.pt",
            model,
            ppo=asdict(cfg),
            optimizer=trainer.opt.state_dict(),
            steps=total,
            updates=update,
            torch_rng=torch.get_rng_state(),
            seed=args.seed,
        )

    while total < args.steps:
        progress = min(total / max(args.steps, 1), 1.0)
        temperature = 1.2 - 0.2 * progress
        epsilon = 0.08 - 0.06 * progress
        cfg.ent_coef = 0.03 - 0.02 * progress
        for group in trainer.opt.param_groups:
            group["lr"] = cfg.lr * (1 - 0.8 * progress)
        before = collector.steps
        batch = collector.collect(model, args.rollout, temperature, epsilon)
        total += collector.steps - before
        stats = trainer.update(batch)
        update += 1
        if replay:
            demos = replay.demonstrations(limit=512, seed=args.seed + update)
            stats["human_imitation_loss"] = imitate(
                model, demos, epochs=1, lr=1e-5, seed=args.seed + update
            )
        record = dict(
            update=update,
            steps=total,
            decisions=len(batch["act"]),
            temperature=temperature,
            epsilon=epsilon,
            **stats,
        )
        if update % args.eval_every == 0 or total >= args.steps:
            seeds = range(
                100000 + update * 100, 100000 + update * 100 + args.eval_deals
            )
            candidate_scores, champion_scores = evaluate(
                model, seeds, mode=args.eval_mode
            ), evaluate(champion, seeds, mode=args.eval_mode)
            gate = promotion_decision(candidate_scores, champion_scores)
            record.update(
                evaluation=summarize(candidate_scores),
                evaluation_mode=args.eval_mode,
                promotion=gate,
            )
            if gate["promote"]:
                champion = deepcopy(model)
                save_checkpoint(
                    champion_path,
                    champion,
                    steps=total,
                    updates=update,
                    evaluation=record["evaluation"],
                    promotion=gate,
                )
            collector.add_snapshot(model)
        if update % args.save_every == 0 or total >= args.steps:
            save()
        print(json.dumps(record), flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    save()
    print(
        f'Saved {out / "latest.pt"}; evaluated serving champion: {champion_path}',
        flush=True,
    )


if __name__ == "__main__":
    main()
