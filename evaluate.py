import argparse
import json
import torch
from murlan.checkpoint import load_model
from murlan.evaluation import evaluate, summarize


def main():
    p = argparse.ArgumentParser(
        description="Seat-balanced baseline evaluation; independent from training"
    )
    p.add_argument(
        "--policy",
        default="strategic",
        help="random, greedy, strategic, or checkpoint path",
    )
    p.add_argument("--deals", type=int, default=32)
    p.add_argument("--seed", type=int, default=900000)
    p.add_argument("--output")
    p.add_argument("--mode", choices=["hand", "match"], default="hand")
    args = p.parse_args()
    if args.deals < 1:
        p.error("--deals must be positive")
    torch.set_num_threads(1)
    policy = (
        args.policy
        if args.policy in ("random", "greedy", "strategic")
        else load_model(args.policy)[0]
    )
    records = evaluate(policy, range(args.seed, args.seed + args.deals), mode=args.mode)
    result = dict(
        policy=args.policy, mode=args.mode, summary=summarize(records), records=records
    )
    print(json.dumps(result["summary"], indent=2))
    if args.output:
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
