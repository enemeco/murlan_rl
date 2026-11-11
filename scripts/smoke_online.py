"""Exercise the real replay -> PPO/imitation -> evaluation loop with a synthetic human.

Run: python -m scripts.smoke_online --output reports/online_smoke.json
No real user records or serving checkpoints are modified.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from app.game_manager import GameManager, MurlanBot
from app.online_learning import OnlineLearner
from app.replay import ReplayStore
from murlan.baselines import choose_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/improved_policy.pt')
    parser.add_argument('--hands', type=int, default=4)
    parser.add_argument('--eval-deals', type=int, default=8)
    parser.add_argument('--output', default='reports/online_smoke.json')
    args = parser.parse_args()
    if min(args.hands, args.eval_deals) < 1:
        parser.error('hands and eval-deals must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(2718)
    with tempfile.TemporaryDirectory(prefix='murlan-online-smoke-') as directory:
        root = Path(directory)
        checkpoint = root / 'champion.pt'
        shutil.copyfile(args.checkpoint, checkpoint)
        replay = ReplayStore(root / 'replay.sqlite3')
        bot = MurlanBot(checkpoint)
        manager = GameManager(bot, replay=replay)
        gid = manager.new_game(seed=2718)
        for turn in range(2000):
            if replay.stats()['completed_hands'] >= args.hands:
                break
            state = manager.get_state(gid)
            if state['done']:
                gid = manager.new_game(seed=2718 + turn)
                continue
            action = choose_action(manager.games[gid].env, 'strategic')
            manager.human_act(gid, action, state['revision'], f'synthetic-{turn:06}')
        else:
            raise RuntimeError('Synthetic game did not collect enough hands')
        before = replay.stats()
        learner = OnlineLearner(bot, replay, min_hands=args.hands,
                                min_decisions=1, eval_deals=args.eval_deals, start=False)
        details = learner.train_once()
        if details is None:
            raise RuntimeError('The candidate update did not run')
        restored = MurlanBot(checkpoint)
        if restored.policy_id != bot.policy_id:
            raise RuntimeError('The saved champion differs from the serving champion')
        report = dict(synthetic_human=True, checkpoint=args.checkpoint,
                      before=before, after=replay.stats(), candidate=details,
                      serving_version=bot.version, persisted_champion_verified=True)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(status=learner.status, human_moves=before['human_moves'],
                              hands=before['completed_hands'], gate=details['gate']), indent=2))


if __name__ == '__main__':
    main()
