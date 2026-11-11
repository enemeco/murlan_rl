# Murlan gameplay benchmark

## Protocol

- Original: the untouched checkpoint supplied in upstream commit `49effc5`.
- Improved: 303,564 environment steps of training, seed 0, 84 PPO updates, plus 5,973 strategic demonstration decisions (96 hands; eight imitation epochs).
- Training was resumed twice with fresh environments and a rebuilt opponent pool. The raw metrics are in `training_metrics.jsonl`.
- All policies are evaluated greedily. Each tested policy controls one seat against three instances of the named baseline; all four seats are tested on each seed.
- Final single-hand test: seeds 1,700,000–1,700,063, 64 deals × 4 seats = **256 hands per opponent**, 768 hands per policy.
- Final full-match test: seeds 1,800,000–1,800,015, 16 deals × 4 seats = **64 matches per opponent**, 192 matches per policy.
- These final test seeds were separate from training and model selection. Earlier exploratory tests used seeds 900,000 and 950,000; their outcomes informed the decision to extend training and evaluate full matches.
- The final candidate was selected before these tests using 32 paired full-match validation deals (seeds 300,000–300,031). Selection results and the exact gate are in `selection.json`.

## Final full-match results

| Opponents | Original match wins | Improved match wins | Original points / hand | Improved points / hand |
| --- | ---: | ---: | ---: | ---: |
| Random | 85.94% (55/64) | 100.00% (64/64) | 2.257 | 2.877 |
| Greedy | 4.69% (3/64) | 42.19% (27/64) | 1.160 | 1.676 |
| Strategic | 0.00% (0/64) | 18.75% (12/64) | 0.933 | 1.452 |

Full matches include exchanges and score progression. Points per hand are averaged within each match before combining seats. The policy improves substantially over the original checkpoint, but **three strategic opponents remain a difficult matchup**. The results do not establish superiority to every baseline or to strong human players.

## Final single-hand results

| Opponents | Original hand wins | Improved hand wins | Original points | Improved points | Strategic baseline points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 55.47% | 80.86% | 2.148 | 2.770 | 2.562 |
| Greedy | 17.58% | 28.91% | 1.215 | 1.656 | 1.793 |
| Strategic | 13.28% | 22.66% | 1.160 | 1.566 | 1.500 |

The strategic baseline beats the learned policy against greedy opponents on this test. It also scores exactly 1.5 points when rotating through four identical strategic agents, as expected. Keep it in the opponent pool and retain it as a separate comparison.

## Interpretation and limits

Only one training seed was run. The included policy has **not been shown to converge**. The raw JSON files provide deal-level records and approximate 95% point intervals clustered by deal; the point intervals are not confidence intervals for the displayed win percentages. Multiple independent training seeds and larger human-player evaluations are still needed for a research claim. The hand-placement reward is a useful proxy, not direct optimization of match win probability.

The 303,564-step candidate passed the pre-specified promotion criteria on the larger validation set. Relative to the 82,617-step incumbent, its average point delta was 0.0292, with an approximate lower 95% bound of 0.0018. Its strategic-opponent match win rate was slightly lower on that validation sample, within the five-percentage-point tolerance; promotion therefore does not mean every metric improved.

## Verification

- **25 automated tests passed**, including full human-versus-three-bot matches, credit attribution, checkpoint migration, on-policy identity checks, duplicate clicks, storage, and promotion/rejection.
- Python compilation, JavaScript syntax validation, and Git whitespace checks passed.
- Browser verification covered card selection, play, automatic bot turns, pass-and-skip behavior, public card suits, move logging, rules, and reload recovery.
- `online_smoke.json` records a real candidate update from a **synthetic human** controlled by the strategic baseline. It exercises storage, PPO, imitation, full-match validation, and saved-champion recovery; it is not a claim about learning from actual human users.
- That smoke run stored 45 synthetic human moves across four completed hands (238 total moves). The candidate failed the eight-deal promotion gate, so the existing champion was correctly retained; its persisted weights were verified after the update.
- Docker/Render deployment was not performed. Deployment configuration and persistence instructions are provided.

## Included checkpoints

| File | Role |
| --- | --- |
| `checkpoints/murlan_policy.pt` | Original upstream weights, unchanged. |
| `checkpoints/improved_policy.pt` | Selected policy at 303,564 steps; default seed for a fresh server. |
| `checkpoints/training_resume.pt` | Latest model, optimizer, CPU RNG, and counters for continued training. |

The improved network has 161,893 trainable parameters and 249 public/own-information observation features.

## Reproduce

```bash
python evaluate.py --policy checkpoints/murlan_policy.pt --deals 64 --seed 1700000
python evaluate.py --policy checkpoints/improved_policy.pt --deals 64 --seed 1700000
python evaluate.py --policy checkpoints/murlan_policy.pt --mode match --deals 16 --seed 1800000
python evaluate.py --policy checkpoints/improved_policy.pt --mode match --deals 16 --seed 1800000
python -m scripts.smoke_online --output runs/online_smoke.json
```
