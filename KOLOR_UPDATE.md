# Kolor rule correction

Install this complete replacement patch over the improved Murlan project with the manual card selection interface. It includes the consecutive-response fix and the full family of Kolors starting at 2; it works whether or not you installed the first five-card-only patch.

## Rules

The five-card sequence is:

```text
A-2-3-4-5
2-3-4-5-6
3-4-5-6-7
4-5-6-7-8
5-6-7-8-9
6-7-8-9-10
7-8-9-10-J
8-9-10-J-Q
9-10-J-Q-K
10-J-Q-K-A
```

Only the next line can beat the current line. For example, 3-4-5-6-7 can be beaten by 4-5-6-7-8, but not by 5-6-7-8-9. There is no wrap from the top back to the bottom. A means ace (1).

The 2-low rule applies to **every length from 5 to 13 cards**. Examples:

| Kolor | Only possible Kolor response |
| --- | --- |
| 2-3-4-5-6-7 | 3-4-5-6-7-8 |
| 2-3-4-5-6-7-8-9-10-J | 3-4-5-6-7-8-9-10-J-Q |
| 2-3-4-5-6-7-8-9-10-J-Q-K | 3-4-5-6-7-8-9-10-J-Q-K-A |
| 2-3-4-5-6-7-8-9-10-J-Q-K-A | None |

All consecutive windows of 5–13 cards along 2-3-4-5-6-7-8-9-10-J-Q-K-A are included, plus the existing A-2-3-4-5. Under the same-length rule, **any Kolor ending in A has no higher Kolor response**; 10-J-Q-K-A is the five-card example. The existing katërshe/quad rule is unchanged: a quad can beat a Kolor.

## Files that implement the correction

- `murlan/actions.py`: generates every 2-low run using `(12, *range(length - 1))` for lengths 5 through 13. These are internal indices; the general card order for singles and groups is still 3, 4, ..., A, 2.
- `murlan/cards.py`: ranks every 2-low run by its ending card, ignoring the 2 when finding that end. A-2-3-4-5 keeps its special low ranking.
- `murlan/env.py`: replaces the higher-rank comparison with `straight_primary_rank(seq) == cur_val + 1`. The same legal mask governs human moves, robot moves, and training.
- `murlan/checkpoint.py`: migrates both 100-action and 101-action checkpoints into the 109-action model. Existing straight IDs remain the same. Nine 2-low actions occupy IDs 92–100; the eight swap actions move to 101–108. New outputs start from related existing straights; an existing learned 2-to-6 output is preserved.
- `murlan/features.py`: retains the original length-feature scaling so migrated weights keep their input meanings. The 13-card value is 13/12 and is not clipped.
- `app/replay.py`: retains old human records but excludes their old action masks from new training. The updated policy fingerprint also causes old PPO hands to be retired rather than reused under different rules.
- `train_selfplay.py`: prevents resuming an old optimizer with incompatible output dimensions. Use `--init` for old checkpoints; new checkpoints support `--resume` as usual.
- `static/index.html`: explains the corrected rules in Albanian.

## Apply to your project

1. Extract `murlan_kolor_fix.zip` and copy its contents into your existing repository root, replacing the matching files. Use the folder containing `app`, `murlan`, `static`, and `.git`.
2. Copy **all** supplied files together. Editing only the action list would cause old model dimensions to mismatch.
3. Keep your checkpoint files. They load automatically through the migration; you do not need to delete your model or replay database. This preserves the learned weights as a starting point, but those weights have not been retrained or benchmarked for the corrected rules.
4. Commit and push from your repository folder:

```powershell
git add .
git commit -m "Correct Kolor sequence rules and migrate saved policies"
git push origin main
```

5. Deploy the latest commit in Render. Restarting the application starts fresh game sessions; refresh the page and begin a new match. No environment-variable changes are needed for this patch.

For optional offline training from an existing model, use a new output directory:

```bash
python train_selfplay.py --init checkpoints/improved_policy.pt --output-dir runs/kolor_corrected
```

The update is prepared locally; applying it and deploying remain your steps.

## Verification

```bash
python -m pytest -q
node --test tests/test_selection.cjs
```

Regression coverage includes every 2-low run from 5 through 13 cards, exact card selection and the 3-spades opening constraint, a 13-card move through the HTTP API, consecutive equal-length responses, all ace-ending runs, both old checkpoint schemas, legacy replay isolation, complete games, and finite learning updates.

Validation completed: 94 Python tests and 7 JavaScript tests passed.
