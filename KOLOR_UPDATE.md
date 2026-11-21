# Kolor rule correction

Install the current `murlan_github_learning_update.zip` over the improved Murlan project with the manual card selection interface. It includes the consecutive-response fix, every Kolor starting at 2, every Kolor starting at A through the other A, and the GitHub model persistence feature. It works whether or not you installed the earlier Kolor patches.

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

The A-low rule applies to **every length from 5 to 14 cards**:

| Kolor | Cards | Only possible Kolor response |
| --- | --- | --- |
| A-2-3-4-5 | 5 | 2-3-4-5-6 |
| A-2-3-4-5-6 | 6 | 2-3-4-5-6-7 |
| A-2-3-4-5-6-7-8-9-10-J | 11 | 2-3-4-5-6-7-8-9-10-J-Q |
| A-2-3-4-5-6-7-8-9-10-J-Q-K | 13 | 2-3-4-5-6-7-8-9-10-J-Q-K-A |
| A-2-3-4-5-6-7-8-9-10-J-Q-K-A | 14 | None |

The 14-card run requires **two different physical aces**, such as AC and AD. One card cannot be selected twice. Every consecutive window of 5–14 cards along A-2-3-4-5-6-7-8-9-10-J-Q-K-A is available; the sequence stops at that second A. Jokers are not allowed in Kolors, and a run cannot end in 2 or wrap from K-A to 2.

Under the same-length rule, **any Kolor ending in A has no higher Kolor response**; 10-J-Q-K-A is the five-card example. The existing katërshe/quad rule is unchanged: a quad can beat a Kolor.

The 13-card A-2-...-K and 2-3-...-A use the same set of ranks. The engine retains the declared sequence so the former ends in K and the latter in A. Manual selection uses the higher interpretation when both are legal, regardless of the order in which cards were clicked.

## Online rule check

Checked on 21 November 2026:

- [Murlan Arena, Playable Combinations](https://www.murlanarena.com/) explicitly gives the full A-2-3-4-5-6-7-8-9-10-J-Q-K-A example and allows runs of five or more cards.
- [Visi Mobile Games, Rregullat e Murlanit](https://www.visixplay.com/murlan/rules.php?lang=al) allows low A and 2 in runs and specifies the immediately-next, equal-length response rule.

Published implementations differ on other rules. This update applies the requested A-starting sequences to this project's existing rule variant.

## Files that implement the correction

- `murlan/actions.py`: includes all 55 straight patterns. Existing straight IDs stay fixed. A-2-3-4-5 retains ID 91, all 2-low runs occupy IDs 92–100, and new A-low lengths 6–14 occupy IDs 101–109.
- `murlan/cards.py`: compares sequences by their actual ending rank, distinguishing the two 13-card interpretations.
- `murlan/env.py`: enforces `straight_primary_rank(seq) == cur_val + 1`, equal lengths, and rank multiplicity (two aces for 14 cards). Both manual and automatic plays retain sequence order. The same legal mask governs human moves, robot moves, and training.
- `murlan/checkpoint.py`: migrates 100-, 101- and 109-action checkpoints into the 118-action model. The eight swap actions move to 110–117. Existing action outputs are preserved; new outputs start from related straights.
- `murlan/features.py`: counts both aces and retains the original length-feature scaling for existing weights. A 14-card value is 14/12 and is not clipped. The strategic baseline also uses these corrected counts when planning its hand.
- `app/replay.py`: retains old human records but excludes their old action masks from new training. The updated policy fingerprint also causes old PPO hands to be retired rather than reused under different rules.
- `train_selfplay.py`: prevents resuming an old optimizer with incompatible output dimensions. Use `--init` for old checkpoints; new checkpoints support `--resume` as usual.
- `static/index.html`: explains the corrected rules in Albanian.

## Apply to your project

1. Extract the newly downloaded `murlan_github_learning_update.zip` and copy its contents into your existing repository root, replacing the matching files. Use the folder containing `app`, `murlan`, `static`, and `.git`.
2. Copy **all** supplied files together. Editing only the action list would cause old model dimensions to mismatch.
3. Keep your checkpoint files. They load automatically through the migration; you do not need to delete your model or replay database. This preserves the learned weights as a starting point, but those weights have not been retrained or benchmarked for the corrected rules.
4. Commit and push from your repository folder:

```powershell
git add .
git commit -m "Correct Kolor sequence rules and migrate saved policies"
git pull --rebase origin main
git push origin main
```

5. Deploy the latest commit in Render. Restarting the application starts fresh game sessions; refresh the page and begin a new match. Existing GitHub saving settings remain valid. To enable that feature for the first time, follow `GITHUB_MODEL_SETUP.md`.

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

Regression coverage includes every 2-low run (5–13 cards), every A-low run (5–14), exact physical card counts, the 3-spades opening constraint, 13- and 14-card HTTP plays, immediate equal-length responses, all ace-ending runs, three older checkpoint schemas, legacy replay isolation, GitHub restore and save across schema migration, complete games, and finite learning updates.

Validation completed: 153 Python tests and 9 JavaScript tests passed.
