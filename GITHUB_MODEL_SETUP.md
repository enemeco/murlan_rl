# Save accepted Murlan improvements to GitHub

The app can replace `checkpoints/improved_policy.pt` on `enemeco/murlan_rl`, branch `main`, after a candidate passes the existing evaluation gate. It saves to GitHub before promoting the candidate for new hands. Rejected candidates do not change the repository. Each startup downloads the current saved checkpoint from GitHub, including when Render restarts an older deployed image.

## 1. Install and push the files

In Render, open your Murlan service → **Settings → Auto-Deploy → Off** first. The app also uses `[skip render] [skip ci]` in model commit messages.

Open PowerShell in your actual Git repository folder. Based on your earlier clone, that was:

```powershell
cd "C:\Users\Ene\Desktop\murlan_rl\murlan_rl"
git rev-parse --show-toplevel
git switch main
git pull --rebase origin main
```

If Git reports uncommitted changes, save/commit your existing work before pulling. Do not reset or force-push.

Extract `murlan_github_learning_update.zip`. Copy its contents into this repository folder, merging folders and replacing matching files. The ZIP includes the complete Kolor correction as well as this update. It is an update for your existing project, not a standalone project. It contains no trained weights and does not overwrite your downloaded model files.

Review and commit the files:

```powershell
git status --short
git add app/github_models.py app/main.py app/online_learning.py app/replay.py murlan/actions.py murlan/cards.py murlan/checkpoint.py murlan/env.py murlan/features.py static/app.js static/index.html tests/test_github_models.py tests/test_kolor_rules.py tests/test_learning.py tests/test_selection.cjs train_selfplay.py README.md render.yaml KOLOR_UPDATE.md GITHUB_MODEL_SETUP.md
git commit -m "Save accepted learning to GitHub and restore on restart"
git push origin main
```

Check that `checkpoints/improved_policy.pt` exists in your GitHub repository. If your project only has `checkpoints/murlan_policy.pt`, use that existing path for `MURLAN_GITHUB_MODEL_PATH` in step 3 instead. The app supports migrating the old checkpoint when loading it.

## 2. Create a restricted GitHub token

Sign into the **enemeco** account that owns the repository. Your previous `mecoene` login produced a permission error; use the repository owner's account for this setup.

Open GitHub **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**, or [open the token creation page](https://github.com/settings/personal-access-tokens/new).

Choose:

| Setting | Value |
| --- | --- |
| Token name | `Murlan accepted model saves` |
| Resource owner | `enemeco` |
| Expiration | Choose an expiry you will renew before it passes, for example 90 days. |
| Repository access | **Only select repositories → murlan_rl** |
| Repository permissions → Contents | **Read and write** |
| Metadata | Keep the automatically included read access. |

Generate and copy the token. Paste it only into Render's secret environment variable in step 3. Do not paste it into chat, source files, `render.yaml`, Git commands, or the browser console. Contents permission applies to the selected repository, not only to the checkpoint file; the app's save code targets only the configured checkpoint.

## 3. Configure Render

In your Murlan service, open **Environment** and add:

| Key | Value |
| --- | --- |
| `MURLAN_GITHUB_SYNC` | `1` |
| `MURLAN_GITHUB_REPO` | `enemeco/murlan_rl` |
| `MURLAN_GITHUB_BRANCH` | `main` |
| `MURLAN_GITHUB_MODEL_PATH` | `checkpoints/improved_policy.pt` |
| `MURLAN_GITHUB_TOKEN` | Paste your token as the secret value. |

Keep `MURLAN_ONLINE_LEARNING=1`. Keep your existing self-ping variables. Run **one** Uvicorn worker; the included Docker command already does this.

Save the variables, then use **Manual Deploy → Deploy latest commit** to deploy the code you pushed. If Render offers a save-and-deploy action, make sure it deploys the latest commit. Changing the YAML alone does not configure an existing manually created service's secret.

## 4. Verify restore and saving

On startup, Render logs should contain:

```text
GitHub model restored: version N from checkpoints/improved_policy.pt (main).
```

`N` is the actual checkpoint version; it might initially be 0. Open:

<https://playmurlan.enemeco.com/api/learning>

Look for a `github_sync` object like:

```json
{
  "enabled": true,
  "state": "restored",
  "saved_version": 0,
  "error": null
}
```

Successful restore verifies reading, not writing. The first accepted improvement verifies write access. Play with learning enabled. The normal worker waits for at least four completed hands and usually 256 sampled robot decisions; evaluation may reject a candidate. A completed game or training attempt does not guarantee a promotion or commit.

After an accepted improvement is saved, expect:

```text
GitHub model saved: accepted version N to checkpoints/improved_policy.pt (main).
```

The repository commit will read:

```text
Save accepted Murlan model vN [skip render] [skip ci]
```

`github_sync.state` becomes `saved` and `saved_version` increases. After you see that confirmation, restart the service once to verify restoration: it should log the same version as restored and `/health` should report that `model_version`. Active games end on a restart, so finish your game first. Replay counts and the number of updates since startup can reset while the model version and learned weights survive.

## 5. Future code changes

Model saves add commits to `main`, so pull before editing/pushing code:

```powershell
git pull --rebase origin main
```

If a model save lands between your pull and push, commit your work, run the pull again, resolve any conflicts, and push normally. Avoid force-pushing or replacing the saved checkpoint with an old downloaded copy. The normal code update ZIP deliberately contains no `.pt` files.

## Troubleshooting

| Status / log | Action |
| --- | --- |
| `enabled: false` | Set `MURLAN_GITHUB_SYNC=1` and deploy the updated code. |
| HTTP 401 | Renew the token in Render; expired tokens cannot restore/save models. |
| HTTP 403 | Check the token's owner, selected repository and Contents write permission. Also check rate limits and branch restrictions. |
| HTTP 404 | Check repository, branch, checkpoint path and token access; the chosen checkpoint must already exist. |
| HTTP 409 / concurrent change | Restart to restore the current GitHub champion before training again. No newer file is overwritten automatically. |
| HTTP 422 | Check whether repository branch rules require a PR, signed commits or checks; this setup uses direct model commits to `main`. |
| `retrying` / network error | The worker retries with backoff starting at 60 seconds, up to 15 minutes; GitHub-requested longer delays are respected. Gameplay remains available. |
| `phase: saving` | Evaluation passed, but live promotion waits for confirmation that the model was saved. The same candidate is retried without retraining it. |
| `phase: restoring` | Startup could not fetch the saved model. Training waits for restoration; available local/bundled robots can still play. |

When you rotate a token, update it in Render and restart/deploy. If you have required branch checks, review those rules before enabling direct checkpoint commits; this code does not bypass repository protections.

## What is preserved

Every **successfully saved accepted model** survives Render sleep, restarts and redeploys. A candidate waiting for an upload is not protected until GitHub confirms the commit. If an upload completes but its response is lost, the next retry checks the exact remote bytes and avoids a duplicate commit.

The feature stores the model and aggregate validation metadata only. Replay moves, SQLite update history and active matches remain local. GitHub keeps previous model versions in commit history; repository size grows with accepted improvements. The current model is under 1 MiB; the app caps synced checkpoints at 10 MiB.

## Verification and references

Tests use a fake GitHub transport to exercise the real request construction, checkpoint validation, conditional updates, timeout recovery, startup restoration, and learner integration without writing to your repository. Account permissions and actual Render deployment must be verified with the steps above.

Validation: 153 Python tests and 9 JavaScript tests passed; JavaScript syntax and Git whitespace checks passed. This includes A-starting Kolors through the full 14-card run and restoration of a previous 109-action GitHub checkpoint into the expanded model.

- [GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents)
- [Fine-grained personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Render deploy settings and skip phrases](https://render.com/docs/deploys)
