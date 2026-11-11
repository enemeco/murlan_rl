# Murlan interface update

This package updates the improved Murlan project from this conversation.

## What changed

- Natural Albanian interface text, using **robotët** throughout.
- Click or tap cards to select them; click again to deselect them.
- **Luaj letrat** activates only when the selection is legal on your turn.
- The exact cards and suits you select are played and shown in the move history.
- **Pas** passes; **Jep letrën** confirms the card exchange between hands.
- Three distinct Albanian names are assigned to the robots in each new match. Names stay the same across hands and page reloads, and all three change in the next match.
- **Stërvitje** lets the robots explore different moves. **Sfidë** uses their strongest predicted move. The training checkbox applies to the next match.

The learning pipeline and trained model are retained. This update does not replace your checkpoints, replay database, environment settings, or `render.yaml`. The existing optional keepalive helper is included for compatibility.

## Install

1. Extract `murlan_interface_update.zip`.
2. Copy its **contents** into your existing repository folder and replace the matching files. Copy into the folder containing `app`, `murlan`, `static`, and `.git`; do not create another nested project folder. Based on your earlier terminal output, this is `C:\Users\Ene\Desktop\murlan_rl\murlan_rl`.
3. Open PowerShell in that repository folder and run:

```powershell
git status
git add .
git commit -m "Improve Albanian interface and manual card selection"
git push origin main
```

These commands update your existing `main` branch. GitHub must authenticate as an account with write access to `enemeco/murlan_rl`.

4. Let Render deploy the new commit. If automatic deployment is disabled, deploy the latest commit from your service dashboard.
5. Refresh the game with **Ctrl+F5**. A server restart starts a new game session.

## Validation

- 37 Python tests passed, including full matches using manual card selection, exact-suit play, opening rules, exchanges, request retries, and name persistence.
- 5 JavaScript selection tests passed.
- Browser checks covered card clicks, keyboard selection, illegal selections, valid play, passing, new-match names, reloading, rules, and a 390-pixel layout without horizontal overflow.

To rerun the automated checks:

```bash
python -m pytest -q
node --test tests/test_selection.cjs
```

Node.js is needed only for the JavaScript tests; the game still runs with the existing Python setup.
