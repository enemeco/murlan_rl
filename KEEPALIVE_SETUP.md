# Render Free: optional 13-minute self-ping

Copy the update files into your existing improved Murlan repository, preserving
the `app/` and `tests/` folders. No trained weights are changed.

Commit and push to main:

```powershell
git add app/main.py app/keepalive.py render.yaml tests/test_keepalive.py KEEPALIVE_SETUP.md README.md
git commit -m "Add optional Render self-ping every 13 minutes"
git push origin main
```

On an existing Render service, set `MURLAN_SELF_PING=1` under Environment.
If you want automatic candidate training, also set `MURLAN_ONLINE_LEARNING=1`.
Then deploy the latest commit. A Blueprint can apply the included render.yaml;
an ordinary web service does not automatically apply that file.

Render supplies RENDER_EXTERNAL_URL automatically. The app sends a public HTTPS
GET to that URL's `/health` endpoint every 780 seconds while the process is
running. No browser tab needs to remain open. Local runs do not self-ping unless
explicitly enabled; outside Render, set MURLAN_PUBLIC_URL to your HTTPS origin.

Startup logs show `Self-ping enabled: public /health every 13 minutes`.
After the first 13 minutes, a successful request logs `Self-ping /health: HTTP 200`.
Network failures are logged and retried at the next interval. Set
MURLAN_SELF_PING=0 to disable it. Requests time out after ten seconds.

This is best effort, not an uptime guarantee. Render may not treat self-generated
requests as qualifying activity. Once the process stops, it cannot ping itself
awake. Deployments, restarts, resource limits, and monthly limits still apply.
Keeping a service active consumes its free instance-hour allowance.

The included Free configuration enables candidate learning as requested, but
PyTorch inference plus training has not been memory-tested on Render Free.
If it runs out of memory, disable online learning or move training elsewhere.
Self-pinging provides NO persistent storage: local replay data and newly trained
weights can still be lost on sleep, restart, or redeploy. It does not replace
external storage or backups, and this update does not implement replay pruning.

Checks: four self-ping tests pass (enable/disable, URL handling, network retry,
and prompt shutdown); changed Python modules compile. No live Render deployment
or 13-minute Render idle-behavior test was performed.
