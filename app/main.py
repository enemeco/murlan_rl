from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import threading
import time
import requests
from .game_manager import GameManager, MurlanBot
from .online_learning import OnlineLearner

MODEL_PATH = os.getenv("MURLAN_MODEL_PATH", "checkpoints/murlan_policy.pt")
SECRET_KEY = os.environ.get("MODEL_DOWNLOAD_KEY")

app = FastAPI(title="Murlan RL")

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

bot = MurlanBot(MODEL_PATH, device="cpu")
learner = OnlineLearner(bot.model, device="cpu", lock=bot.lock)
manager = GameManager(bot=bot, learner=learner)

class ActRequest(BaseModel):
    action: int

@app.get("/", response_class=HTMLResponse)
def index():
    with open(static_dir / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/new")
def new_game():
    gid = manager.new_game(human_seat=0)
    return manager.get_state(gid)

@app.get("/api/state/{gid}")
def state(gid: str):
    return manager.get_state(gid)

@app.post("/api/act/{gid}")
def act(gid: str, req: ActRequest):
    return manager.human_act(gid, req.action)


# ... your existing imports and code ...

def keep_alive():
    """Keep the Render app awake by pinging itself every 13 minutes"""
    time.sleep(30)

    app_url = "https://murlan-rl.onrender.com/health"  # <-- better than "/"
    while True:
        try:
            time.sleep(13 * 60)
            r = requests.get(app_url, timeout=10)
            print(f"Keep-alive ping: {r.status_code}")
        except Exception as e:
            print(f"Keep-alive error: {e}")

@app.on_event("startup")
def start_keep_alive_thread():
    # optional: disable locally with ENABLE_KEEPALIVE=0
    if os.getenv("ENABLE_KEEPALIVE", "1") == "1":
        threading.Thread(target=keep_alive, daemon=True).start()
        
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/download-model")
def download_model(x_api_key: str = Header(None)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(MODEL_PATH, filename="murlan_policy.pt")
