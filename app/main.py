from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .game_manager import GameManager, MurlanBot
from .online_learning import OnlineLearner

MODEL_PATH = os.getenv("MURLAN_MODEL_PATH", "checkpoints/murlan_policy.pt")

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