from __future__ import annotations
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .game_manager import GameManager, MurlanBot
from .online_learning import OnlineLearner
from .replay import ReplayStore
from .keepalive import start_self_ping
from murlan.actions import ACTION_DIM
from murlan.checkpoint import load_model, save_checkpoint

ROOT = Path(__file__).resolve().parent.parent


class NewGameRequest(BaseModel):
    mode: str = "learn"
    learning_enabled: bool = True
    human_seat: int = Field(default=0, ge=0, le=3, strict=True)


class ActRequest(BaseModel):
    action: int = Field(ge=0, lt=ACTION_DIM, strict=True)
    revision: int = Field(ge=0, strict=True)
    request_id: str = Field(min_length=8, max_length=80)


def create_app(model_path=None, replay_path=None, online=None):
    seed_model = model_path is None
    model_path = model_path or os.getenv(
        "MURLAN_MODEL_PATH", str(ROOT / "data/champion.pt")
    )
    replay_path = replay_path or os.getenv(
        "MURLAN_REPLAY_PATH", str(ROOT / "data/replay.sqlite3")
    )
    enabled = (
        online
        if online is not None
        else os.getenv("MURLAN_ONLINE_LEARNING", "1") == "1"
    )

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv("MURLAN_TORCH_THREADS", "1")))
        replay = ReplayStore(replay_path)
        if seed_model and not Path(model_path).exists():
            default_seed = ROOT / "checkpoints/improved_policy.pt"
            if not default_seed.exists():
                default_seed = ROOT / "checkpoints/murlan_policy.pt"
            initial = Path(os.getenv("MURLAN_INITIAL_MODEL_PATH", str(default_seed)))
            if initial.is_file():
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                seed, metadata = load_model(initial)
                save_checkpoint(
                    model_path,
                    seed,
                    version=int(metadata.get("version", 0)),
                    provenance=str(initial),
                )
        bot = MurlanBot(model_path)
        learner = (
            OnlineLearner(
                bot, replay, eval_deals=int(os.getenv("MURLAN_EVAL_DEALS", "12"))
            )
            if enabled
            else None
        )
        app.state.manager = GameManager(bot, learner, replay)
        app.state.bot, app.state.learner, app.state.replay = bot, learner, replay
        pinger = start_self_ping()
        try:
            yield
        finally:
            if pinger:
                pinger.close()
            if learner:
                learner.close()

    app = FastAPI(title="Murlan · Learning Table", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static/index.html")

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, "Game not found or expired")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    @app.post("/api/new")
    def new_game(req: NewGameRequest | None = None):
        req = req or NewGameRequest()
        m = app.state.manager
        gid = call(
            m.new_game,
            human_seat=req.human_seat,
            mode=req.mode,
            learn=req.learning_enabled,
        )
        return m.get_state(gid)

    @app.get("/api/state/{gid}")
    def state(gid: str):
        return call(app.state.manager.get_state, gid)

    @app.post("/api/act/{gid}")
    def act(gid: str, req: ActRequest):
        return call(
            app.state.manager.human_act, gid, req.action, req.revision, req.request_id
        )

    @app.delete("/api/game/{gid}")
    def abandon(gid: str):
        m = app.state.manager
        s = call(m._session, gid)
        with s.lock, m.lock:
            m.games.pop(gid, None)
        return {"ok": True}

    @app.get("/api/learning")
    def learning():
        learner = app.state.learner
        bot = app.state.bot
        return dict(
            **app.state.replay.stats(),
            enabled=enabled,
            phase=learner.status if learner else "disabled",
            version=bot.version,
            bot_kind="strategic" if bot.fallback else "policy",
            min_hands=learner.min_hands if learner else 0,
            min_decisions=learner.min_decisions if learner else 0,
            error=(
                "The last candidate update failed; current bots remain available."
                if learner and learner.error
                else None
            ),
        )

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_version": app.state.bot.version,
            "online_learning": enabled,
        }

    @app.get("/download-model")
    def download_model(x_api_key: str | None = Header(default=None)):
        secret = os.getenv("MODEL_DOWNLOAD_KEY")
        if not secret or not x_api_key or not hmac.compare_digest(secret, x_api_key):
            raise HTTPException(401, "Unauthorized")
        if not Path(model_path).is_file():
            raise HTTPException(404, "No trained checkpoint yet")
        return FileResponse(model_path, filename="murlan_policy.pt")

    return app


app = create_app()
