"""Run the Python app from tooling that supplies Vite-style preview flags."""

import argparse
import sys
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--strictPort", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    uvicorn.run("app.main:app", host=args.host, port=args.port)
