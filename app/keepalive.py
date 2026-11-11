"""Optional best-effort public health checks; not persistent storage."""

import logging
import os
import threading
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

logger = logging.getLogger("uvicorn.error")


class SelfPinger:
    def __init__(self, base_url, interval=13 * 60):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("Self-ping needs an HTTPS origin without credentials or a path")
        self.url = base_url.rstrip("/") + "/health"
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="murlan-self-ping", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                request = Request(self.url, headers={"User-Agent": "Murlan-HealthCheck/1.0"})
                with urlopen(request, timeout=10) as response:
                    logger.info("Self-ping /health: HTTP %s", response.status)
            except Exception as exc:
                # A temporary network error must not terminate future checks.
                logger.warning("Self-ping failed (%s); retrying next interval", type(exc).__name__)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def start_self_ping():
    if os.getenv("MURLAN_SELF_PING", "0") != "1":
        return None
    base_url = os.getenv("MURLAN_PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL", "")
    if not base_url:
        logger.warning("Self-ping disabled: no public URL is configured")
        return None
    try:
        pinger = SelfPinger(base_url)
    except ValueError as exc:
        logger.warning("Self-ping disabled: %s", exc)
        return None
    pinger.start()
    logger.info("Self-ping enabled: public /health every 13 minutes")
    return pinger
