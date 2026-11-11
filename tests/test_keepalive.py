import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from app.keepalive import SelfPinger, start_self_ping


class KeepaliveTests(unittest.TestCase):
    def test_opt_in_and_render_url(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(start_self_ping())
        with patch.dict(os.environ, {"MURLAN_SELF_PING": "1", "RENDER_EXTERNAL_URL": "https://demo.onrender.com"}, clear=True):
            with patch("app.keepalive.SelfPinger") as constructor:
                result = start_self_ping()
                constructor.assert_called_once_with("https://demo.onrender.com")
                result.start.assert_called_once()

    def test_missing_or_invalid_url_disables_without_crashing(self):
        for url in ("", "http://example.com", "https://user:pass@example.com", "https://example.com/path"):
            with patch.dict(os.environ, {"MURLAN_SELF_PING": "1", "MURLAN_PUBLIC_URL": url}, clear=True):
                self.assertIsNone(start_self_ping())

    def test_network_error_retries_and_success_closes_response(self):
        pinger = SelfPinger("https://demo.onrender.com/", interval=0.01)
        received = threading.Event()
        response = MagicMock()
        response.__enter__.return_value.status = 200
        calls = 0
        def reply(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("offline")
            received.set()
            pinger.stop.set()
            return response
        with patch("app.keepalive.urlopen", side_effect=reply) as opening:
            pinger.start()
            try:
                self.assertTrue(received.wait(2))
            finally:
                pinger.close()
            self.assertEqual(opening.call_count, 2)
            request = opening.call_args.args[0]
            self.assertEqual(request.full_url, "https://demo.onrender.com/health")
            self.assertEqual(opening.call_args.kwargs["timeout"], 10)
            response.__exit__.assert_called_once()
            self.assertFalse(pinger.thread.is_alive())

    def test_shutdown_interrupts_thirteen_minute_wait(self):
        pinger = SelfPinger("https://demo.onrender.com")
        self.assertEqual(pinger.interval, 780)
        with patch("app.keepalive.urlopen") as opening:
            pinger.start()
            pinger.close()
            self.assertFalse(pinger.thread.is_alive())
            opening.assert_not_called()
