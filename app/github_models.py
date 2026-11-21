"""Save accepted champions to GitHub and restore them independently of the image."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import torch

from murlan.checkpoint import load_model, policy_fingerprint

log = logging.getLogger("uvicorn.error")
MAX_MODEL_BYTES = 10 * 1024 * 1024


class GitHubSyncError(RuntimeError):
    def __init__(self, message, retry_after=60):
        super().__init__(message)
        self.retry_after = retry_after


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the repository token to a redirect destination.
        return None


def blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def validate_checkpoint(path):
    try:
        model, metadata = load_model(path)  # torch.load uses weights_only=True
        version = metadata.get("version", 0)
        if type(version) is not int or version < 0:
            raise ValueError("Invalid version")
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("Non-finite weights")
        return model, metadata
    except Exception:
        raise GitHubSyncError("The checkpoint is invalid or incompatible with this app.") from None


class GitHubModelStore:
    def __init__(self, token, repo, branch="main", path="checkpoints/improved_policy.pt"):
        if not token or not token.strip():
            raise ValueError("Set MURLAN_GITHUB_TOKEN before enabling GitHub model saving.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("MURLAN_GITHUB_REPO must be owner/repository.")
        if not branch or any(ord(c) < 32 for c in branch):
            raise ValueError("MURLAN_GITHUB_BRANCH must name a branch.")
        parts = path.split("/")
        if (not path.endswith(".pt") or parts[0] != "checkpoints"
                or any(p in ("", ".", "..") for p in parts) or "\\" in path):
            raise ValueError("MURLAN_GITHUB_MODEL_PATH must be a .pt file under checkpoints/.")
        self._token = token.strip()
        self.repo, self.branch, self.path = repo, branch, path
        self._base = "https://api.github.com/repos/" + repo
        self._contents = "/contents/" + quote(path, safe="/")
        self._opener = build_opener(NoRedirect())
        self.sha = None
        self.version = None
        self.policy_id = None
        self.restored = False
        self.retry_at = 0.0
        self.failures = 0
        self._status = dict(enabled=True, state="starting", saved_version=None, error=None)

    @classmethod
    def from_env(cls):
        if os.getenv("MURLAN_GITHUB_SYNC", "0") != "1":
            return None
        return cls(
            os.getenv("MURLAN_GITHUB_TOKEN", ""),
            os.getenv("MURLAN_GITHUB_REPO", "enemeco/murlan_rl"),
            os.getenv("MURLAN_GITHUB_BRANCH", "main"),
            os.getenv("MURLAN_GITHUB_MODEL_PATH", "checkpoints/improved_policy.pt"),
        )

    def status(self):
        return dict(self._status)

    def can_retry(self):
        return time.monotonic() >= self.retry_at

    def defer(self, error):
        self.failures += 1
        delay = max(error.retry_after, min(900, 60 * 2 ** min(self.failures - 1, 4)))
        self.retry_at = time.monotonic() + delay
        self._status = dict(enabled=True, state="retrying", saved_version=self.version,
                            error=str(error), retry_in_seconds=delay)
        log.warning("GitHub model sync: %s Retrying in %s seconds.", error, delay)

    def _success(self, state, version):
        self.failures, self.retry_at = 0, 0.0
        self.version = version
        self._status = dict(enabled=True, state=state, saved_version=version, error=None)

    def _request(self, method, endpoint, payload=None):
        request = Request(
            self._base + endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            method=method,
            headers={
                "Authorization": "Bearer " + self._token,
                "Accept": ("application/vnd.github.object+json"
                           if method == "GET" and endpoint.startswith("/contents/")
                           else "application/vnd.github+json"),
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "Murlan-Model-Sync/1.0",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
        )
        try:
            with self._opener.open(request, timeout=15) as response:
                body = response.read(2 * MAX_MODEL_BYTES + 1)
            if len(body) > 2 * MAX_MODEL_BYTES:
                raise GitHubSyncError("GitHub returned an oversized checkpoint response.")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("Expected object")
            return result
        except HTTPError as exc:
            code = exc.code
            delay = 60
            if code in (403, 429):
                try:
                    delay = max(delay, int(exc.headers.get("Retry-After", "0")))
                    if exc.headers.get("X-RateLimit-Remaining") == "0":
                        delay = max(delay, int(exc.headers.get("X-RateLimit-Reset", "0")) - int(time.time()) + 1)
                except (TypeError, ValueError):
                    pass
            messages = {
                401: "GitHub authentication failed. Replace the expired or invalid token in Render.",
                403: "GitHub denied the request. Check Contents write permission, branch rules, and rate limits.",
                404: "GitHub could not find the checkpoint. Check repository, branch, file path, and token access.",
                409: "The GitHub model changed concurrently. Restart to load the latest champion.",
                422: "GitHub rejected the model commit. Check branch rules and repository access.",
                429: "GitHub rate limit reached.",
            }
            exc.close()
            raise GitHubSyncError(messages.get(code, f"GitHub returned HTTP {code}."), delay) from None
        except (URLError, TimeoutError, OSError):
            raise GitHubSyncError("GitHub could not be reached; the model has not been confirmed saved.") from None
        except (ValueError, TypeError):
            raise GitHubSyncError("GitHub returned an invalid response.") from None

    def _remote(self):
        result = self._request("GET", self._contents + "?" + urlencode({"ref": self.branch}))
        if (result.get("type") != "file"
                or not re.fullmatch(r"[a-f0-9]{40}", result.get("sha", ""))
                or not isinstance(result.get("size"), int)
                or not 0 < result["size"] <= MAX_MODEL_BYTES):
            raise GitHubSyncError("The GitHub checkpoint must be a regular file of at most 10 MiB.")
        return result

    def restore(self, local_path):
        remote = self._remote()
        data = remote
        if data.get("encoding") != "base64":
            # Files above 1 MiB omit contents. Fetch the immutable blob, not a
            # branch URL that could change between the metadata and byte reads.
            data = self._request("GET", "/git/blobs/" + remote["sha"])
        try:
            if data.get("encoding") != "base64":
                raise ValueError("Expected base64")
            content = base64.b64decode("".join(data["content"].split()), validate=True)
        except (KeyError, TypeError, ValueError):
            raise GitHubSyncError("GitHub returned invalid checkpoint contents.") from None
        if len(content) != remote["size"] or blob_sha(content) != remote["sha"]:
            raise GitHubSyncError("GitHub checkpoint integrity check failed.")
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".restore.pt")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            model, metadata = validate_checkpoint(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.sha, self.restored = remote["sha"], True
        self.policy_id = policy_fingerprint(model)
        self._success("restored", metadata.get("version", 0))
        log.info("GitHub model restored: version %s from %s (%s).", self.version, self.path, self.branch)
        return model, self.version

    def publish(self, checkpoint):
        if not self.restored:
            raise GitHubSyncError("Restore the GitHub champion before saving improvements.")
        path = Path(checkpoint)
        if not 0 < path.stat().st_size <= MAX_MODEL_BYTES:
            raise GitHubSyncError("The model exceeds the 10 MiB backup limit.")
        model, metadata = validate_checkpoint(path)
        validation = metadata.get("validation", {})
        if validation.get("gate", {}).get("promote") is not True:
            raise GitHubSyncError("Only a candidate that passed evaluation can be saved.")
        content = path.read_bytes()
        wanted_sha = blob_sha(content)
        remote = self._remote()
        # A timed-out PUT might already have succeeded. Confirm its exact bytes
        # instead of making another commit or rejecting our own successful save.
        if remote["sha"] != wanted_sha:
            if remote["sha"] != self.sha:
                raise GitHubSyncError("The GitHub model changed concurrently. Restart to load the latest champion.")
            if (metadata.get("version") != self.version + 1
                    or validation.get("source_policy_id") != self.policy_id):
                raise GitHubSyncError("The candidate was not evaluated against the saved GitHub champion.")
            result = self._request("PUT", self._contents, {
                "message": f"Save accepted Murlan model v{metadata['version']} [skip render] [skip ci]",
                "branch": self.branch,
                "sha": self.sha,
                "content": base64.b64encode(content).decode("ascii"),
            })
            if result.get("content", {}).get("sha") != wanted_sha:
                raise GitHubSyncError("GitHub did not confirm the uploaded model; retrying verification.")
        self.sha = wanted_sha
        self.policy_id = policy_fingerprint(model)
        self._success("saved", metadata["version"])
        log.info("GitHub model saved: accepted version %s to %s (%s).", self.version, self.path, self.branch)
