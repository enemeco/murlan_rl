import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from app.github_models import GitHubModelStore, GitHubSyncError, NoRedirect, blob_sha
from app.game_manager import MurlanBot
from app.main import create_app
from app.online_learning import OnlineLearner
from murlan.checkpoint import load_model, policy_fingerprint, save_checkpoint

torch.set_num_threads(1)
TOKEN = "test-only-secret-not-a-real-token"


class FakeGitHub:
    """Exercise actual JSON/HTTP request construction, without a GitHub write."""
    def __init__(self, content):
        self.content = content
        self.requests = []
        self.commits = []
        self.lose_put_response = False
        self.http_error = None
        self.omit_content = False

    def open(self, request, timeout):
        assert timeout == 15
        assert request.get_header("Authorization") == "Bearer " + TOKEN
        assert request.get_header("X-github-api-version") == "2026-03-10"
        if request.method == "PUT":
            assert request.get_header("Accept") == "application/vnd.github+json"
        url = urlsplit(request.full_url)
        assert url.hostname == "api.github.com"
        self.requests.append(request)
        if self.http_error:
            code, headers = self.http_error
            raise HTTPError(request.full_url, code, TOKEN, headers, io.BytesIO(TOKEN.encode()))
        sha = blob_sha(self.content)
        if request.method == "GET":
            if "/git/blobs/" in url.path:
                assert url.path.endswith(sha)
            else:
                assert parse_qs(url.query) == {"ref": ["main"]}
            result = dict(type="file", sha=sha, size=len(self.content), encoding="base64",
                          content=base64.b64encode(self.content).decode())
            if self.omit_content and "/git/blobs/" not in url.path:
                result.update(content="", encoding="none")
        else:
            assert request.method == "PUT"
            body = json.loads(request.data)
            assert body["branch"] == "main"
            assert body["sha"] == sha
            assert "[skip render]" in body["message"]
            assert "[skip ci]" in body["message"]
            self.content = base64.b64decode(body["content"])
            self.commits.append(body)
            if self.lose_put_response:
                self.lose_put_response = False
                raise URLError(TOKEN)
            result = {"content": {"sha": blob_sha(self.content)}}
        return io.BytesIO(json.dumps(result).encode())


@pytest.fixture
def remote(tmp_path):
    model, _ = load_model("checkpoints/improved_policy.pt")
    seed = tmp_path / "seed.pt"
    save_checkpoint(seed, model, version=3)
    return FakeGitHub(seed.read_bytes())


def make_store(remote):
    store = GitHubModelStore(TOKEN, "enemeco/murlan_rl")
    store._opener = remote
    return store


def candidate(tmp_path, store, source, name="accepted.pt", increment=0.01, promote=True):
    model, _ = load_model(source)
    with torch.no_grad():
        next(model.parameters()).add_(increment)
    path = tmp_path / name
    save_checkpoint(path, model, version=store.version + 1, validation={
        "gate": {"promote": promote}, "source_policy_id": store.policy_id,
    })
    return path


def test_restore_replaces_old_image_and_successful_save_survives_restart(tmp_path, remote):
    store = make_store(remote)
    local = tmp_path / "champion.pt"
    local.write_bytes(b"old deployed image")
    model, version = store.restore(local)
    assert version == 3
    assert local.read_bytes() == remote.content
    assert store.status()["state"] == "restored"
    accepted = candidate(tmp_path, store, local)
    store.publish(accepted)
    assert len(remote.commits) == 1
    assert remote.content == accepted.read_bytes()
    assert store.status()["saved_version"] == 4
    # New container has neither the working model nor its local replay database.
    new_store = make_store(remote)
    new_local = tmp_path / "new-container" / "champion.pt"
    restored, version = new_store.restore(new_local)
    assert version == 4
    assert policy_fingerprint(restored) != policy_fingerprint(model)
    assert new_local.read_bytes() == accepted.read_bytes()
    assert TOKEN not in json.dumps(new_store.status())


def test_put_timeout_is_verified_without_duplicate_commit(tmp_path, remote):
    store = make_store(remote)
    local = tmp_path / "champion.pt"
    store.restore(local)
    accepted = candidate(tmp_path, store, local)
    remote.lose_put_response = True
    with pytest.raises(GitHubSyncError) as error:
        store.publish(accepted)
    assert TOKEN not in str(error.value)
    assert store.version == 3
    assert len(remote.commits) == 1
    store.publish(accepted)
    assert len(remote.commits) == 1
    assert store.version == 4


def test_stale_instance_cannot_overwrite_new_champion(tmp_path, remote):
    first, second = make_store(remote), make_store(remote)
    local = tmp_path / "champion.pt"
    first.restore(local)
    second.restore(tmp_path / "other.pt")
    accepted = candidate(tmp_path, first, local)
    stale = candidate(tmp_path, second, local, name="stale.pt", increment=0.02)
    first.publish(accepted)
    with pytest.raises(GitHubSyncError, match="changed concurrently"):
        second.publish(stale)
    assert remote.content == accepted.read_bytes()
    assert len(remote.commits) == 1


def test_rejected_candidate_and_wrong_parent_never_write(tmp_path, remote):
    store = make_store(remote)
    local = tmp_path / "champion.pt"
    store.restore(local)
    rejected = candidate(tmp_path, store, local, promote=False)
    with pytest.raises(GitHubSyncError, match="passed evaluation"):
        store.publish(rejected)
    accepted = candidate(tmp_path, store, local)
    store.policy_id = "another-policy"
    with pytest.raises(GitHubSyncError, match="evaluated against"):
        store.publish(accepted)
    assert remote.commits == []


@pytest.mark.parametrize("invalid", [b"not a torch checkpoint", b"version https://git-lfs.github.com/spec/v1"])
def test_invalid_remote_never_replaces_local(tmp_path, invalid):
    store = make_store(FakeGitHub(invalid))
    local = tmp_path / "champion.pt"
    local.write_bytes(b"keep me")
    with pytest.raises(GitHubSyncError, match="invalid or incompatible"):
        store.restore(local)
    assert local.read_bytes() == b"keep me"
    assert not store.restored


def test_large_content_fetch_uses_immutable_blob(tmp_path, remote):
    remote.omit_content = True
    store = make_store(remote)
    store.restore(tmp_path / "champion.pt")
    assert "/git/blobs/" in remote.requests[-1].full_url


@pytest.mark.parametrize("code", [401, 403, 404, 409, 422, 429, 503])
def test_http_errors_are_safe_and_back_off(tmp_path, remote, code, caplog):
    store = make_store(remote)
    remote.http_error = (code, {"Retry-After": "120"})
    with pytest.raises(GitHubSyncError) as error:
        store.restore(tmp_path / "champion.pt")
    store.defer(error.value)
    assert not store.can_retry()
    if code in (403, 429):
        assert store.status()["retry_in_seconds"] >= 120
    assert TOKEN not in str(error.value) + caplog.text + json.dumps(store.status())


def test_disabled_sync_and_missing_credentials(monkeypatch):
    monkeypatch.delenv("MURLAN_GITHUB_SYNC", raising=False)
    monkeypatch.delenv("MURLAN_GITHUB_TOKEN", raising=False)
    assert GitHubModelStore.from_env() is None
    monkeypatch.setenv("MURLAN_GITHUB_SYNC", "1")
    with pytest.raises(ValueError, match="MURLAN_GITHUB_TOKEN"):
        GitHubModelStore.from_env()
    assert NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com") is None


@pytest.mark.parametrize("path", ["../model.pt", ".github/workflows/model.pt", "checkpoints/../x.pt", "checkpoints/model.txt"])
def test_only_checkpoint_paths_allowed(path):
    with pytest.raises(ValueError, match="under checkpoints"):
        GitHubModelStore(TOKEN, "enemeco/murlan_rl", path=path)


def learner_fixture(tmp_path, remote, monkeypatch, promote=True):
    store = make_store(remote)
    local = tmp_path / "champion.pt"
    store.restore(local)
    bot = MurlanBot(local)
    replay = Mock()
    replay.pending.return_value = ([1], [])
    replay.demonstrations.return_value = []
    replay.stats.return_value = {"updates": 0}
    evaluate = Mock(return_value=[])
    monkeypatch.setattr("app.online_learning.evaluate", evaluate)
    monkeypatch.setattr("app.online_learning.summarize", lambda result: {})
    monkeypatch.setattr("app.online_learning.promotion_decision", lambda *args: {"promote": promote})
    learner = OnlineLearner(bot, replay, min_hands=1, start=False, model_store=store)
    return learner, store, bot, replay, evaluate


def test_failed_upload_retries_same_candidate_before_promoting(tmp_path, remote, monkeypatch):
    learner, store, bot, replay, evaluate = learner_fixture(tmp_path, remote, monkeypatch)
    remote.http_error = (503, {})
    assert learner.train_once() is None
    assert learner.status == "saving"
    assert bot.version == 3
    replay.mark_processed.assert_not_called()
    assert len(evaluate.call_args_list) == 2
    learner.train_once()  # backoff does not re-evaluate or start another candidate
    assert len(evaluate.call_args_list) == 2
    remote.http_error = None
    store.retry_at = 0
    assert learner.train_once()["gate"]["promote"]
    assert bot.version == 4
    assert len(remote.commits) == 1
    assert len(evaluate.call_args_list) == 2
    replay.mark_processed.assert_called_once_with([1])
    assert MurlanBot(bot.ckpt_path).version == 4


def test_rejected_evaluation_never_calls_github(tmp_path, remote, monkeypatch):
    learner, store, bot, replay, evaluate = learner_fixture(tmp_path, remote, monkeypatch, promote=False)
    requests_before = len(remote.requests)
    learner.train_once()
    assert learner.status == "retained"
    assert len(remote.requests) == requests_before
    assert not remote.commits
    assert bot.version == 3


def test_app_restores_from_github_and_exposes_safe_status(tmp_path, remote, monkeypatch):
    store = make_store(remote)
    monkeypatch.setattr(GitHubModelStore, "from_env", lambda: store)
    monkeypatch.setenv("MURLAN_SELF_PING", "0")
    app = create_app(tmp_path / "champion.pt", tmp_path / "replay.db", online=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == 3
        status = client.get("/api/learning").json()["github_sync"]
        assert status["saved_version"] == 3
        assert status["state"] == "restored"
        assert TOKEN not in json.dumps(status)


def test_startup_outage_keeps_game_available_and_pauses_training(tmp_path, remote, monkeypatch):
    store = make_store(remote)
    remote.http_error = (503, {})
    monkeypatch.setattr(GitHubModelStore, "from_env", lambda: store)
    monkeypatch.setenv("MURLAN_SELF_PING", "0")
    app = create_app(tmp_path / "champion.pt", tmp_path / "replay.db", online=True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/api/new", json={}).status_code == 200
        assert client.get("/api/learning").json()["github_sync"]["state"] == "retrying"
        assert not remote.commits
        remote.http_error = None
        store.retry_at = 0
        app.state.learner.train_once()
        assert app.state.bot.version == 3
        assert store.restored


def test_github_restores_previous_109_action_model_then_saves_new_schema(tmp_path, remote):
    from murlan.actions import ACTION_DIM, SWAP_BASE
    payload = torch.load(io.BytesIO(remote.content), weights_only=True)
    for key in ('pi.weight', 'pi.bias'):
        source = payload['model'][key]
        payload['model'][key] = torch.cat([source[:101], source[SWAP_BASE:]])
    payload.update(schema_version=4, action_dim=109)
    encoded = io.BytesIO()
    torch.save(payload, encoded)
    remote.content = encoded.getvalue()
    store = make_store(remote)
    local = tmp_path / 'restored-legacy.pt'
    model, version = store.restore(local)
    assert model.pi.out_features == ACTION_DIM == 118
    accepted = candidate(tmp_path, store, local)
    store.publish(accepted)
    saved = torch.load(io.BytesIO(remote.content), weights_only=True)
    assert saved['schema_version'] == 5 and saved['action_dim'] == 118
    assert saved['version'] == version + 1
