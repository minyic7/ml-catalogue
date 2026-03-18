"""Tests for the /api/chat endpoints."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import chat as chat_module


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Reset session store, lock, and cached client between tests."""
    chat_module._sessions.clear()
    chat_module._client = None
    chat_module._sessions_lock = asyncio.Lock()
    yield
    chat_module._sessions.clear()
    chat_module._client = None


@pytest.fixture()
def client():
    return TestClient(app)


def _mock_anthropic_response(text: str = "Mock response"):
    """Create a mock that mimics anthropic.Anthropic().messages.create(...)."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ── POST /api/chat ──────────────────────────────────────────────────────────


class TestChat:
    @patch.object(chat_module, "_get_client")
    def test_basic_message(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("Hello!")
        mock_get_client.return_value = mock_client

        resp = client.post("/api/chat", json={
            "session_id": "s1",
            "message": "Hi",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello!"
        assert data["session_id"] == "s1"
        mock_client.messages.create.assert_called_once()

    @patch.object(chat_module, "_get_client")
    def test_with_page_context(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("ctx reply")
        mock_get_client.return_value = mock_client

        resp = client.post("/api/chat", json={
            "session_id": "s2",
            "message": "Explain this",
            "page_context": "Chapter 1 content",
        })

        assert resp.status_code == 200
        # Verify system prompt was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Chapter 1 content" in call_kwargs["system"]

    @patch.object(chat_module, "_get_client")
    def test_with_image(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("I see the image")
        mock_get_client.return_value = mock_client

        resp = client.post("/api/chat", json={
            "session_id": "s3",
            "message": "What is this?",
            "image": "data:image/png;base64,iVBORw0KGgo=",
        })

        assert resp.status_code == 200
        assert resp.json()["response"] == "I see the image"
        # Verify image block was built
        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        user_content = messages[0]["content"]
        assert any(b.get("type") == "image" for b in user_content)

    @patch.object(chat_module, "_get_client")
    def test_multi_turn_conversation(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1
        mock_client.messages.create.return_value = _mock_anthropic_response("Answer 1")
        resp1 = client.post("/api/chat", json={"session_id": "s4", "message": "Q1"})
        assert resp1.status_code == 200

        # Turn 2
        mock_client.messages.create.return_value = _mock_anthropic_response("Answer 2")
        resp2 = client.post("/api/chat", json={"session_id": "s4", "message": "Q2"})
        assert resp2.status_code == 200
        assert resp2.json()["response"] == "Answer 2"

        # Session should now have 4 messages total (2 turns x user+assistant)
        session = chat_module._sessions["s4"]
        assert len(session["messages"]) == 4
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][1]["role"] == "assistant"
        assert session["messages"][2]["role"] == "user"
        assert session["messages"][3]["role"] == "assistant"

    def test_missing_message_field(self, client):
        resp = client.post("/api/chat", json={"session_id": "s5"})
        assert resp.status_code == 422

    def test_empty_message(self, client):
        resp = client.post("/api/chat", json={"session_id": "s5", "message": ""})
        assert resp.status_code == 422

    def test_empty_session_id(self, client):
        resp = client.post("/api/chat", json={"session_id": "", "message": "hi"})
        assert resp.status_code == 422

    def test_malformed_image_data_uri(self, client):
        resp = client.post("/api/chat", json={
            "session_id": "s-bad-img",
            "message": "look at this",
            "image": "data:image/png;base64",  # no comma
        })
        assert resp.status_code == 422
        assert "Malformed image data URI" in resp.json()["detail"]

    @patch.object(chat_module, "_get_client")
    def test_image_too_large(self, mock_get_client, client):
        mock_get_client.return_value = MagicMock()

        huge_image = "x" * (21 * 1024 * 1024)  # > 20 MB
        resp = client.post("/api/chat", json={
            "session_id": "s6",
            "message": "big image",
            "image": huge_image,
        })
        assert resp.status_code == 413

    @patch.object(chat_module, "_get_client")
    def test_api_error_removes_message(self, mock_get_client, client):
        import anthropic as anthropic_lib
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic_lib.APIError(
            message="rate limit",
            request=MagicMock(),
            body=None,
        )
        mock_get_client.return_value = mock_client

        resp = client.post("/api/chat", json={"session_id": "s7", "message": "test"})
        assert resp.status_code == 502
        # The failed message should have been removed from history
        assert len(chat_module._sessions["s7"]["messages"]) == 0


# ── POST /api/chat/compact ──────────────────────────────────────────────────


class TestCompact:
    @patch.object(chat_module, "_get_client")
    def test_session_not_found(self, mock_get_client, client):
        mock_get_client.return_value = MagicMock()
        resp = client.post("/api/chat/compact", json={"session_id": "nonexistent"})
        assert resp.status_code == 404

    @patch.object(chat_module, "_get_client")
    def test_compact_empty_session(self, mock_get_client, client):
        mock_get_client.return_value = MagicMock()
        # Create an empty session
        chat_module._sessions["empty"] = {"messages": [], "last_active": time.time()}

        resp = client.post("/api/chat/compact", json={"session_id": "empty"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_messages"] == 0
        assert data["compacted_messages"] == 0

    @patch.object(chat_module, "_get_client")
    def test_compact_replaces_history(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("Summary of conversation")
        mock_get_client.return_value = mock_client

        # Pre-populate session with messages
        chat_module._sessions["c1"] = {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            "last_active": time.time(),
        }

        resp = client.post("/api/chat/compact", json={"session_id": "c1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_messages"] == 4
        assert data["compacted_messages"] == 2

        # Session should now have exactly 2 messages
        session = chat_module._sessions["c1"]
        assert len(session["messages"]) == 2
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][1]["role"] == "assistant"
        assert "Summary of conversation" in session["messages"][0]["content"]


# ── GET /api/chat/context-usage/{session_id} ────────────────────────────────


class TestContextUsage:
    def test_session_not_found(self, client):
        resp = client.get("/api/chat/context-usage/missing")
        assert resp.status_code == 404

    def test_empty_session(self, client):
        chat_module._sessions["u1"] = {"messages": [], "last_active": time.time()}

        resp = client.get("/api/chat/context-usage/u1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "u1"
        assert data["estimated_tokens"] == 0
        assert data["max_tokens"] == 200_000
        assert data["usage_percent"] == 0.0
        assert data["message_count"] == 0

    def test_with_messages(self, client):
        # Use enough text so that estimated tokens > 0 after integer division
        long_text = "x" * 400  # 400 chars -> 100 tokens
        chat_module._sessions["u2"] = {
            "messages": [
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": long_text},
            ],
            "last_active": time.time(),
        }

        resp = client.get("/api/chat/context-usage/u2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_tokens"] == 200
        assert data["message_count"] == 2
        assert data["usage_percent"] > 0

    def test_with_image_content(self, client):
        chat_module._sessions["u3"] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "abc"}},
                        {"type": "text", "text": "What is this?"},
                    ],
                },
            ],
            "last_active": time.time(),
        }

        resp = client.get("/api/chat/context-usage/u3")
        assert resp.status_code == 200
        data = resp.json()
        # Image adds ~1000 tokens (4000 chars / 4)
        assert data["estimated_tokens"] >= 1000


# ── Session TTL & LRU eviction ──────────────────────────────────────────────


class TestSessionManagement:
    def test_expired_sessions_evicted(self, client):
        # Create an expired session
        chat_module._sessions["old"] = {
            "messages": [{"role": "user", "content": "old msg"}],
            "last_active": time.time() - chat_module._SESSION_TTL - 1,
        }
        # Create a fresh session to trigger eviction
        asyncio.run(chat_module._get_or_create_session("new"))

        assert "old" not in chat_module._sessions
        assert "new" in chat_module._sessions

    def test_lru_eviction(self, client):
        original_max = chat_module._MAX_SESSIONS
        chat_module._MAX_SESSIONS = 3
        try:
            for i in range(3):
                asyncio.run(chat_module._get_or_create_session(f"s{i}"))
            # All 3 should exist
            assert len(chat_module._sessions) == 3

            # Adding one more should evict the oldest (s0)
            asyncio.run(chat_module._get_or_create_session("s3"))
            assert len(chat_module._sessions) == 3
            assert "s0" not in chat_module._sessions
            assert "s3" in chat_module._sessions
        finally:
            chat_module._MAX_SESSIONS = original_max

    def test_accessing_session_moves_to_end(self, client):
        asyncio.run(chat_module._get_or_create_session("a"))
        asyncio.run(chat_module._get_or_create_session("b"))
        asyncio.run(chat_module._get_or_create_session("c"))

        # Access "a" to move it to the end
        asyncio.run(chat_module._get_or_create_session("a"))

        keys = list(chat_module._sessions.keys())
        assert keys[-1] == "a"
