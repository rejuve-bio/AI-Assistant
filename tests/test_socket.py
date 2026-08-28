"""
Tests for Socket.IO event handlers (connect, disconnect, join_room, question).

Uses a lightweight FakeSio stand-in that records all calls made by the
handlers, so we can assert correct behavior without needing a live server.
"""
import asyncio
import os

import jwt
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")

JWT_SECRET = os.getenv("JWT_SECRET")
if JWT_SECRET is None:
    raise ValueError("JWT_SECRET environment variable is not set.")

TEST_USER_ID = "socket_test_user"
TEST_TOKEN = jwt.encode({"user_id": TEST_USER_ID}, JWT_SECRET, algorithm="HS256")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_environ(token=None):
    """Build a minimal WSGI-like environ dict with an Authorization header."""
    env = {}
    if token is not None:
        env["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    return env


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# FakeSio — lightweight stand-in for socketio.AsyncServer
# ---------------------------------------------------------------------------

class FakeSio:
    """Records every call the handlers make so tests can assert on them."""

    def __init__(self):
        self.handlers = {}
        self.sessions = {}
        self.rooms = {}       # sid → set of room names
        self.emitted = []     # list of (event, data, room)

    # --- decorator used by register_socket_events ---
    def event(self, func):
        self.handlers[func.__name__] = func
        return func

    # --- session management ---
    async def save_session(self, sid, data):
        self.sessions[sid] = data

    async def get_session(self, sid):
        return self.sessions.get(sid, {})

    # --- room management ---
    async def enter_room(self, sid, room):
        self.rooms.setdefault(sid, set()).add(room)

    async def leave_room(self, sid, room):
        if sid in self.rooms:
            self.rooms[sid].discard(room)

    # --- emitting ---
    async def emit(self, event, data, room=None, **kwargs):
        self.emitted.append((event, data, room))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sio():
    return FakeSio()


@pytest.fixture
def mock_app():
    app = MagicMock()
    assistant = MagicMock()
    assistant.agent = MagicMock(
        return_value={"text": "Test answer", "json_format": None}
    )
    app.state.ai_assistant = assistant
    return app


@pytest.fixture
def handlers(sio, mock_app):
    from app.socket_manager import register_socket_events
    register_socket_events(sio, mock_app)
    return sio.handlers


# ===================================================================
# Test: connect
# ===================================================================

class TestConnect:
    def test_valid_token_accepts_connection(self, handlers, sio):
        result = _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        # None means accepted (only `return False` rejects)
        assert result is None

    def test_valid_token_saves_session(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        assert sio.sessions["sid1"]["user_id"] == TEST_USER_ID
        assert sio.sessions["sid1"]["token"] == TEST_TOKEN

    def test_valid_token_auto_joins_user_room(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        assert TEST_USER_ID in sio.rooms.get("sid1", set())

    def test_valid_token_emits_success_message(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        response_emits = [(e, d) for e, d, r in sio.emitted if e == "response"]
        assert len(response_emits) == 1
        assert response_emits[0][1]["message"] == "Connected successfully"

    def test_no_token_rejects(self, handlers, sio):
        result = _run(handlers["connect"]("sid2", {}))
        assert result is False

    def test_no_token_does_not_save_session(self, handlers, sio):
        _run(handlers["connect"]("sid2", {}))
        assert "sid2" not in sio.sessions

    def test_invalid_token_rejects(self, handlers, sio):
        result = _run(handlers["connect"]("sid3", _make_environ("garbage.token.here")))
        assert result is False

    def test_expired_token_rejects(self, handlers, sio):
        import time
        expired = jwt.encode(
            {"user_id": "u", "exp": int(time.time()) - 3600},
            JWT_SECRET, algorithm="HS256",
        )
        result = _run(handlers["connect"]("sid4", _make_environ(expired)))
        assert result is False


# ===================================================================
# Test: disconnect
# ===================================================================

class TestDisconnect:
    def test_disconnect_leaves_user_room(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        assert TEST_USER_ID in sio.rooms["sid1"]

        _run(handlers["disconnect"]("sid1"))
        assert TEST_USER_ID not in sio.rooms.get("sid1", set())

    def test_disconnect_without_prior_connect(self, handlers, sio):
        """Should not crash even if the session is empty."""
        _run(handlers["disconnect"]("unknown_sid"))


# ===================================================================
# Test: join_room
# ===================================================================

class TestJoinRoom:
    def test_join_room_enters_user_room(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        # Clear auto-join to test explicit join
        sio.rooms["sid1"].clear()

        _run(handlers["join_room"]("sid1", {}))
        assert TEST_USER_ID in sio.rooms["sid1"]

    def test_join_room_emits_confirmation(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        _run(handlers["join_room"]("sid1", {}))
        response_emits = [(e, d) for e, d, r in sio.emitted if e == "response"]
        assert len(response_emits) == 1
        assert TEST_USER_ID in response_emits[0][1]["response"]


# ===================================================================
# Test: question
# ===================================================================

class TestQuestion:
    def test_valid_question_emits_completed_response(self, handlers, sio, mock_app):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        _run(handlers["question"]("sid1", {"question": "What is aging?"}))

        response_emits = [(e, d) for e, d, r in sio.emitted if e == "response"]
        assert len(response_emits) >= 1
        last = response_emits[-1]
        assert last[1]["status"] == "completed"
        assert last[1]["response"]["text"] == "Test answer"

    def test_valid_question_calls_agent(self, handlers, sio, mock_app):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))

        _run(handlers["question"]("sid1", {"question": "What is aging?"}))

        mock_app.state.ai_assistant.agent.assert_called_once()
        args = mock_app.state.ai_assistant.agent.call_args
        assert args[0][0] == "What is aging?"
        assert args[0][1] == TEST_USER_ID

    def test_missing_question_field_emits_error(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        _run(handlers["question"]("sid1", {}))

        error_emits = [(e, d) for e, d, r in sio.emitted if e == "error"]
        assert len(error_emits) == 1
        assert "Invalid" in error_emits[0][1]["error"]

    def test_empty_question_emits_error(self, handlers, sio):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        _run(handlers["question"]("sid1", {"question": ""}))

        error_emits = [(e, d) for e, d, r in sio.emitted if e == "error"]
        assert len(error_emits) == 1

    def test_agent_exception_emits_error_event(self, handlers, sio, mock_app):
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        mock_app.state.ai_assistant.agent.side_effect = RuntimeError("LLM is down")

        _run(handlers["question"]("sid1", {"question": "test"}))

        error_emits = [(e, d) for e, d, r in sio.emitted if e == "error"]
        assert len(error_emits) == 1
        assert "LLM is down" in error_emits[0][1]["error"]

    def test_response_emitted_to_user_room_not_sid(self, handlers, sio, mock_app):
        """The response should target the user's room, not the raw sid."""
        _run(handlers["connect"]("sid1", _make_environ(TEST_TOKEN)))
        sio.emitted.clear()

        _run(handlers["question"]("sid1", {"question": "test"}))

        response_emits = [(e, d, r) for e, d, r in sio.emitted if e == "response"]
        assert len(response_emits) >= 1
        # Room should be the user_id, not "sid1"
        assert response_emits[-1][2] == TEST_USER_ID
