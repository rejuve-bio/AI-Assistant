import asyncio
import inspect
import logging
import os

import socketio
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from app.lib.auth import decode_socket_token
from app.storage.redis import redis_manager

logger = logging.getLogger(__name__)

load_dotenv()

# Module-level handles so emit_to_user (called from deep inside synchronous
# business logic all over the codebase) can reach the running server.
sio = None
_main_loop = None


def set_event_loop(loop):
    """Called once from the FastAPI lifespan startup so emit_to_user can
    schedule emits onto the loop Socket.IO is actually running on, safely,
    from worker threads that don't have a loop of their own."""
    global _main_loop
    _main_loop = loop


def create_socket_app(fastapi_app):
    """Builds the Socket.IO AsyncServer and wraps fastapi_app in its ASGI app.

    Returns the combined ASGI app that Uvicorn should actually serve --
    requests under /socket.io/ go to Socket.IO, everything else falls
    through to fastapi_app.
    """
    global sio

    client_manager = None
    try:
        if redis_manager.is_available:
            redis_url = redis_manager.get_socketio_redis_url()
            client_manager = socketio.AsyncRedisManager(redis_url)
            logger.info(f"Socket.IO using Redis message queue at {redis_url}")
        else:
            logger.info("Socket.IO initialized without Redis message queue")
    except Exception as e:
        logger.error(f"Error setting up Socket.IO Redis message queue, falling back: {e}")
        client_manager = None

    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*", client_manager=client_manager)
    register_socket_events(sio, fastapi_app)

    return socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


def register_socket_events(sio_instance, fastapi_app):
    """Register event handlers for socket connections."""

    @sio_instance.event
    async def connect(sid, environ, auth=None):
        try:
            auth_header = environ.get("HTTP_AUTHORIZATION")
            auth_context = decode_socket_token(auth_header)
        except Exception as e:
            logger.error(f"Socket auth error: {e}")
            return False  # rejects the connection

        await sio_instance.save_session(sid, {"user_id": auth_context.user_id, "token": auth_context.token})
        logger.info("Client connected")
        await sio_instance.emit("response", {"message": "Connected successfully"}, room=sid)

    @sio_instance.event
    async def disconnect(sid):
        logger.info("Client disconnected")

    @sio_instance.event
    async def join_room(sid, data):
        """Handle client joining a specific room (usually user-specific)."""
        session = await sio_instance.get_session(sid)
        user_id = session.get("user_id")
        if user_id:
            await sio_instance.enter_room(sid, user_id)
            logger.info(f"User {user_id} joined room")
            await sio_instance.emit("response", {"response": f"user {user_id} joined room"}, room=user_id)

    @sio_instance.event
    async def question(sid, data):
        """Handle incoming questions from clients."""
        session = await sio_instance.get_session(sid)
        user_id = session.get("user_id")
        token = session.get("token")
        query = data.get("question")

        if not (user_id and query):
            logger.error("Invalid question data received")
            await sio_instance.emit("error", {"error": "Invalid question data"}, room=sid)
            return

        logger.info(f"Received question from {user_id}: {query}")

        try:
            ai_assistant = fastapi_app.state.ai_assistant

            if inspect.iscoroutinefunction(getattr(ai_assistant, "assistant", None)):
                responses = await ai_assistant.assistant(query=query, user_id=user_id, token=token)
            else:
                responses = await run_in_threadpool(ai_assistant.agent, query, user_id, token)

            logger.info(f"Responses generated for user {user_id}")

            # Emit the final structured response back to the client.
            # The agent() path already fires intermediate "update" events
            # via emit_to_user(); this final emit carries the complete answer
            # and signals the client that processing is done.
            await sio_instance.emit(
                "response",
                {"status": "completed", "response": responses},
                room=user_id,
            )
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            await sio_instance.emit("error", {"error": str(e)}, room=user_id)


def get_socketio():
    return sio


def emit_to_user(user, message, status="update"):
    """Helper method to emit updates to a user, safe to call from any thread.

    Called synchronously from deep inside business logic (main.py,
    hypothesis.py, annotated_graph.py) that isn't running on the event
    loop, so this schedules the emit onto the loop instead of awaiting it
    directly -- fire-and-forget, matching the previous behavior.
    """
    try:
        if sio is None or _main_loop is None:
            logger.warning("Socket.IO not ready yet, dropping emit")
            return
        asyncio.run_coroutine_threadsafe(
            sio.emit("update", {"status": status, "response": message}, room=user),
            _main_loop,
        )
    except Exception as e:
        logger.error(f"Error emitting: {e}")
