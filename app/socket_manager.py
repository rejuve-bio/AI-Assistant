
from flask_socketio import SocketIO, emit, send, disconnect, join_room
from flask import session
import redis
import json
import os
from dotenv import load_dotenv
import logging
import asyncio
from app.lib.auth import socket_token_required
from app.storage.redis import redis_manager

logger = logging.getLogger(__name__)

load_dotenv()

# Initialize SocketIO with Redis integration for scaling
socketio = None
redis_client = None
user = None
def init_socketio(app):
    """Initialize the SocketIO instance with Redis message queue for scaling."""
    global socketio, redis_client
    
    # Check if Redis is configured
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_enabled = os.getenv('ENABLE_REDIS', 'true').lower() == 'true'
    
    logger.info(f"Initializing SocketIO with Redis: {redis_enabled}")
    
    try:
        if redis_manager.is_available:
            redis_url = redis_manager.get_socketio_redis_url()
            socketio = SocketIO(
                app, 
                message_queue=redis_url, 
                cors_allowed_origins="*",
                async_mode='eventlet'
            )
            logger.info(f"SocketIO initialized with Redis message queue at {redis_url}")
        else:
            socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
            logger.info("SocketIO initialized without Redis message queue")
            
        # Register socket event handlers
        register_socket_events(socketio)
        return socketio
    except Exception as e:
        logger.error(f"Error initializing SocketIO: {e}")
        # Fallback to regular SocketIO without Redis
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
        logger.info("Fallback: SocketIO initialized without Redis message queue")
        register_socket_events(socketio)
        return socketio

def register_socket_events(socketio_instance):
    """Register event handlers for socket connections."""
    
    @socketio_instance.on('connect')
    @socket_token_required
    def handle_connect(current_user_id,token,args):
        logger.info(f"Client connected")
        session['user_id']= current_user_id
        session['token'] = token 
        emit('response', {'message': 'Connected successfully'})
    
    @socketio_instance.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected")
    
    @socketio_instance.on('join_room')
    def handle_join_room(data):
        """Handle client joining a specific room (usually user-specific)"""
        user_id = session['user_id']
        if user_id:
            join_room(user_id)
            logger.info(f"User {user_id} joined room")
            if redis_client:
                cache_key = f'{user_id}_room'
                redis_client.set(cache_key, json.dumps({'user_id': user_id}))
            emit('response', {'response': f'user {user_id} joined room'}, room=user_id)


    def _handle_question_processing(data, sid):
        query = data.get('question')
        user_id = session['user_id']
        token = session['token']

        if not (user_id and query):
            logger.error("Invalid question data received")
            emit('error', {'error': 'Invalid question data'})
            return

        logger.info(f"Received question from {user_id}: {query}")

        try:
            from flask import current_app
            ai_assistant = current_app.config['ai_assistant']

            ai_assistant.socketio = socketio_instance

            try:
                import inspect
                if hasattr(ai_assistant, 'assistant') and inspect.iscoroutinefunction(ai_assistant.assistant):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        responses = loop.run_until_complete(
                            ai_assistant.assistant(
                                query=query,
                                user_id=user_id,
                                token=token
                            )
                        )
                        loop.close()
                    except Exception as async_e:
                        logger.error(f"Async execution failed: {async_e}")
                        responses = {"text": "Error processing request"}
                else:
                    responses = ai_assistant.agent(query, user_id, token)

                logger.info(f"Responses generated for user {user_id}: {responses}")

            except Exception as e:
                logger.error(f"Error processing question: {e}")
                socketio_instance.emit('error', {'error': str(e)}, room=user_id)

        except Exception as e:
            logger.error(f"Error setting up question processing: {e}")
            emit('error', {'error': 'Failed to process question'}, room=user_id)

    @socketio_instance.on('question')
    def handle_question(data):
        """Handle incoming questions from clients."""
        _handle_question_processing(data, None)
# Make socketio instance available globally
def get_socketio():
    return socketio

def emit_to_user(user,message,status="update"):
    """Helper method to emit updates to user"""
    try:
        socketio_instance = get_socketio()
        if socketio_instance:
            socketio_instance.emit('update', {'status': status, 'response': message}, room=user)
    except Exception as e:
        logger.error(f"Error emitting: {e}")
        