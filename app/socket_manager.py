import logging
from flask import Flask
from flask_socketio import SocketIO
import redis
import json
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize SocketIO with Redis integration for scaling
socketio = None
redis_client = None

def init_socketio(app):
    """Initialize the SocketIO instance with Redis message queue for scaling."""
    global socketio, redis_client
    
    # Check if Redis is configured
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_enabled = os.getenv('ENABLE_REDIS', 'true').lower() == 'true'
    
    logger.info(f"Initializing SocketIO with Redis: {redis_enabled}")
    
    try:
        if redis_enabled:
            # Setup Redis client
            redis_client = redis.from_url(redis_url)
            socketio = SocketIO(
                app, 
                message_queue=redis_url, 
                cors_allowed_origins="*",
                async_mode='eventlet'  # Using eventlet for better performance
            )
            logger.info(f"SocketIO initialized with Redis message queue at {redis_url}")
        else:
            socketio = SocketIO(app, cors_allowed_origins="*")
            logger.info("SocketIO initialized without Redis message queue")
            
        # Register socket event handlers
        register_socket_events(socketio)
        
        return socketio
    except Exception as e:
        logger.error(f"Error initializing SocketIO: {e}")
        # Fallback to regular SocketIO without Redis
        socketio = SocketIO(app, cors_allowed_origins="*")
        logger.info("Fallback: SocketIO initialized without Redis message queue")
        register_socket_events(socketio)
        return socketio

def register_socket_events(socketio_instance):
    """Register event handlers for socket connections."""
    
    @socketio_instance.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
    
    @socketio_instance.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio_instance.on('join_room')
    def handle_join_room(data):
        """Handle client joining a specific room (usually user-specific)"""
        user_id = data.get('user_id')
        if user_id:
            from flask_socketio import join_room
            join_room(user_id)
            logger.info(f"User {user_id} joined room")
            emit('status', {'message': f'Joined room for user {user_id}'}, room=user_id)

def emit_processing_update(user_id, event_type, data):
    """
    Emit processing updates to specific user room
    
    Args:
        user_id (str): The user ID to send the message to (room)
        event_type (str): The type of event ('json_format', 'analysis_started', etc)
        data (dict): The data to send with the event
    """
    global socketio
    
    if not socketio:
        logger.warning("SocketIO not initialized, cannot emit event")
        return
    
    try:
        # Add timestamp to the data
        from datetime import datetime
        data['timestamp'] = datetime.now().isoformat()
        data['event'] = event_type
        
        logger.info(f"Emitting '{event_type}' event to user {user_id}: {data}")
        socketio.emit('processing_update', data, room=user_id)
    except Exception as e:
        logger.error(f"Error emitting socket event: {e}")

# Convenience functions for different types of updates
def emit_json_formatting(user_id, status, details=None):
    """Emit update about JSON formatting process"""
    data = {
        'stage': 'json_formatting',
        'status': status,  # 'started', 'in_progress', 'completed'
    }
    if details:
        data['details'] = details
    emit_processing_update(user_id, 'json_format', data)

def emit_analysis_update(user_id, status, progress=None, details=None):
    """Emit update about analysis process"""
    data = {
        'stage': 'analysis',
        'status': status,  # 'started', 'in_progress', 'completed'
    }
    if progress is not None:
        data['progress'] = progress  # 0-100 percentage
    if details:
        data['details'] = details
    emit_processing_update(user_id, 'analysis', data)

def emit_rag_update(user_id, status, details=None):
    """Emit update about RAG retrieval process"""
    data = {
        'stage': 'rag_retrieval',
        'status': status,  # 'started', 'in_progress', 'completed'
    }
    if details:
        data['details'] = details
    emit_processing_update(user_id, 'rag', data)

def emit_hypothesis_update(user_id, status, details=None):
    """Emit update about hypothesis generation process"""
    data = {
        'stage': 'hypothesis',
        'status': status,  # 'started', 'in_progress', 'completed'
    }
    if details:
        data['details'] = details
    emit_processing_update(user_id, 'hypothesis', data)

def emit_error(user_id, error_type, message, details=None):
    """Emit error notification"""
    data = {
        'error_type': error_type,
        'message': message,
    }
    if details:
        data['details'] = details
    emit_processing_update(user_id, 'error', data)