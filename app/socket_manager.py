
from flask_socketio import SocketIO,emit, send,disconnect,join_room
import redis
import json
import os
from dotenv import load_dotenv
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
        logger.info(f"Client connected")
        send('User is connected')
    
    @socketio_instance.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected")
        send('disconnected')
        disconnect()
    
    @socketio_instance.on('join_room')
    def handle_join_room(data):
        """Handle client joining a specific room (usually user-specific)"""
        user_id = data.get('user_id')
        token = data.get('token')
        query = data.get('query')
        if user_id:
            join_room(user_id)
            from app.main import AiAssistance
            ai_assistant = AiAssistance()
            responses = ai_assistant.assistant_response(query,user_id,token)
            socketio.emit('query_response', {'response': responses}, room=user_id)
