# app/storage/redis_manager.py
import redis
import os
import logging
from typing import Optional
from dotenv import load_dotenv
import uuid

load_dotenv()

logger = logging.getLogger(__name__)


class RedisManager:
    """Centralized Redis connection manager with singleton pattern."""

    _instance = None
    _redis_client = None
    _redis_binary_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._redis_client is None:
            self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection based on environment variables."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_enabled = os.getenv("ENABLE_REDIS", "true").lower() == "true"

        if not redis_enabled:
            logger.info("Redis is disabled in configuration")
            self._redis_client = None
            self._redis_binary_client = None
            return

        try:
            # Redis client for text data (with decode_responses=True)
            self._redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Separate Redis client for binary data (without decode_responses=True)
            self._redis_binary_client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            self._redis_client.ping()
            self._redis_binary_client.ping()
            logger.info(f"Redis connection established: {redis_url}")

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis_client = None
            self._redis_binary_client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}")
            self._redis_client = None
            self._redis_binary_client = None

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get Redis client instance for text data."""
        return self._redis_client

    @property
    def binary_client(self) -> Optional[redis.Redis]:
        """Get Redis client instance for binary data."""
        return self._redis_binary_client

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception:
            return False

    def get_socketio_redis_url(self) -> Optional[str]:
        """Get Redis URL for SocketIO message queue."""
        if self.is_available:
            return os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return None

    def reconnect(self):
        """Attempt to reconnect to Redis."""
        logger.info("Attempting to reconnect to Redis...")
        self._redis_client = None
        self._redis_binary_client = None
        self._initialize_redis()

    def create_graph(self, graph_id=None, graph_summary=None, context=None):
        """Create a new graph that expires in 24 hours."""
        if not self.is_available:
            logger.warning("Redis not available, cannot create graph")
            return {"graph_id": graph_id or str(uuid.uuid4())}

        graph_id = graph_id or str(uuid.uuid4())
        key = f"graph:{graph_id}"

        data = {
            "graph_id": graph_id,
            "graph_summary": graph_summary or "",
            "context": context or "",
        }
        self._redis_client.hset(key, mapping=data)
        self._redis_client.expire(key, 86400)  # 24 hours in seconds
        return {"graph_id": graph_id}

    def get_graph_by_id(self, graph_id):
        """Retrieve a graph by its ID if it has not yet expired."""
        if not self.is_available:
            logger.warning("Redis not available, cannot retrieve graph")
            return None

        key = f"graph:{graph_id}"

        if not self._redis_client.exists(key):
            return None

        data = self._redis_client.hgetall(key)
        if not data:
            return None
        return {"graph_id": graph_id, **data}

    def acquire_thread_lock(self, thread_id: str, ttl_seconds: int = 300) -> bool:

        if not self.is_available:
            logger.warning("Redis not available, thread-lock check skipped (failing open)")
            return True
        try:
            return bool(self._redis_client.set(f"thread_lock:{thread_id}", "1", nx=True, ex=ttl_seconds))
        except Exception as e:
            logger.warning(f"Thread-lock acquire failed for {thread_id}, failing open: {e}")
            return True

    def release_thread_lock(self, thread_id: str) -> None:
        if not self.is_available:
            return
        try:
            self._redis_client.delete(f"thread_lock:{thread_id}")
        except Exception as e:
            logger.warning(f"Thread-lock release failed for {thread_id}: {e}")

    def set_audio_cache(self, key, audio_bytes, expire_seconds=600):
        """Store audio bytes in Redis with a TTL (default 10 minutes)."""
        if not self.is_available or self._redis_binary_client is None:
            logger.warning("Redis not available, cannot cache audio")
            return

        try:
            # Use binary client for audio data
            self._redis_binary_client.set(key, audio_bytes, ex=expire_seconds)
            logger.debug(f"Audio cached with key: {key}")
        except Exception as e:
            logger.error(f"Failed to cache audio: {e}")

    def get_audio_cache(self, key):
        """Retrieve audio bytes from Redis. Returns None if not found."""
        if not self.is_available or self._redis_binary_client is None:
            logger.warning("Redis not available, cannot retrieve cached audio")
            return None

        try:
            # Use binary client for audio data
            audio_data = self._redis_binary_client.get(key)
            return audio_data
        except Exception as e:
            logger.error(f"Failed to retrieve cached audio: {e}")
            return None


# Global instance
redis_manager = RedisManager()
