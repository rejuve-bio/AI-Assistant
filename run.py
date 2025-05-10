import eventlet  # Required for socketio with eventlet mode
# Set eventlet as the async backend
eventlet.monkey_patch()

from dotenv import load_dotenv
from app import create_app
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
port = int(os.getenv('FLASK_PORT', 5003))
debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'

app, socketio = create_app()

if __name__ == '__main__':
    logger.info(f"Starting application on port {port} with debug={debug}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, use_reloader=debug)