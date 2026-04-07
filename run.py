import eventlet
eventlet.monkey_patch()


import os
import logging
from logging.handlers import TimedRotatingFileHandler
from app import create_app
from dotenv import load_dotenv

# Creating log directory 
log_dir = "/AI-Assistant/logfiles"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "Assistant.log")

handler = TimedRotatingFileHandler(
    filename=log_file,
    when="midnight",
    interval=1,
    backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

load_dotenv()
port = int(os.getenv('FLASK_PORT', 5003))
app, socketio = create_app()

if __name__ == '__main__':
    logger.info(f"Starting application on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False)