import os
import logging
from logging.handlers import TimedRotatingFileHandler
from app import create_app
from dotenv import load_dotenv

# Creating log directory
log_dir = os.path.join(os.path.dirname(__file__), "logfiles")
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
port = int(os.getenv('FASTAPI_PORT', 5003))

# app: the plain FastAPI instance (used by tests, anything needing app.state)
# asgi_app: app wrapped with the Socket.IO ASGI app -- this is what Uvicorn
# actually serves ("run:asgi_app"), so requests under /socket.io/ reach
# Socket.IO and everything else falls through to FastAPI's routes.
app, asgi_app = create_app()

if __name__ == '__main__':
    import uvicorn
    logger.info(f"Starting application on port {port}")
    uvicorn.run(asgi_app, host="0.0.0.0", port=port, log_level="info")
