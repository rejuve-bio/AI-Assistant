import logging
import os

from langgraph.checkpoint.mongodb import MongoDBSaver

logger = logging.getLogger(__name__)


def create_checkpointer(mongo_client):
    """Builds the LangGraph checkpointer, using MongoDB.
    """
    db_name = os.getenv("MONGO_CHECKPOINT_DATABASE", "ai_assistant_checkpoints")
    checkpointer = MongoDBSaver(mongo_client, db_name=db_name)
    logger.info(f"LangGraph checkpointer initialized (Mongo db={db_name})")
    return checkpointer
