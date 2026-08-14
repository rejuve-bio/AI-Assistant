import logging
import os
import json
import traceback

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .logging_config import setup_logging
setup_logging()

from .routes import main_bp
from app.main import AiAssistance
from app.rag.rag import RAG
from app.socket_manager import init_socketio
from app.storage.qdrant import Qdrant
from app.storage.mongo_storage import MongoManager
from app.annotation_graph.schema_handler import SchemaHandler
from app.ingestion.scheduler import start_scheduler
from app.llm_handle.llm_models import (
    get_llm_model,
    sentence_transformer_embedding_model,
    gemini_embedding_model,
    openai_embedding_model,
    local_embedding_model,
    get_embedding_vector_size,
)

logger = logging.getLogger(__name__)


def load_config():
    load_dotenv()
    return {}


def initialize_database():
    """Initialize MongoDB database - collections are created automatically"""
    try:
        logger.info("MongoDB collections are created automatically when first document is inserted")
        logger.info("Database initialization completed!")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)


def _init_embedding_model(embedding):
    if embedding == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found")
        model = openai_embedding_model
    elif embedding == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not found")
        model = gemini_embedding_model
    elif embedding == "local_model":
        if not os.getenv("LOCAL_EMBEDDING_HOST"):
            raise ValueError("LOCAL_EMBEDDING_HOST not set")
        model = local_embedding_model
    else:
        model = sentence_transformer_embedding_model
    return model, get_embedding_vector_size(model)


def _upload_sample_data(qdrant_client, advanced_llm, collection):
    logger.info("collection not found, uploading sample web data to qdrant db")
    with open("sample_data.json") as data:
        sample_site_data = json.load(data)
    rag = RAG(advanced_llm, qdrant_client=qdrant_client)
    rag.save_doc_to_rag(data=sample_site_data, collection_name=collection, is_content=False)
    logger.info("Successfully populated SITE INFORMATION collection.")


def _init_site_collection(qdrant_client, advanced_llm):
    collection = os.getenv("VECTOR_COLLECTION")
    try:
        qdrant_client.client.get_collection(collection_name=collection)
        logger.info("collection already exists, skipping population data")
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            _upload_sample_data(qdrant_client, advanced_llm, collection)
        else:
            logger.error(
                f"An unexpected error occurred when checking for SITE INFORMATION collection: {e}",
                exc_info=True,
            )


def _seed_faq_questions(mongo_db_manager):
    faq_file_path = "faq_sample_data.json"
    if not os.path.exists(faq_file_path):
        logger.warning(f"FAQ sample data file not found at {faq_file_path}")
        return
    with open(faq_file_path, "r", encoding="utf-8") as f:
        initial_faqs = json.load(f)
    mongo_db_manager.seed_faq_questions(initial_faqs)


def create_app():
    """Creates and configures the Flask application."""
    logger.info("Creating Flask app")
    app = Flask(__name__)
    CORS(app)

    config = load_config()
    app.config.update(config)
    logger.info("App config updated with loaded configuration")

    Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute"],
        storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        in_memory_fallback_enabled=True,
    )
    logger.info("FlaskLimiter initialized")

    schema_handler = SchemaHandler(
        schema_config_path="./app/annotation_graph/schema/human/schema_config.yaml",
        biocypher_config_path="./app/annotation_graph/schema/human/biocypher_config.yaml",
        enhanced_schema_path="./app/annotation_graph/schema/human/enhanced_schema.txt",
    )
    logger.info("SchemaHandler (human) initialized")

    fly_schema_handler = SchemaHandler(
        schema_config_path="./app/annotation_graph/schema/fly/dmel_full_schema_config.yaml",
        biocypher_config_path="./app/annotation_graph/schema/fly/fly_biocypher_config.yaml",
        enhanced_schema_path="./app/annotation_graph/schema/fly/fly_enhanced_schema.txt",
    )
    logger.info("SchemaHandler (fly) initialized")

    basic_llm_provider = os.getenv("BASIC_LLM_PROVIDER")
    basic_llm_version = os.getenv("BASIC_LLM_VERSION")
    logger.info(f"Initializing BASIC LLM model with provider={basic_llm_provider} and version={basic_llm_version}")
    basic_llm = get_llm_model(model_provider=basic_llm_provider, model_version=basic_llm_version)
    logger.info("BASIC LLM model initialized successfully")

    advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER")
    advanced_llm_version = os.getenv("ADVANCED_LLM_VERSION")
    logger.info(f"Initializing ADVANCED LLM model with provider={advanced_llm_provider} and version={advanced_llm_version}")
    advanced_llm = get_llm_model(model_provider=advanced_llm_provider, model_version=advanced_llm_version)
    logger.info("ADVANCED LLM model initialized successfully")

    embedding = os.getenv("EMBEDDING_MODEL", "sentence_transformer")
    embedding_model, vector_size = _init_embedding_model(embedding)

    qdrant_client = Qdrant(embedding_model=embedding_model, vector_size=vector_size)
    app.config["qdrant_client"] = qdrant_client
    app.config["embedding_model"] = embedding_model
    app.config["embedding_vector_size"] = vector_size

    try:
        _init_site_collection(qdrant_client, advanced_llm)
    except Exception as e:
        logger.error(f"An error occurred during the application setup for SITE INFORMATION: {e}", exc_info=True)

    mongo_db_manager = MongoManager()
    app.config["mongo_db_manager"] = mongo_db_manager
    logger.info("MongoDB manager initialized and stored in app config")

    try:
        _seed_faq_questions(mongo_db_manager)
    except Exception as e:
        logger.error(f"Error seeding FAQ questions: {e}")

    ai_assistant = AiAssistance(
        advanced_llm,
        basic_llm,
        schema_handler,
        fly_schema_handler=fly_schema_handler,
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        mongo_db_manager=mongo_db_manager,
    )
    logger.info("AiAssistance initialized")

    app.config["basic_llm"] = basic_llm
    app.config["advanced_llm"] = advanced_llm
    app.config["schema_handler"] = schema_handler
    app.config["fly_schema_handler"] = fly_schema_handler
    app.config["ai_assistant"] = ai_assistant
    logger.info("App config populated with models and assistants")

    socketio = init_socketio(app)
    app.config["socketio"] = socketio
    logger.info("SocketIO initialized and stored in app config")

    try:
        initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

    app.register_blueprint(main_bp)
    logger.info('Blueprint "main_bp" registered')

    # Start PubMed background ingestion scheduler
    # We assign these directly to the app object for the scheduler to access
    app.qdrant_client = qdrant_client
    app.mongo_db = mongo_db_manager.db
    
    app.scheduler = start_scheduler(app)

    logger.info("Flask app created successfully")
    return app, socketio
