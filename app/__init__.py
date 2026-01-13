import os
import json
import yaml
import logging
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .routes import main_bp
from app.main import AiAssistance
from app.rag.rag import RAG
from app.socket_manager import init_socketio
from app.storage.qdrant import Qdrant
from app.storage.mongo_storage import MongoManager
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import (
    get_llm_model,
    sentence_transformer_embedding_model,
    gemini_embedding_model,
    openai_embedding_model,
    get_embedding_vector_size,
)
from app.storage.qdrant import Qdrant
from app.main import AiAssistance
from app.rag.rag import RAG
from .routes import main_bp
import os
import yaml
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Loads the application configuration from a YAML file."""
    logger.info("Loading environment variables from .env file")
    load_dotenv()  # Load environment variables from .env

    config_path = "./config/config.yaml"
    logger.info(f"Reading configuration from {config_path}")

    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Configuration loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise


def initialize_database():
    """Initialize MongoDB database - collections are created automatically"""
    try:
        print(
            "MongoDB collections are created automatically when first document is inserted"
        )
        print("Database initialization completed!")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        traceback.print_exc()


def create_app():
    """Creates and configures the Flask application."""
    logger.info("Creating Flask app")
    app = Flask(__name__)
    CORS(app)

    config = load_config()
    app.config.update(config)
    logger.info("App config updated with loaded configuration")

    # Code execution defaults from environment with safe fallbacks
    app.config["CODE_EXEC_TIMEOUT"] = int(os.getenv("CODE_EXEC_TIMEOUT", "60"))
    app.config["CODE_EXEC_MAX_MEM_MB"] = int(os.getenv("CODE_EXEC_MAX_MEM_MB", "1024"))
    app.config["CODE_EXEC_ALLOW_NETWORK"] = os.getenv("CODE_EXEC_ALLOW_NETWORK", "false").lower() == "true"
    app.config["ARTIFACT_TTL_MINUTES"] = int(os.getenv("ARTIFACT_TTL_MINUTES", "120"))

    # Apply rate limiting to the entire app (200 requests per minute)
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute"],
    )
    logger.info("FlaskLimiter initialized")

    # Initialize SchemaHandler
    schema_handler = SchemaHandler(
        schema_config_path="./config/schema_config.yaml",
        biocypher_config_path="./config/biocypher_config.yaml",
        enhanced_schema_path="./config/enhanced_schema.txt",
    )
    logger.info("SchemaHandler initialized")

    # Initialize Basic LLM model
    basic_llm_provider = os.getenv("BASIC_LLM_PROVIDER")
    basic_llm_version = os.getenv("BASIC_LLM_VERSION")
    logger.info(
        f"Initializing BASIC LLM model with provider={basic_llm_provider} and version={basic_llm_version}"
    )
    basic_llm = get_llm_model(
        model_provider=basic_llm_provider, model_version=basic_llm_version
    )
    logger.info("BASIC LLM model initialized successfully")

    # Initialize Advanced LLM model
    advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER")
    advanced_llm_version = os.getenv("ADVANCED_LLM_VERSION")
    logger.info(
        f"Initializing ADVANCED LLM model with provider={advanced_llm_provider} and version={advanced_llm_version}"
    )
    advanced_llm = get_llm_model(
        model_provider=advanced_llm_provider, model_version=advanced_llm_version
    )
    logger.info("ADVANCED LLM model initialized successfully")

    # Initialize Embedding Model
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    logger.info(f"Initializing embedding model with provider: {embedding_provider}")

    if embedding_provider == "openai":
        embedding_model = openai_embedding_model
    elif embedding_provider == "gemini":
        embedding_model = gemini_embedding_model
    else:
        embedding_model = sentence_transformer_embedding_model

    vector_size = get_embedding_vector_size(embedding_model)
    qdrant_client = Qdrant(embedding_model=embedding_model, vector_size=vector_size)
    app.config["qdrant_client"] = qdrant_client
    app.config["embedding_model"] = embedding_model
    app.config["embedding_vector_size"] = vector_size

    # Check for SITE_INFORMATION collection and upload sample data if needed
    try:
        # Check if collection exists and has data
        collection_exists = False
        is_empty = True
        
        try:
            collection_info = qdrant_client.client.get_collection("SITE_INFORMATION")
            collection_exists = True
            if collection_info.points_count > 0:
                is_empty = False
                logger.info(
                    f"SITE_INFORMATION collection exists with {collection_info.points_count} points. Skipping population."
                )
            else:
                logger.info("SITE_INFORMATION collection exists but is empty.")
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info("SITE_INFORMATION collection not found.")
            else:
                logger.error(f"Error checking collection status: {e}")
                raise e

        # Trigger ingestion if collection is missing OR empty
        # if not collection_exists or is_empty:
        #     logger.info("Starting population of SITE_INFORMATION collection...")
            
        #     with open("sample_data.json") as data:
        #         sample_site_data = json.load(data)

        #     # Initialize a RAG instance to handle the data upload
        #     rag = RAG(
        #         advanced_llm,
        #         qdrant_client=qdrant_client,
        #     )
        #     # Upload the data to the specified collection
        #     rag.save_doc_to_rag(
        #         data=sample_site_data,
        #         collection_name="SITE_INFORMATION",
        #         is_content=False,
        #     )
        #     logger.info("Successfully populated SITE_INFORMATION collection.")

        # --- CONTEXT SHARING TEST DATA INGESTION ---
        # Force re-ingestion for testing purposes
        logger.info("Starting population of SITE_INFORMATION collection with TEST DATA...")
        
        # Delete existing collection to ensure clean test state
        try:
            qdrant_client.client.delete_collection("SITE_INFORMATION")
            logger.info("Deleted existing SITE_INFORMATION collection for testing.")
        except Exception as e:
            logger.warning(f"Could not delete collection (might not exist): {e}")

        with open("test_context_sharing_sample_data.json") as data:
            test_data = json.load(data)
            
        rag = RAG(
            advanced_llm,
            qdrant_client=qdrant_client,
        )
        
        rag.save_doc_to_rag(
            data=test_data,
            collection_name="SITE_INFORMATION",
            is_content=False,
        )
        logger.info("Successfully populated SITE_INFORMATION collection with TEST DATA.")
        # -------------------------------------------

    except Exception as e:
        logger.error(
            f"An error occurred during the application setup for SITE_INFORMATION: {e}",
            exc_info=True,
        )

    except Exception as e:
        logger.error(
            f"An error occurred during the application setup for SITE_INFORMATION: {e}",
            exc_info=True,
        )

    # Initialize MongoDB manager
    mongo_db_manager = MongoManager()
    app.config["mongo_db_manager"] = mongo_db_manager
    logger.info("MongoDB manager initialized and stored in app config")

    # Initialize AiAssistance with shared qdrant_client and embedding_model
    ai_assistant = AiAssistance(
        advanced_llm,
        basic_llm,
        schema_handler,
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
    )
    logger.info("AiAssistance initialized")

    # Store objects in app config
    app.config["basic_llm"] = basic_llm
    app.config["advanced_llm"] = advanced_llm
    app.config["schema_handler"] = schema_handler
    app.config["ai_assistant"] = ai_assistant
    logger.info("App config populated with models and assistants")

    # Initialize SocketIO
    socketio = init_socketio(app)
    app.config["socketio"] = socketio
    logger.info("SocketIO initialized and stored in app config")

    try:
        initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    # Register routes
    app.register_blueprint(main_bp)
    logger.info('Blueprint "main_bp" registered')

    logger.info("Flask app created successfully")
    return app, socketio
