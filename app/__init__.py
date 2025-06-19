import logging
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from flask_cors import CORS
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import get_llm_model
from app.storage.qdrant import Qdrant
from app.main import AiAssistance
from app.rag.rag import RAG
from .routes import main_bp
import os
import yaml
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Loads the application configuration from a YAML file."""
    logger.info('Loading environment variables from .env file')
    load_dotenv()  # Load environment variables from .env

    config_path = './config/config.yaml'
    logger.info(f'Reading configuration from {config_path}')
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            logger.info('Configuration loaded successfully')
            return config
    except Exception as e:
        logger.error(f'Error loading config file: {e}')
        raise

from app.storage.sql_redis_storage import create_tables, db_manager
import os

def initialize_database():
    """Initialize the SQLite database with tables only - no sample data"""
    try:
        print("Initializing SQLite database...")
        
        data_dir = os.getenv('DATABASE_DIR', './data')
        os.makedirs(data_dir, exist_ok=True)
        
        create_tables()
        print("Database tables created successfully!")
               
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()

def create_app():
    """Creates and configures the Flask application."""
    logger.info('Creating Flask app')
    app = Flask(__name__)
    CORS(app)
    
    config = load_config()
    app.config.update(config)
    logger.info('App config updated with loaded configuration')

    # Apply rate limiting to the entire app (200 requests per minute)
    limiter = Limiter(get_remote_address, 
                      app=app,
                      default_limits=["200 per minute"],
                      )
    logger.info('FlaskLimiter initialized')

    # Initialize SchemaHandler
    schema_handler = SchemaHandler(
        schema_config_path='./config/schema_config.yaml', 
        biocypher_config_path='./config/biocypher_config.yaml',
        enhanced_schema_path='./config/enhanced_schema.txt'
    )
    logger.info('SchemaHandler initialized')

    # Initialize Basic LLM model
    basic_llm_provider = os.getenv('BASIC_LLM_PROVIDER')
    basic_llm_version = os.getenv('BASIC_LLM_VERSION')
    logger.info(f'Initializing BASIC LLM model with provider={basic_llm_provider} and version={basic_llm_version}')
    basic_llm = get_llm_model(model_provider=basic_llm_provider, model_version=basic_llm_version)
    logger.info('BASIC LLM model initialized successfully')

    # Initialize Advanced LLM model
    advanced_llm_provider = os.getenv('ADVANCED_LLM_PROVIDER')
    advanced_llm_version = os.getenv('ADVANCED_LLM_VERSION')
    logger.info(f'Initializing ADVANCED LLM model with provider={advanced_llm_provider} and version={advanced_llm_version}')
    advanced_llm = get_llm_model(model_provider=advanced_llm_provider, model_version=advanced_llm_version)
    logger.info('ADVANCED LLM model initialized successfully')

    # Initialize AiAssistance
    ai_assistant = AiAssistance(advanced_llm, basic_llm, schema_handler)
    logger.info('AiAssistance initialized')

    # Store objects in app config
    app.config['basic_llm'] = basic_llm
    app.config['advanced_llm'] = advanced_llm
    app.config['schema_handler'] = schema_handler
    app.config['ai_assistant'] = ai_assistant
    logger.info('App config populated with models and assistants')

    # intialize qdrant connection
    # uploading data first time
    try:
        client = Qdrant()

        collections = client.client.get_collections()
        if collections and collections.collections:
            logger.info("collections on the qdrant database already exist skipping population data")
        else:
            logger.info('uploading sample web data to qdrant db')
            with open('sample_data.json') as data:
                data = json.load(data)
            rag = RAG(client,advanced_llm)
            rag.save_doc_to_rag(data=data)
    except:
        import traceback
        traceback.print_exc()
        logger.warning("Qdrant Connection Failed!!! If you are running locally Please connect qdrant database by running docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant")

    try:
        initialize_database()
        logger.info('Database initialized successfully')
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    # Register routes
    app.register_blueprint(main_bp)
    logger.info('Blueprint "main_bp" registered')

    logger.info('Flask app created successfully')
    return app

from app import routes