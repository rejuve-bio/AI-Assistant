from app.services.schema_handler import SchemaHandler
from flask import Flask
from dotenv import load_dotenv
from .routes import main_bp
import os
import yaml

schema_handler = SchemaHandler('./config/schema_config.yaml', './config/biocypher_config.yaml')
def load_config():
    load_dotenv()
    
    config_path = './config/config.yaml'
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)
    
def create_app():
    app = Flask(__name__)
    
    config = load_config()
    app.config.update(config)
    
    app.register_blueprint(main_bp)
    
    return app
    