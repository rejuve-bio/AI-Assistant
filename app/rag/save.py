import numpy as np
import pandas as pd
import logging
from app.llm_handle.llm_models import openai_embedding_model
from .qdrant import Qdrant
import traceback
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
USER_COLLECTION = os.getenv("USER_COLLECTION")
class Save:

    def __init__(self) -> None:
        self.qdrant = Qdrant()

    def chunking_data(df):
        '''
        here chunking datas and dynamically accepts differnt types of documents will be implemented
        '''
        pass
    
    def get_contents_embed(self, df):
        try:

            # Generate dense embeddings
            embeddings = openai_embedding_model(df['content'].tolist())
            embed = np.array(embeddings)
            embedding = embed.reshape(-1, 1536)    
            df['dense'] = embedding.tolist()
            return df

        except Exception as e:

            logger.error(f"Error generating dense embeddings: {e}")           
            traceback.print_exc()

    def save_to_collection(self, collection_name, df):
        try:
            df = self.get_contents_embed(df)
            if df is not None:
                self.qdrant.upsert_data(collection_name, df)
            else:
                logger.error(f"Embedding generation failed, data not upserted to collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error saving to collection {collection_name}: {e}")
            traceback.print_exc()
