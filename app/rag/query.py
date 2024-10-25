import numpy as np
from .qdrant import Qdrant
from app.llm_handle.llm_models import chat_completion, openai_embedding_model
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class RAG:

    def __init__(self) -> None:
        self.client = Qdrant()

    def query(self, query_str, collection_name):
        try:
            logger.info("Query embedding started")

            if isinstance(query_str, str):
                query_str = [query_str]

            query = {}
            embeddings = openai_embedding_model(query_str)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate dense embeddings")
                return None
            
            embed = np.array(embeddings)
            query["dense"] = embed.reshape(-1, 1536).tolist()[0]

            result = self.client.retrieve_data(collection_name, query)
            return result    
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return None

    def result(self, query_str, collection_name):
        try:
            query_result = self.query(query_str, collection_name)
            print(query_result)
            if query_result is None:
                logger.error("No query result to process")
                return None

            result = chat_completion(query_str, query_result)
            return result

        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None


