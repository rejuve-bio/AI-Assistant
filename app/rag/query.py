import numpy as np
from .qdrant import OPEN_AI_VECTOR_SIZE, Qdrant
from app.prompts.rag_prompts import SYSTEM_PROMPT, RETRIEVE_PROMPT
from app.llm_handle.llm_models import LLMInterface, embedding_model
import traceback
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
USER_COLLECTION = os.getenv("USER_COLLECTION")
class RAG:

    def __init__(self,llm:LLMInterface) -> None:
        self.client = Qdrant()
        self.llm = llm

    def query(self, query_str,user_id):
        try:
            logger.info("Query embedding started")

            if isinstance(query_str, str):
                query_str = [query_str]

            query = {}
            embeddings = embedding_model(query_str)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate dense embeddings")
                return None
            
            embed = np.array(embeddings)
            query["dense"] = embed.reshape(-1, OPEN_AI_VECTOR_SIZE).tolist()[0]

            query_result = self.client.retrieve_data(VECTOR_COLLECTION, query)
            user_query = self.client.retrieve_user_data(USER_COLLECTION, query,user_id)

            result = query_result,user_query
            return result    
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return None

    def result(self, query_str, user_id):
        try:
            query_result = self.query(query_str,user_id)
            if query_result is None:
                logger.error("No query result to process")
                return None
            prompt = RETRIEVE_PROMPT.format(query=query_str,retrieved_content=query_result)
            result = self.llm.generate(prompt)
            return result

        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None


