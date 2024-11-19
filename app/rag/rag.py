from app.prompts.rag_prompts import SYSTEM_PROMPT, RETRIEVE_PROMPT
from app.llm_handle.llm_models import LLMInterface, openai_embedding_model
import traceback
import os
import numpy as np
import pandas as pd
import logging
from app.llm_handle.llm_models import openai_embedding_model
from ..storage.qdrant import Qdrant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")
USER_COLLECTION = os.getenv("USER_COLLECTION")

class RAG:

    def __init__(self, llm: LLMInterface) -> None:
        """
        Initializes the RAG (Retrieval Augmented Generation) class.
        Sets up the Qdrant client and LLM interface for query handling.

        :param llm: An instance of the LLMInterface for generating responses.
        """
        self.client = Qdrant()
        self.llm = llm
        logger.info("RAG initialized with LLM model and Qdrant client.")

    def chunking_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is a placeholder for data chunking implementation, 
        which will handle dynamic chunking of various types of documents.

        :param df: A DataFrame containing the data to be chunked.
        :return: DataFrame with chunked data (Not yet implemented).
        """
        logger.info("Chunking data started. (Implementation pending)")
        pass
    
    def get_contents_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates dense embeddings for the content column of the provided DataFrame.

        :param df: DataFrame containing the documents to embed.
        :return: DataFrame with the added 'dense' column containing embeddings.
        """
        try:
            logger.info("Generating embeddings for content column.")
            embeddings = openai_embedding_model(df['content'].tolist())
            embed = np.array(embeddings)
            embedding = embed.reshape(-1, 1536)  # Reshaping for Qdrant's format
            df['dense'] = embedding.tolist()  # Add embeddings to DataFrame
            logger.info("Embeddings generated successfully.")
            return df
        except Exception as e:
            logger.error(f"Error generating dense embeddings: {e}")
            traceback.print_exc()

    def save_to_collection(self, collection_name: str, df: pd.DataFrame):
        """
        Saves the DataFrame with embeddings to the specified Qdrant collection.

        :param collection_name: The name of the collection to save data to.
        :param df: DataFrame containing the data to be saved, including embeddings.
        """
        try:
            logger.info(f"Saving data to collection {collection_name}.")
            df = self.get_contents_embed(df)
            if df is not None:
                self.client.upsert_data(collection_name, df)
                logger.info(f"Data successfully upserted to collection {collection_name}.")
            else:
                logger.error(f"Embedding generation failed. Data not upserted to collection {collection_name}.")
        except Exception as e:
            logger.error(f"Error saving to collection {collection_name}: {e}")
            traceback.print_exc()

    def query(self, query_str: str, user_id: str):
        """
        Processes a query string by generating its embeddings and retrieving related content 
        from the Qdrant vector collection.

        :param query_str: The query string to process.
        :param user_id: The ID of the user making the query.
        :return: Retrieved content from the collection or None if no content is found.
        """
        try:
            logger.info("Query embedding started.")
            if isinstance(query_str, str):
                query_str = [query_str]  # Ensure it's a list if a single string is provided

            query = {}
            embeddings = openai_embedding_model(query_str)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate dense embeddings for the query.")
                return None

            embed = np.array(embeddings)
            query["dense"] = embed.reshape(-1, 1536).tolist()[0]

            result = self.client.retrieve_data(VECTOR_COLLECTION, query)
            if result:
                logger.info("Query retrieved successfully.")
            else:
                logger.warning("No results found for the query.")
            return result    
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return None

    def result(self, query_str: str, user_id: str):
        """
        Retrieves the result for a query by calling the query method 
        and generating a response based on the retrieved content.

        :param query_str: The query string to process.
        :param user_id: The ID of the user making the request.
        :return: The result from the LLM generated based on the query and retrieved content.
        """
        try:
            logger.info("Generating result for the query.")
            query_result = self.query(query_str, user_id)
            if query_result is None:
                logger.error("No query result to process.")
                return None

            prompt = RETRIEVE_PROMPT.format(query=query_str, retrieved_content=query_result)
            result = self.llm.generate(prompt)
            logger.info("Result generated successfully.")
            return result
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None
