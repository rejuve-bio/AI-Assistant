from app.prompts.rag_prompts import SYSTEM_PROMPT, RETRIEVE_PROMPT
from app.prompts.pdf_prompt import PDF_SUMMARY_PROMPT
from app.llm_handle.llm_models import LLMInterface, openai_embedding_model
from app.llm_handle.llm_models import openai_embedding_model
from app.memory_layer import MemoryManager
from PyPDF2 import PdfReader
import traceback
import os
import numpy as np
import pandas as pd
import logging
import re
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION","SITE_INFORMATION")
USER_COLLECTION = os.getenv("USER_COLLECTION","CHAT_MEMORY")
USERS_PDF_COLLECTION = os.getenv("PDF_COLLECTION","PDF_COLLECTION")
PDF_LIMIT=2
class RAG:

    def __init__(self, client, llm: LLMInterface) -> None:
        """
        Initializes the RAG (Retrieval Augmented Generation) class.
        Sets up the Qdrant client and LLM interface for query handling.

        :param llm: An instance of the LLMInterface for generating responses.
        """
        self.client = client
        self.llm = llm
        if self.llm.__class__.__name__ == 'GeminiModel':
            self.max_token=2000
        elif self.llm.__class__.__name__ == 'OpenAIModel':
            self.max_token=8000
        logger.info("RAG initialized with LLM model and Qdrant client.")

        self.user_pdf_file = "user_pdf.json"
        if os.path.exists(self.user_pdf_file):
            with open(self.user_pdf_file, "r") as f:
                self.user_pdf = json.load(f)
        else:
            self.user_pdf = {}

    def extract_preprocess_pdf(self,pdf, file_name):
        logger.info("Extracting text using PyPDF2...")
        try:
            reader = PdfReader(pdf)
            docs = []
            for page in reader.pages:
                docs.append(page.extract_text())
            logger.info("extracting pdf is done")

            #create a topic and summary of the pdf
            PROMPT = PDF_SUMMARY_PROMPT.format(pdf=docs)
            summary = self.llm.generate(PROMPT)
            docs.append(f"{file_name} summary: {summary}")
            return docs
        except Exception as e:
            traceback.print_exc()

    def chunking_data(self, datas) -> pd.DataFrame:
        """
        This function is a placeholder for data chunking implementation, 
        which will handle dynamic chunking of various types of documents.

        :datas:A data to be chunked.
        :return: DataFrame with chunked data 
        """
        """Process documents to ensure each chunk has at most self.max_token."""
        
        if isinstance(datas, list) and all(isinstance(d, dict) for d in datas):
            return pd.DataFrame(datas)
        '''
        todo:
        add a token length counter for the models
        add a chunking mechanism for the dict files 
        '''
        result = []
        for doc in datas:
            tokens = doc.split()
            chunks = []
            while len(tokens) > self.max_token:
                chunks.append(" ".join(tokens[:self.max_token]))
                tokens = tokens[self.max_token:]
            if tokens:
                chunks.append(" ".join(tokens))
            result.extend(chunks)
        df =pd.DataFrame({"content":result})
        return df
    
    def get_contents_embed(self, df) -> pd.DataFrame:
        """
        Generates dense embeddings for the content column of the provided DataFrame.

        :param data: DataFrame containing the documents to embed.
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

    def save_doc_to_rag(self, file_name,data,user_id=None,collection_name=VECTOR_COLLECTION):
        """
        Saves the DataFrame with embeddings to the specified Qdrant collection.

        :param collection_name: The name of the collection to save data to.
        :param data: data to be saved.
        :param userid: user ids to be saved when this is passed datas passed will be save in the users collection
        """
        try:
            df = self.chunking_data(data)
            df["filename"] = file_name
            logger.info(f"Embedding contents")
            df = self.get_contents_embed(df)
            if df is not None:
                logger.info(f"Saving data to collection {collection_name}.")
                response = self.client.upsert_data(collection_name, df,user_id)
                return response
        except Exception as e:
            logger.error(f"Embedding generation failed. Data not upserted to collection {collection_name}. and Data {df}")
            logger.error(f"Error saving to collection {collection_name}: {e}")
            traceback.print_exc()

    def save_retrievable_docs(self,file,user_id,filter=True):
        try:
            if user_id not in self.user_pdf:
                self.user_pdf[user_id] = {"count": 0, "names": []}
            
            file_name = file.filename
            if file_name in self.user_pdf[user_id]["names"]:
                return {"error": "PDF already exists."}
            if self.user_pdf[user_id]["count"] >= PDF_LIMIT:
                return {"error": "Your quota is full."}

            data = self.extract_preprocess_pdf(file, file_name)
            saved_data = self.save_doc_to_rag(file_name,data=data,user_id=user_id,collection_name=USERS_PDF_COLLECTION)
            
            self.user_pdf[user_id]["count"]+=1
            self.user_pdf[user_id]["names"].append(file_name)
            
            with open(self.user_pdf_file, 'w') as f:
                json.dump(self.user_pdf,f)

            memory = MemoryManager(self.llm,self.client).add_memory(f"pdf file : {file_name}", user_id)
            return saved_data
        except:
            traceback.print_exc()

    def query(self, query_str: str, user_id=None,collection=VECTOR_COLLECTION, filter=None):
        """
        Processes a query string by generating its embeddings and retrieving related content 
        from the Qdrant vector collection.

        :param query_str: The query string to process.
        :param user_id: The ID of the user making the query.
        :return: Retrieved content from the collection or None if no content is found.
        """
        try:
            if filter:
                collection=USERS_PDF_COLLECTION

            logger.info("Query embedding started.")
            if isinstance(query_str, str):
                query_str = [query_str]  

            query = {}
            embeddings = openai_embedding_model(query_str)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate dense embeddings for the query.")
                return None

            embed = np.array(embeddings)
            query["dense"] = embed.reshape(-1, 1536).tolist()[0]

            result = self.client.retrieve_data(collection, query["dense"],user_id,filter)
            logger.warning("results found for the query.")
            return result
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return {}

    def get_result_from_rag(self, query_str: str, user_id: str):
        """
        Retrieves the result for a query by calling the query method 
        and generating a response based on the retrieved content.

        :param query_str: The query string to process.
        :param user_id: The ID of the user making the request.
        :return: The result from the LLM generated based on the query and retrieved content.
        """
        try:
            logger.info("Generating result for the query.")
            result1 = self.query(query_str=query_str, user_id=user_id)
            result2 = self.query(query_str=query_str, user_id=user_id,filter=True)
            query_result = {**result1, **result2}
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
