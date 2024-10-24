
from .prompt import SYSTEM_PROMPT,RETRIEVE_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
import openai
import time
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
EMBEDDING_MODEL = "text-embedding-3-small"
api = os.getenv('OPENAI_KEY')

def chat_completion(query,retrieved_content,model: str = "gpt-4o") -> str:
        """
        Generate an answer to a user question based on the provided content.
        """
        try:
            print("chat completion")
           
            client =  OpenAI(api_key=api)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content":f'user asked {query} and similiar result filtered are {retrieved_content}'
                                f"{RETRIEVE_PROMPT}"
                    },
                ]
            )
            result =  response.choices[0].message.content
            return result
        except Exception as e:
            print(e)
            return ""
    
'''
todo list: gemini chat completion and embbedding model
'''


# Function to generate OpenAI embeddings
def openai_embedding_model(batch):
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i:i + batch_size]
        print(batch_segment)
        logger.info(f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}")

        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_segment
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(sleep_time)
    
    return embeddings
