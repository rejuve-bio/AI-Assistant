from .prompt import SYSTEM_PROMPT,RETRIEVE_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
import openai
import time
import os
import logging
import json
from typing import Any, Dict
import requests


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
api = os.getenv('OPENAI_API_KEY')

def chat_completion(query,retrieved_content,model: str = "gpt-4o") -> str:
        """
        Generate an answer to a user question based on the provided content.
        """
        try:
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
    openai.api_key = api
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

class LLMInterface:
    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the generate method")


class GeminiModel(LLMInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "topK": 1,
                "topP": 1
            }
        }
        response = requests.post(f"{self.api_url}?key={self.api_key}", headers=headers, json=data)
        response.raise_for_status()

        content = response.json()['candidates'][0]['content']['parts'][0]['text']
    
        json_content = self._extract_json_from_codeblock(content)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return json_content

    def _extract_json_from_codeblock(self, content: str) -> str:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            return content


class OpenAIModel(LLMInterface):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        
        json_content = self._extract_json_from_codeblock(content)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return json_content

    def _extract_json_from_codeblock(self, content: str) -> str:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            return content
