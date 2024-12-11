import google.generativeai as genai
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
GEMINI_EMBEDDING_MODEL="models/text-embedding-004"
api = os.getenv('OPENAI_API_KEY')

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

# Function to generate gemini embeddings
def gemini_embedding_model(batch):
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i:i + batch_size]
        print(batch_segment)
        logger.info(f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}")

    
        genai.configure(api_key=api)
        try:
                response = genai.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    content=batch_segment
                )
                batch_embeddings = response['embedding']
                embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(sleep_time)
    
    return embeddings


def get_llm_model(model_provider, model_version=None):
    # model_type = config['LLM_MODEL']

    if model_provider == 'openai':
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found")
        
        return OpenAIModel(openai_api_key, model_provider, model_version or "gpt-3.5-turbo")
    elif model_provider == 'gemini':
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("Gemini API key not found")
        return GeminiModel(gemini_api_key, model_provider, model_version or "gemini-pro")
    else:
        raise ValueError("Invalid model type in configuration")


class LLMInterface:
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the generate method")



class GeminiModel(LLMInterface):
    def __init__(self, api_key: str, model_provider,model_name="gemini-pro"): 
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name or "gemini-pro")
        self.model_name = model_name
        self.model_provider = model_provider
        self.api_key = api_key

    def generate(self, prompt: str,system_prompt=None, temperature=0.0, top_k=1) -> Dict[str, Any]:
        response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    top_k=top_k
                )
            )
            
        content = response.text

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
    def __init__(self, api_key: str, model_provider, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.model_provider = model_provider
        openai.api_key = self.api_key
    
    def generate(self, prompt: str, system_prompt=None) -> Dict[str, Any]:
        if system_prompt:
            response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        else:
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