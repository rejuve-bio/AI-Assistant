from dotenv import load_dotenv
import openai
import time
import os
import logging
import json
from typing import Any, Dict
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
api = os.getenv("OPENAI_API_KEY")
gemini_api = os.getenv("GEMINI_API_KEY")


# Function to generate OpenAI embeddings
def openai_embedding_model(batch):
    openai.api_key = api
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i : i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}"
        )

        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL, input=batch_segment
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
        batch_segment = batch[i : i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}"
        )

        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_api
            )
            response = embeddings_model.embed_documents(batch)
            embeddings.extend(response["embedding"])

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(sleep_time)

    return embeddings


# Load the SentenceTransformer model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate sentence transformers embeddings
def sentence_transformer_embedding_model(batch):
    return model.encode(batch, convert_to_numpy=True).tolist()


def get_embedding_vector_size(embedding_fn):
    if embedding_fn == openai_embedding_model:
        return 1536
    elif embedding_fn == gemini_embedding_model:
        return 768
    elif embedding_fn == sentence_transformer_embedding_model:
        return 384
    else:
        raise ValueError("Unknown embedding function")


def get_llm_model(model_provider, model_version=None):
    if model_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found")
        return OpenAIModel(
            openai_api_key, model_provider, model_version or "gpt-3.5-turbo"
        )

    elif model_provider == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found")
        return GeminiModel(
            gemini_api_key, model_provider, model_version or "gemini-pro"
        )

    elif model_provider == "ollama":
        return OllamaModel(
            model_name=model_version or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        )

    else:
        raise ValueError("Invalid model type in configuration")


class LLMInterface:
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the generate method")


class OllamaModel(LLMInterface):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        self.model_provider = "ollama"
        host = os.getenv("OLLAMA_HOST")
        self.client = openai.OpenAI(
            base_url=f"{host}/v1",
            api_key="ollama"
        )
        logger.info(f"OllamaModel initialized: {self.model_name} at {host}")

    def generate(self, prompt: str, system_prompt=None, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=1000,
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
            return content[start + 7 : end].strip()
        return content


class GeminiModel(LLMInterface):
    def __init__(self, api_key: str, model_provider, model_name="gemini-2.5-flash"):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
        )
        self.model_provider = model_provider

    def generate(self, prompt: str, system_prompt=None, top_k=1) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.model.invoke(messages)
        content = getattr(response, "content", response)
        json_content = self._extract_json_from_codeblock(content)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return json_content

    def _extract_json_from_codeblock(self, content: str) -> str:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            return content[start + 7 : end].strip()
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1000,
            )
        else:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
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
            json_content = content[start + 7 : end].strip()
            return json_content
        else:
            return content
