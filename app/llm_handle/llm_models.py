"""
LangChain-native LLM interface with retry, fallback, and structured output support.

This module provides a unified `LangChainLLM` class that:
- Uses LangChain's `BaseChatModel` internally (ChatGoogleGenerativeAI, ChatOpenAI)
- Preserves the `.generate(prompt)` API for backward compatibility (26 call sites)
- Adds `.generate_structured(prompt, schema)` for type-safe Pydantic output parsing
- Built-in retry with exponential backoff via LangChain's `.with_retry()`
- Built-in fallback chains via `.with_fallbacks()`
"""

from dotenv import load_dotenv
import openai
import time
import os
import logging
import json
from typing import Any, Dict, Type, Optional, TypeVar
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
api = os.getenv("OPENAI_API_KEY")
gemini_api = os.getenv("GEMINI_API_KEY")

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────
# Embedding Functions (unchanged)
# ─────────────────────────────────────────────

def openai_embedding_model(batch):
    openai.api_key = api
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i : i + batch_size]
        print(batch_segment)
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


def gemini_embedding_model(batch):
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i : i + batch_size]
        print(batch_segment)
        logger.info(
            f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}"
        )

        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api)
            response = embeddings_model.embed_documents(batch)
            embeddings.extend(response["embedding"])

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(sleep_time)

    return embeddings


# Load the SentenceTransformer model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")

def sentence_transformer_embedding_model(batch):
    return model.encode(batch, convert_to_numpy=True).tolist()


def get_embedding_vector_size(embedding_fn):
    """Utility to get the vector size for a given embedding function."""
    if embedding_fn == openai_embedding_model:
        return 1536
    elif embedding_fn == gemini_embedding_model:
        return 768
    elif embedding_fn == sentence_transformer_embedding_model:
        return 384
    else:
        raise ValueError("Unknown embedding function")


# ─────────────────────────────────────────────
# LangChain-Native LLM Wrapper
# ─────────────────────────────────────────────

class LLMInterface:
    """Base interface — kept for backward compatibility with type hints."""
    def generate(self, prompt: str, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the generate method")


class LangChainLLM(LLMInterface):
    """
    LangChain-native LLM wrapper with retry, fallback, and structured output.

    Provides:
    - `.generate(prompt)` — backward-compatible string-in/string-out (or parsed JSON dict)
    - `.generate_structured(prompt, schema)` — returns a Pydantic model instance
    - `.chat_model` — direct access to the underlying LangChain BaseChatModel
    - Automatic retry (3 attempts) with exponential backoff on transient errors
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        model_provider: str,
        fallback_model: Optional[BaseChatModel] = None,
        max_retries: int = 3,
    ):
        self.model_provider = model_provider
        self._base_model = chat_model
        self._fallback_model = fallback_model
        self._max_retries = max_retries

        # Build the chain: base model → retry → optional fallback
        self._model_with_retry = chat_model.with_retry(
            stop_after_attempt=max_retries,
            retry_if_exception_type=(Exception,),
        )

        if fallback_model:
            fallback_with_retry = fallback_model.with_retry(
                stop_after_attempt=max_retries,
                retry_if_exception_type=(Exception,),
            )
            self.chat_model = self._model_with_retry.with_fallbacks(
                [fallback_with_retry]
            )
        else:
            self.chat_model = self._model_with_retry

        logger.info(
            f"LangChainLLM initialized: provider={model_provider}, "
            f"model={getattr(chat_model, 'model_name', getattr(chat_model, 'model', 'unknown'))}, "
            f"retries={max_retries}, fallback={'yes' if fallback_model else 'no'}"
        )

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Any:
        """
        Backward-compatible generate method.

        - Sends prompt to the LangChain chat model (with retry + fallback)
        - Attempts to parse JSON from the response
        - Returns parsed dict if valid JSON, otherwise returns raw string

        This preserves behavior for all 26 existing call sites.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.chat_model.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to extract JSON from code blocks (preserving original behavior)
            json_content = self._extract_json_from_codeblock(content)
            try:
                return json.loads(json_content)
            except (json.JSONDecodeError, TypeError):
                return json_content

        except Exception as e:
            logger.error(f"LLM generation failed after {self._max_retries} retries: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: str = None,
    ) -> T:
        """
        Generate a response and parse it into a Pydantic model.

        Uses LangChain's `.with_structured_output()` for reliable JSON parsing.
        Falls back to manual parsing if structured output is not supported.

        Args:
            prompt: The user prompt
            output_schema: A Pydantic BaseModel class to parse the response into
            system_prompt: Optional system prompt

        Returns:
            An instance of output_schema
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            # Try native structured output first (supported by Gemini and OpenAI)
            structured_model = self._base_model.with_structured_output(output_schema)

            # Apply retry to the structured model
            structured_with_retry = structured_model.with_retry(
                stop_after_attempt=self._max_retries,
                retry_if_exception_type=(Exception,),
            )

            result = structured_with_retry.invoke(messages)
            logger.info(f"Structured output parsed successfully: {type(result).__name__}")
            return result

        except Exception as e:
            logger.warning(
                f"Structured output failed ({e}), falling back to manual JSON parsing"
            )
            # Fallback: use regular generate + manual Pydantic parsing
            raw_response = self.generate(prompt, system_prompt=system_prompt)

            if isinstance(raw_response, dict):
                return output_schema.model_validate(raw_response)
            elif isinstance(raw_response, str):
                # Try to parse string as JSON
                cleaned = raw_response.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
                return output_schema.model_validate(parsed)
            else:
                raise ValueError(
                    f"Cannot parse response into {output_schema.__name__}: {raw_response}"
                )

    @staticmethod
    def _extract_json_from_codeblock(content: str) -> str:
        """Extract JSON content from markdown code blocks."""
        if not isinstance(content, str):
            return str(content)
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            return content[start + 7 : end].strip()
        return content


# ─────────────────────────────────────────────
# Deprecated wrappers (backward-compatible aliases)
# ─────────────────────────────────────────────

class GeminiModel(LangChainLLM):
    """Backward-compatible alias. Use LangChainLLM directly for new code."""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "gemini-2.5-flash"):
        chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
        )
        super().__init__(
            chat_model=chat_model,
            model_provider=model_provider,
        )


class OpenAIModel(LangChainLLM):
    """Backward-compatible alias. Use LangChainLLM directly for new code."""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "gpt-3.5-turbo"):
        try:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0,
                max_tokens=1000,
            )
        except ImportError:
            logger.warning(
                "langchain-openai not installed, falling back to legacy OpenAI wrapper. "
                "Install with: pip install langchain-openai"
            )
            # Fallback: use the Gemini-style direct invocation via base ChatModel
            raise ImportError(
                "langchain-openai is required for OpenAI models. "
                "Install it with: pip install langchain-openai"
            )

        super().__init__(
            chat_model=chat_model,
            model_provider=model_provider,
        )


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def get_llm_model(model_provider: str, model_version: str = None) -> LangChainLLM:
    """
    Factory function to create LLM instances.

    Returns a LangChainLLM with retry and (optionally) fallback support.
    The returned object has the same `.generate()` API as the old wrappers.
    """
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
            gemini_api_key, model_provider, model_version or "gemini-2.5-flash"
        )

    else:
        raise ValueError(f"Invalid model provider: {model_provider}")
