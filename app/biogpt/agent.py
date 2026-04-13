"""BioGPT Agent for biomedical question answering."""

from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import logging
import os
from threading import Lock

logger = logging.getLogger(__name__)


class BioGPTAgent:
    """
    Lazy-loaded BioGPT agent for biomedical question answering.
    Uses the kirubel1738/biogpt-bioqa-lora-merged model.
    """

    _model = None
    _tokenizer = None
    _device = None
    _lock = Lock()

    def __init__(self, llm=None, model_name="kirubel1738/biogpt-bioqa-lora-merged"):
        """
        Initialize BioGPT agent.
        
        Args:
            llm: Optional LLM instance (for compatibility, not used)
            model_name: HuggingFace model name to use
        """
        self.model_name = model_name
        self.llm = llm
        self.service_url = os.getenv("BIOGPT_SERVICE_URL")
        self.api_timeout = int(os.getenv("BIOGPT_API_TIMEOUT", "30"))

    def _load_if_needed(self):
        """Load model/tokenizer once lazily using class-level caching."""
        if BioGPTAgent._model is None:
            with BioGPTAgent._lock:
                # Double-check inside lock
                if BioGPTAgent._model is None:
                    logger.info(f"Lazy-loading BioGPT model: {self.model_name}...")

                    BioGPTAgent._tokenizer = BioGptTokenizer.from_pretrained(self.model_name)
                    BioGPTAgent._model = BioGptForCausalLM.from_pretrained(self.model_name)

                    BioGPTAgent._device = "cuda" if torch.cuda.is_available() else "cpu"
                    BioGPTAgent._model.to(BioGPTAgent._device)
                    BioGPTAgent._model.eval()  # Set to evaluation mode

                    logger.info(f"BioGPT loaded successfully on {BioGPTAgent._device}")

    def _generate_via_api(self, query: str, max_length: int) -> str:
        """
        Generate answer using the remote BioGPT API.
        
        Args:
            query: The biomedical question
            max_length: Maximum generation length
            
        Returns:
            str: Generated answer
            
        Raises:
            requests.RequestException: If the API call fails
        """
        import requests
        
        url = f"{self.service_url}/generate"
        payload = {
            "prompt": query,
            "max_length": max_length
        }
        
        logger.info(f"Generating answer via Remote BioGPT API: {url}")
        response = requests.post(url, json=payload, timeout=self.api_timeout)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response found.")

    def generate_answer(self, query: str, max_length: int = 150) -> str:
        """
        Generate an answer to a biomedical question.
        Agorithm:
        1. Try Remote API if configured.
        2. If API fails or not configured, fallback to Local Model.
        
        Args:
            query: The biomedical question to answer
            max_length: Maximum length for generation (default: 150)
            
        Returns:
            Generated answer text
        """
        # 1. Try Remote API
        if self.service_url:
            try:
                return self._generate_via_api(query, max_length)
            except Exception as e:
                logger.warning(f"Remote BioGPT failed ({str(e)}). Falling back to local model.")
        
        # 2. Local Fallback
        try:
            self._load_if_needed()

            inputs = BioGPTAgent._tokenizer(query, return_tensors="pt").to(BioGPTAgent._device)

            with torch.no_grad():
                output_ids = BioGPTAgent._model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=BioGPTAgent._tokenizer.eos_token_id,
                )

            answer = BioGPTAgent._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Remove the question from the start of the answer if present
            if answer.startswith(query):
                answer = answer[len(query):].strip()
            
            return answer.strip()

        except Exception as e:
            logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
            return f"BioGPT Error: {str(e)}"
