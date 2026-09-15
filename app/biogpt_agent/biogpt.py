import os
import logging
import requests

logger = logging.getLogger(__name__)

BIOGPT_SERVICE_URL = os.getenv("BIOGPT_SERVICE_URL")

# Reused across calls so each request doesn't repeat the TCP/TLS handshake
# to the remote GPU box.
_session = requests.Session()

FALLBACK_SYSTEM_PROMPT = (
    "You are a biomedical domain expert. Answer the user's biological or medical "
    "question accurately and concisely, using established scientific knowledge — "
    "correct gene symbols, chromosomal locations, molecular mechanisms, and disease "
    "associations where relevant. If you are not confident in a specific detail, "
    "say so rather than guessing. Keep the answer focused and factual, 2-4 sentences."
)

CRITIC_PROMPT_TEMPLATE = (
    "A specialized biomedical model (BioGPT) was asked a question and gave an answer below. "
    "BioGPT is a small model and sometimes misidentifies entities (e.g. misreading an ID's "
    "prefix as an unrelated abbreviation) or drifts off-topic from what was actually asked.\n\n"
    "Review its answer for factual accuracy and relevance to the question:\n"
    "- If it is accurate, or you can confidently correct a small factual error in it, respond "
    "with ONLY the approved/corrected answer text — no meta-commentary about what you changed.\n"
    "- If it is fundamentally wrong, irrelevant to the question, or not something you can "
    "confidently fix, respond with exactly this one word and nothing else: null\n\n"
    "Question: {query}\n"
    "BioGPT's answer: {raw_answer}\n\n"
    "Response:"
)


class BioGPTAgent:
    """
    Calls the dedicated BioGPT inference service (TGI, hosted on the GPU box).
    """

    def __init__(self, basic_llm=None, advanced_llm=None, model_name="kirubel1738/biogpt-bioqa-lora-merged"):
        self.model_name = model_name
        self.critic_llm = basic_llm
        self.fallback_llm = advanced_llm

    def generate_answer(self, query: str, max_length: int = 50) -> str:
        if not BIOGPT_SERVICE_URL:
            logger.warning("BIOGPT_SERVICE_URL not set — using LLM fallback")
            return self._fallback(query)

        try:
            response = _session.post(
                f"{BIOGPT_SERVICE_URL}/generate",
                json={
                    "inputs": query,
                    "parameters": {"max_new_tokens": max_length},
                },
                timeout=30,
            )
            response.raise_for_status()
            raw_answer = response.json()["generated_text"].strip()
            approved = self._critique_and_fix(query, raw_answer)
            if approved is None:
                logger.info("Critic rejected BioGPT's answer as unsalvageable — using LLM fallback")
                return self._fallback(query)
            return approved

        except Exception as e:
            logger.error(f"BioGPT service unavailable, falling back to LLM: {e}", exc_info=True)
            return self._fallback(query)

    def _critique_and_fix(self, query: str, raw_answer: str):
        """
        BioGPT is a small, narrowly fine-tuned model and prone to specific errors (e.g.
        misreading an ID's prefix as an unrelated abbreviation). Have the larger
        general-purpose LLM approve-and-correct its answer, or reject it outright.
        """
        if not self.critic_llm:
            return raw_answer
        try:
            prompt = CRITIC_PROMPT_TEMPLATE.format(query=query, raw_answer=raw_answer)
            result = self.critic_llm.generate(prompt)
            result = result.strip() if isinstance(result, str) else None
            if not result or result.lower() == "null":
                return None
            return result
        except Exception as e:
            logger.error(f"BioGPT critique/fix step failed, using raw answer: {e}", exc_info=True)
            return raw_answer

    def _fallback(self, query: str) -> str:
        if not self.fallback_llm:
            return "BioGPT service is currently unavailable."
        try:
            return self.fallback_llm.generate(query, system_prompt=FALLBACK_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"LLM fallback also failed: {e}", exc_info=True)
            return "BioGPT service is currently unavailable."
