import logging
from app.llm_handle.llm_models import get_llm_model
from app.prompts.rag_prompts import (
    KEYWORDS_PROMPT,
    TOPICS_PROMPT,
    SUMMARY_PROMPT,
    QUESTION_GENERATION_PROMPT,
)
import re


logger = logging.getLogger(__name__)


class PDFAnalyzer:
    def __init__(self, use_openai=False):
        # Use the LLM models from llm_models.py
        if use_openai:
            self.llm_model = get_llm_model("openai", "gpt-3.5-turbo")
        else:
            self.llm_model = get_llm_model("gemini", "gemini-1.5-flash")

    def _generate_text(self, prompt):
        # Generate text using the LLM model
        try:
            if hasattr(self.llm_model, "model") and hasattr(
                self.llm_model.model, "generate_content"
            ):
                # For Gemini models
                response = self.llm_model.model.generate_content(prompt)
                return response.text
            elif hasattr(self.llm_model, "generate"):
                # For OpenAI models
                result = self.llm_model.generate(prompt)
                if isinstance(result, dict):
                    return result.get("content", str(result))
                return str(result)
            else:
                raise ValueError("Unsupported LLM model type")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise e

    def _parse_list(self, response):
        text = response.strip()
        numbered = re.findall(r"(?:^\d+\.\s*|\n\d+\.\s*)([^\n]+)", text)
        if len(numbered) >= 2:
            items = [item.strip() for item in numbered if item.strip()]
        else:
            items = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        return [f"{i+1}. {item}" for i, item in enumerate(items)]

    def extract_keywords(self, text_content):
        try:
            max_chars = 8000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            prompt = KEYWORDS_PROMPT.format(text_content=text_content)
            response = self._generate_text(prompt)
            return self._parse_list(response)
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def extract_topics(self, text_content):
        try:
            max_chars = 8000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            prompt = TOPICS_PROMPT.format(text_content=text_content)
            response = self._generate_text(prompt)
            return self._parse_list(response)
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []

    def generate_summary(self, text_content, user_id=None, pdf_id=None):
        try:
            max_chars = 12000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            prompt = SUMMARY_PROMPT.format(text_content=text_content)
            response = self._generate_text(prompt)
            summary_text = (
                response.strip()
                if response and response.strip()
                else "Summary could not be generated: LLM returned empty response."
            )

            return summary_text
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            return "Unable to generate summary at this time."

    def generate_suggested_questions(self, text_content):
        try:
            max_chars = 8000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            prompt = QUESTION_GENERATION_PROMPT.format(text_content=text_content)
            response = self._generate_text(prompt)
            return self._parse_list(response)
        except Exception as e:
            logger.error(f"Error generating suggested questions: {e}")
            return []

    def analyze_pdf_content(self, text_content):
        try:
            keywords = self.extract_keywords(text_content)
            topics = self.extract_topics(text_content)
            summary = self.generate_summary(text_content)
            questions = self.generate_suggested_questions(text_content)
            return {
                "keywords": keywords,
                "topics": topics,
                "summary": summary,
                "suggested_questions": questions,
            }
        except Exception as e:
            logger.error(f"Error in complete PDF analysis: {e}")
            return {
                "keywords": [],
                "topics": [],
                "summary": "Unable to generate summary at this time.",
                "suggested_questions": [],
            }
