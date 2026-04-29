import logging
from typing import List

logger = logging.getLogger(__name__)


class SimpleWebSearch:
    def get_context_urls(self, query: str, num_results: int = 3) -> List[str]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            return [r["href"] for r in results]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
