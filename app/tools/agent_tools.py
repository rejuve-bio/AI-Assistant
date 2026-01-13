"""Tools for the Orchestrator to interact with specialized agents."""

from langchain.tools import BaseTool
from typing import Optional, Any
from pydantic import Field


class RAGTool(BaseTool):
    """Tool for retrieving information using RAG."""
    name: str = "rag_search"
    description: str = "Search for scientific literature, facts, or background information. Use this when you need to find information about a topic."
    rag_instance: Any = Field(default=None, exclude=True)
    
    def _run(self, query: str) -> str:
        """Execute RAG search."""
        try:
            result = self.rag_instance.get_result_from_rag(query, "system")
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            return str(result)
        except Exception as e:
            return f"Error in RAG search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)

