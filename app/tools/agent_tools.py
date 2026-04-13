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


class AnnotationTool(BaseTool):
    name: str = "annotation_graph"
    description: str = (
        "Use this tool ONLY for looking up existing structured biological data, "
        "gene-protein interactions, and pathways in the knowledge graph. "
        "Inputs should be specific entities (genes, proteins) or specific relationships. "
        "Do NOT use this for 'generating hypotheses' or 'mechanistic analysis' of variants."
    )
    db_handler: Any = None
    token: str = ""
    user_id: str = "orchestrator"
    shared_state: Any = Field(default=None, exclude=True)

    def _run(self, query: str) -> str:
        try:
            if not self.db_handler:
                return "Annotation graph not initialized."
            
            # Call the full pipeline
            result = self.db_handler.process_annotation_query(
                query=query,
                user_id=self.user_id, 
                query_type="annotation_biological",
                token=self.token
            )
            
            if result.get("success", False):
                summary = result.get("summary", "")
                
                # Check for graph/json data to save for frontend, bypassing the text-only LLM
                if self.shared_state is not None:
                    if "json_format" in result and result.get("json_format"):
                        self.shared_state["resource"] = result["json_format"]
                        
                return summary if summary else "Query processed but no results found."
            else:
                error = result.get("error", "Unknown error")
                return f"Annotation query failed: {error}"
                
        except Exception as e:
            return f"Error in Annotation Tool: {str(e)}"


class HypothesisTool(BaseTool):
    name: str = "hypothesis_generation"
    description: str = (
        "CRITICAL: Use this tool for ALL requests to 'generate a hypothesis' or "
        "analyze the 'mechanism' of specific genetic variants (rsIDs) and tissues. "
        "This tool performs a 4-step biological enrichment and mechanistic generation process. "
        "Priority tool for any 'Hypothesis' or 'Mechanistic' queries."
    )
    hypothesis_instance: Any = None
    token: str = ""
    user_id: str = ""
    shared_state: Any = Field(default=None, exclude=True)

    def _run(self, query: str) -> str:
        try:
            if not self.hypothesis_instance:
                return "Hypothesis tool not initialized."
            
            # Call the generation logic with context
            result = self.hypothesis_instance.generate_hypothesis(
                token=self.token,
                user_query=query,
                user_id=self.user_id
            )
            
            # Intercept the visual graph payload for the frontend
            if self.shared_state is not None and isinstance(result, dict):
                if "resource" in result:
                    self.shared_state["resource"] = result["resource"]
            
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            return str(result)
        except Exception as e:
            return f"Error generating hypothesis: {str(e)}"


class GalaxyTool(BaseTool):
    name: str = "galaxy_tools"
    description: str = (
        "Use this tool to interact with the Galaxy platform for bioinformatics "
        "workflows. Useful for running tools, retrieving tool information, "
        "and managing Galaxy histories."
    )
    galaxy_handler: Any = None

    def _run(self, query: str) -> str:
        try:
            # Placeholder for Galaxy interaction logic
            return f"Interacted with Galaxy for: {query}. (Galaxy functionality pending integration)"
        except Exception as e:
            return f"Error interacting with Galaxy: {str(e)}"



class BioGPTTool(BaseTool):
    name: str = "biogpt_search"
    description: str = (
        "LAST RESORT ONLY. Use this tool ONLY when annotation_graph and rag_search "
        "have both been tried and failed to return a satisfactory answer. "
        "Do NOT use this as a first choice for any question. "
        "Suitable for general biomedical Q&A when no structured data source is available."
    )
    biogpt_agent: Any = None

    def _run(self, query: str) -> str:
        try:
            if self.biogpt_agent:
                answer = self.biogpt_agent.generate_answer(query)
                return answer if answer else "No answer generated."
            return "BioGPT agent not initialized."
        except Exception as e:
            return f"Error in BioGPT: {str(e)}"


class MemoryWriteTool(BaseTool):
    """Tool for storing important facts in memory."""
    name: str = "memory_write"
    description: str = (
        "Store an important biological fact for later retrieval. "
        "Input MUST be in the format 'key: value'. "
        "Example: 'causal_gene: FTO' or 'p_value: 0.005'. "
        "Use this whenever you discover a NEW biological entity, result, or conclusion."
    )
    memory_store: Any = Field(default=None, exclude=True)

    def _run(self, input_str: str) -> str:
        try:
            if not self.memory_store:
                return "Memory store not initialized."
            
            if ":" not in input_str:
                return "Error: Memory write requires 'key: value' format."
            
            key, value = input_str.split(":", 1)
            return self.memory_store.write(key, value)
        except Exception as e:
            return f"Error writing to memory: {str(e)}"

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class MemoryReadTool(BaseTool):
    """Tool for retrieving facts from memory."""
    name: str = "memory_read"
    description: str = (
        "Retrieve a previously stored fact by its key. "
        "Use this if you think you have already discovered relevant data. "
        "Input should be the key name (e.g., 'causal_gene')."
    )
    memory_store: Any = Field(default=None, exclude=True)

    def _run(self, key: str) -> str:
        try:
            if not self.memory_store:
                return "Memory store not initialized."
            return self.memory_store.read(key)
        except Exception as e:
            return f"Error reading from memory: {str(e)}"

    async def _arun(self, key: str) -> str:
        return self._run(key)
