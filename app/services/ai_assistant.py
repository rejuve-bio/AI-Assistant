import json
from typing import Any, Dict

from app.services.llm_handler import LLMPromptHandler
from .llm_models import LLMInterface
from app.services.graph_handler import DFSHandler


class AIAssistantSystem:
    def __init__(self, llm: LLMInterface, schema: str):
        self.prompt_handler = LLMPromptHandler(llm, schema)
        self.dfs_traversal = DFSHandler(llm,schema)

    def process_query(self, query: str) -> Dict[str, Any]:
        extracted_info = self.prompt_handler.process_initial_query(query)
        print("================== Extracted Info ===============")
        print(extracted_info)
        return self.prompt_handler.convert_to_json_format(extracted_info, query)
    
    def process_query_traversing(self, query: str) -> Dict[str, Any]:
        extracted_info = self.dfs_traversal.json_format(query)
        print("================== Extracted Info ===============")
        return extracted_info

    def process_kg_response(self, original_query: str, json_query: Dict[str, Any], kg_response: Dict[str, Any]):
        return self.prompt_handler.process_kg_response(original_query, json_query, kg_response)

   