import logging

from app.llm_handle.llm_models import LLMInterface
from prompts.annotation_prompts import EXTRACT_RELEVANT_INFORMATION_PROMPT, JSON_CONVERSION_PROMPT
from .dfs_handler import *
from .llm_handler import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Graph:
    def __init__(self, llm: LLMInterface, schema: str) -> None:
        self.llm = llm
        self.schema = schema # Enhanced or preprocessed schema
        logger.info("Graph instance created with LLM and schema.")

    def query_knowledge_graph(self, validated_json):
        try:
            logger.info("Querying knowledge graph.")
            # Implement the actual query logic here.
            return {}
        except Exception as e:
            logger.error(f"Failed to query knowledge graph: {e}")
            raise

    def process_query(self, query):
        try:
            logger.info(f"Starting query processing for query: '{query}'")
            relevant_information = self._extract_relevant_information(query)
            initial_json = self._convert_to_annotation_json(relevant_information)
            validated_json = self._validate_and_update(initial_json)
            graph = self.query_knowledge_graph(validated_json)
            summary = self.summarize_graph(query, graph)
            logger.info("Completed query processing.")
            return summary, graph
        except Exception as e:
            logger.error(f"An error occurred during graph generation: {e}")
            raise

    def _extract_relevant_information(self, query):
        try:
            logger.info("Extracting relevant information from the query.")
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(schema=self.schema, query=query)
            extracted_info =  self.llm.generate(prompt)
            logger.debug(f"Extracted data: {extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            prompt = JSON_CONVERSION_PROMPT.format(schema=self.schema, query=query, extracted_information=relevant_information)
            json_data = self.llm.generate(prompt)
            logger.debug(f"Converted JSON: {json_data}")
            return json_data
        except Exception as e:
            logger.error(f"Failed to convert information to annotation JSON: {e}")
            raise

    def _validate_and_update(self, initial_json):
        try:
            logger.info("Validating and updating the JSON structure.")
            # Validation and update logic goes here.
            validated_json = initial_json  # Placeholder for actual validation
            logger.debug(f"Validated and updated JSON: {validated_json}")
            return validated_json
        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            raise

    def summarize_graph(self, query, graph):
        try:
            logger.info("Generating summary based on the retrieved graph data.")
            # Summarization logic goes here.
            summary = "Summary"  # Placeholder for actual summary generation
            logger.debug(f"Generated Summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize graph: {e}")
            raise