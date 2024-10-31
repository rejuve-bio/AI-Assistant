import logging
import os
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.llm_handle.llm_models import LLMInterface
from prompts.annotation_prompts import EXTRACT_RELEVANT_INFORMATION_PROMPT, JSON_CONVERSION_PROMPT, SELECT_PROPERTY_VALUE_PROMPT
from .dfs_handler import *
from .llm_handler import *

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Graph:
    def __init__(self, llm: LLMInterface, schema: str) -> None:
        self.llm = llm
        self.schema = schema # Enhanced or preprocessed schema
        self.neo4j = Neo4jConnection(uri=os.getenv('NEO4J_URI'), 
                                            username=os.getenv('NEO4J_USERNAME'), 
                                            password=os.getenv('NEO4J_PASSWORD'))

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

            # Validate node properties
            if "nodes" not in initial_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")
            for node in initial_json.get("nodes"):
                node_type = node.get('type')
                properties = node.get('properties', {})

                for property_key in list(properties.keys()):
                    property_value = properties[property_key]

                    if not property_value and property_value != 0:
                        del properties[property_key]
                    elif isinstance(property_value, str):
                        similar_values = self.neo4j.get_similar_property_values(node_type, property_key, property_value)
                        if similar_values:
                            selected_property_value = self._select_best_matching_property_value(property_value, similar_values)
                            if selected_property_value.get("selected_value"):
                                properties[property_key] = selected_property_value.get("selected_value")
                            else:
                                logger.debug(f"No suitable property found for {node_type} with key {property_key} and value {property_value}.")
                                raise ValueError(f"No suitable property found for {node_type} with key {property_key} and value {property_value}.") 
                        else:
                            logger.debug(f"No suitable property found for {node_type} with key {property_key} and value {property_value}.")
                            raise ValueError(f"No suitable property found for {node_type} with key {property_key} and value {property_value}.")

            logger.debug(f"Validated and updated JSON: {initial_json}")
            return initial_json
        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            raise

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(search_query = user_input_value, possible_values=possible_values)
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value: {selected_value}")
            return selected_value
        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise
