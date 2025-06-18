import copy
import json
import logging
from flask import current_app
import requests
import os
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import EXTRACT_RELEVANT_INFORMATION_PROMPT, JSON_CONVERSION_PROMPT, SELECT_PROPERTY_VALUE_PROMPT
from .dfs_handler import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class Graph:
    def __init__(self, llm: LLMInterface, schema_handler:SchemaHandler) -> None:
        self.llm = llm
        self.schema_handler = schema_handler
        self.enhanced_schema = schema_handler.enhanced_schema # Enhanced or preprocessed schema
        self.neo4j = Neo4jConnection(uri=os.getenv('NEO4J_URI'), 
                                    username=os.getenv('NEO4J_USERNAME'), 
                                    password=os.getenv('NEO4J_PASSWORD'))
        self.kg_service_url = os.getenv('ANNOTATION_SERVICE_URL')

    def query_knowledge_graph(self, json_query, token):
        """
        Query the knowledge graph service.

        Args:
            json_query (dict): The JSON query to be sent.

        Returns:
            dict: The JSON response from the knowledge graph service or an error message.
        """
        if isinstance(json_query, str):
            logger.info("passed json is a string changing it to a dicitionary")
            json_query = json.loads(json_query)

        logger.info("Starting knowledge graph query...")
        source = "ai-assistant"
        limit = 100
        
        params = {
            "source": source,
            "limit": limit,  
            "properties": True
        }
        payload = {"requests": json_query}
        
        try:
            logger.debug(f"Sending request to {self.kg_service_url} with payload: {payload}")
            response = requests.post(
                self.kg_service_url+'/query',
                json=payload,
                params=params,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            json_response = response.json()
            # logger.info(f"Successfully queried the knowledge graph. 'nodes count': {len(json_response.get('nodes'))} 'edges count': {len(json_response.get('edges', []))}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying knowledge graph: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return {"error": f"Failed to query knowledge graph: {str(e)}"}

    def validated_json(self,query):
            logger.info(f"Starting annotation query processing for question: '{query}'")

            # Extract relevant information
            relevant_information = self._extract_relevant_information(query)
            
            # Convert to initial JSON
            initial_json = self._convert_to_annotation_json(relevant_information, query)
            
            # Validate and update
            validation = self._validate_and_update(initial_json)
            
            # If validation failed, return the intermediate steps
            if validation["validation_report"]["validation_status"] == "failed":
                logger.error("Validation is failing *****sending the intial json format")
                return {
                    "text": "Here is the structured JSON for your question. Please review and confirm if it's accurate.",
                    "json_format": initial_json,
                }

            # Use the updated JSON for subsequent steps
            validated_json = validation["updated_json"]
            # validated_json["question"] = query
            '''
            TODO
            add query along with job id to specifiy to what query is the json requested is related to.
            '''
            return {
                    "text": "Here is the structured JSON for your question. Please review and confirm if it's accurate.",
                    "json_format": validated_json,
                }

    def generate_graph(self, query, validated_json, token):
        try:        
            graph = self.query_knowledge_graph(validated_json, token) 

            # Generate final answer using validated JSON
            # final_answer = self._provide_text_response(query, validated_json, graph)
            response = {
                "text": graph["answer"],
                "resource": {"id": graph["annotation_id"], 
                             "type": "annotation"},
            }
            logger.info("Completed query processing.")
            return response
            
        except Exception as e:
            logger.error(f"An error occurred during graph generation: {e}")
            return {"text": f"I apologize, but I wasn't able to generate the graph you requested. Could you please rephrase your question or provide additional details so I can better understand what you're looking for?"}

    def _extract_relevant_information(self, query):
        try:
            logger.info("Extracting relevant information from the query.")
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(schema=self.enhanced_schema, query=query)
            extracted_info =  self.llm.generate(prompt)
            logger.info(f"Extracted data: \n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            prompt = JSON_CONVERSION_PROMPT.format(query=query, extracted_information=relevant_information, schema=self.enhanced_schema)
            json_data = self.llm.generate(prompt)
            logger.info(f"Converted JSON:\n{json.dumps(json_data, indent=2)}")
            return json_data
        except Exception as e:
            logger.error(f"Failed to convert information to annotation JSON: {e}")
            raise

    def _validate_and_update(self, initial_json):
        try:
            logger.info("Validating and updating the JSON structure.")
            node_types = {}
            validation_report = {
                "property_changes": [],
                "direction_changes": [],
                "removed_properties": [],
                "validation_status": "success"
            }
            
            # Create a deep copy to track changes
            updated_json = copy.deepcopy(initial_json)
            
            # Validate node properties
            if "nodes" not in updated_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")
                
            for node in updated_json.get("nodes"):
                node_type = node.get('type')
                properties = node.get('properties', {})
                node_id = node.get('node_id')
                node_types[node_id] = node_type
                
                # Track removed properties
                for property_key in list(properties.keys()):
                    property_value = properties[property_key]
                    
                    if not property_value and property_value != 0:
                        del properties[property_key]
                        validation_report["removed_properties"].append({
                            "node_type": node_type,
                            "node_id": node_id,
                            "property": property_key,
                            "original_value": property_value
                        })
                    elif isinstance(property_value, str):
                        similar_values = self.neo4j.get_similar_property_values(
                            node_type, property_key, property_value
                        )
                        
                        if similar_values:
                            selected_property = self._select_best_matching_property_value(
                                property_value, similar_values
                            )
                            
                            if selected_property.get("selected_value"):
                                new_value = selected_property.get("selected_value")
                                if new_value != property_value:
                                    validation_report["property_changes"].append({
                                        "node_type": node_type,
                                        "node_id": node_id,
                                        "property": property_key,
                                        "original_value": property_value,
                                        "new_value": new_value,
                                        "similar_values": similar_values
                                    })
                                properties[property_key] = new_value
                            else:
                                raise ValueError(
                                    f"No suitable property found for {node_type} with key {property_key} "
                                    f"and value {property_value}."
                                )
                        else:
                            raise ValueError(
                                f"No suitable property found for {node_type} with key {property_key} "
                                f"and value {property_value}."
                            )
            
            # Validate edge direction
            for edge in updated_json.get("predicates", []):
                s = node_types.get(edge['source'])
                t = node_types.get(edge['target'])
                rel = edge['type']
                conn = f'{s}-{rel}-{t}'
                
                if conn not in self.schema_handler.processed_schema:
                    rev = f'{t}-{rel}-{s}'
                    if rev not in self.schema_handler.processed_schema:
                        raise ValueError(
                            f"Invalid source {s} and target {t} for predicate {rel}"
                        )
                    # Track direction changes
                    validation_report["direction_changes"].append({
                        "relation_type": rel,
                        "original": f"{s} → {t}",
                        "corrected": f"{t} → {s}"
                    })
                    # Swap source and target
                    temp_s = edge['source']
                    edge['source'] = edge['target']
                    edge['target'] = temp_s

            logger.info(f"Validated and updated JSON: \n{json.dumps(updated_json, indent=2)}")
            
            return {
                "updated_json": updated_json,
                "validation_report": validation_report
            }
            
        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            validation_report["validation_status"] = "failed"
            validation_report["error_message"] = str(e)
            return {
                "updated_json": initial_json,
                "validation_report": validation_report
            }

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(search_query = user_input_value, possible_values=possible_values)
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value: {selected_value}")
            return selected_value
        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise
