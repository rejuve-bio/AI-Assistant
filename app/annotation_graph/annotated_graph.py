import copy
import json
import logging
from flask import current_app
import random
import requests
import os
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import EXTRACT_RELEVANT_INFORMATION_PROMPT, JSON_CONVERSION_PROMPT, SELECT_PROPERTY_VALUE_PROMPT
from .dfs_handler import *
from app.summarizer import Graph_Summarizer


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
        self.graph_summarizer = Graph_Summarizer(self.llm)


################################################################################################################################################3
    # Mock function to query the knowledge graph locally
    def query_knowledge_graph_local(self, json_query, user_query=None, token=None):
        # Initialize the Neo4j driver and prepare the graph container.
        driver = self.neo4j.get_driver()
        graph = {"nodes": [], "edges": []}
        added_nodes = {}  # To avoid duplicate nodes keyed by node id

        # Helper function to build property conditions and parameters.
        def build_conditions_and_params(properties, prefix="src"):
            conditions = []
            params = {}
            for key, value in properties.items():
                param_key = f"{prefix}_{key}"
                conditions.append(f"{prefix}.{key} = ${param_key}")
                params[param_key] = value
            return " AND ".join(conditions), params

        try:
            with driver.session() as session:
                # Process each predicate in the json_query.
                for predicate in json_query.get("predicates", []):
                    source_key = predicate.get("source")
                    target_key = predicate.get("target")
                    relationship_type = predicate.get("type")
                    
                    # Look up the corresponding node definitions.
                    source_node_def = next((node for node in json_query["nodes"] if node["node_id"] == source_key), None)
                    target_node_def = next((node for node in json_query["nodes"] if node["node_id"] == target_key), None)
                    
                    # Skip this predicate if either node definition is missing.
                    if not source_node_def or not target_node_def:
                        continue

                    source_label = source_node_def.get("type", "")
                    target_label = target_node_def.get("type", "")
                    source_props = source_node_def.get("properties", {})

                    # Build the property condition and parameters for the source node.
                    conditions, params = build_conditions_and_params(source_props, prefix="src")
                    where_clause = f"WHERE {conditions}" if conditions else ""
                    
                    # Construct the dynamic Cypher query.
                    query = f"""
                        MATCH (src:{source_label})
                        {where_clause}
                        MATCH (src)-[r:{relationship_type}]->(tgt:{target_label})
                        RETURN src, tgt
                    """
                    
                    # Execute the query.
                    result = session.run(query, **params)
                    
                    # Process the result records.
                    for record in result:
                        src_node = record["src"]
                        tgt_node = record["tgt"]

                        # Add the source node if not already added.
                        if src_node.id not in added_nodes:
                            node_data = {
                                "id": str(src_node.id),
                                "label": source_label,
                                "properties": dict(src_node)
                            }
                            graph["nodes"].append(node_data)
                            added_nodes[str(src_node.id)] = node_data

                        # Add the target node similarly.
                        if tgt_node.id not in added_nodes:
                            node_data = {
                                "id": str(tgt_node.id),
                                "label": target_label,
                                "properties": dict(tgt_node)
                            }
                            graph["nodes"].append(node_data)
                            added_nodes[str(tgt_node.id)] = node_data

                        # Add the edge for the relationship.
                        edge_data = {
                            "source": str(src_node.id),
                            "target": str(tgt_node.id),
                            "type": relationship_type
                        }
                        graph["edges"].append(edge_data)
                        
        finally:
            driver.close()

        # Reformat the graph so each node and edge is encapsulated within a "data" key.
        formatted_graph = {
            "nodes": [{"data": node} for node in graph["nodes"]],
            "edges": [
                {"data": {"source": edge["source"], "target": edge["target"], "label": edge["type"]}}
                for edge in graph["edges"]
            ]
        }

        response = {}
        response["graph"] = formatted_graph
        logger.info(f"Graph data: {formatted_graph}")
        response["answer"] = self.graph_summarizer.summary(graph=formatted_graph, user_query=user_query)
        response["annotation_id"] = random.randint(100000, 999999)
        return response

###################################################################################################################################
    def query_knowledge_graph(self, json_query, token):
        """
        Query the knowledge graph service.

        Args:
            json_query (dict): The JSON query to be sent.

        Returns:
            dict: The JSON response from the knowledge graph service or an error message.
        """
        logger.info("Starting knowledge graph query...")
        source = "ai-assistant"
        limit = 100
        property =  True
        
        params = {
            "source": source,
            "limit": limit,  
            "properties": property
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
            #logger.info(f"Successfully queried the knowledge graph. 'nodes count': {len(json_response.get('nodes'))} 'edges count': {len(json_response.get('edges', []))}")
           
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"Error querying knowledge graph: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return {"error": f"Failed to query knowledge graph: {str(e)}"}

    def generate_graph(self, query, token):
        try:
            logger.info(f"Starting annotation query processing for question: '{query}'")

            # Extract relevant information
            relevant_information = self._extract_relevant_information(query)
            
            # Convert to initial JSON
            initial_json = self._convert_to_annotation_json(relevant_information, query)
            
            # Validate and update
            validation = self._validate_and_update(initial_json)
            
            # If validation failed, return the intermediate steps
            if validation["validation_report"]["validation_status"] == "failed":
                logger.error("Validation failed for the constructed json query")
                return {"text": f"Unable to generate graph from the query: {query}"}
            
            # Use the updated JSON for subsequent steps
            validated_json = validation["updated_json"]
            validated_json["question"] = query
            # Query knowledge graph with validated JSON
            #####################################################################
            graph = self.query_knowledge_graph_local(json_query= validated_json, user_query= query, token=token)
            #######################################################################
        
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
            return {"text": f"Unable to generate graph from the query: {query}"}

    def _extract_relevant_information(self, query):
        try:
            logger.info("Extracting relevant information from the query.")
            #print(f"enhanced_schema: {self.enhanced_schema}")
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(schema=self.enhanced_schema, query=query)
            extracted_info =  self.llm.generate(prompt, system_prompt='Don\'t take the examples as information, they are just a guide on how to format it ont information. Don\'t return the examples as information Return only the extracted information from the query alone, never add any additional information.')
            logger.info(f"Extracted data: \n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            prompt = JSON_CONVERSION_PROMPT.format(query=query, extracted_information=relevant_information, schema=self.enhanced_schema)
            json_data = self.llm.generate(prompt=prompt, system_prompt='Return only the JSON in the targeted format, from only the extracted information. Never add any other information')
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
                #print(f"processe schema {self.schema_handler.processed_schema}")
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
            selected_value = self.llm.generate(prompt, system_prompt='Output should be as described and nothing more.')
            logger.info(f"Selected value: {selected_value}")
            return selected_value
        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise
