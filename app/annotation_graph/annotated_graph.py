import copy
import json
import logging
import os
import re
import requests
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import (
    EXTRACT_RELEVANT_INFORMATION_PROMPT,
    JSON_CONVERSION_PROMPT,
    SELECT_PROPERTY_VALUE_PROMPT,
    RESULT_SUMMARIZATION_PROMPT,
)
# from app.socket_manager import emit_to_user # Removed for Orchestrator tool compatibility
from .json_to_cypher import JsonToCypherConverter


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class Graph:
    def __init__(self, llm: LLMInterface, schema_handler: SchemaHandler, 
                 annotation_service_url: str = None, use_external_api: bool = False) -> None:
        self.llm = llm
        self.schema_handler = schema_handler
        self.enhanced_schema = (
            schema_handler.enhanced_schema
        )  # Enhanced or preprocessed schema
        self.neo4j = Neo4jConnection(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
        
        # Dual-mode configuration
        self.annotation_service_url = annotation_service_url or os.getenv("ANNOTATION_SERVICE_URL")
        self.use_external_api = use_external_api or bool(self.annotation_service_url)
        
        logger.info(f"Annotation Graph initialized in {'External API' if self.use_external_api else 'Local Neo4j'} mode")

    def query_knowledge_graph(self, json_query, token=None):
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

        params = {"source": source, "limit": limit, "properties": True}
        payload = {"requests": json_query}

        try:
            logger.debug(
                f"Sending request to {self.annotation_service_url} with payload: {payload}"
            )
            
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
            response = requests.post(
                self.annotation_service_url, # + "/query", # Assuming URL is base, usually /query is appended but in implementation plan I said set full URL? 
                # Let's check user env: ANNOTATION_SERVICE_URL=http://localhost:5000/query?limit=100&properties=true
                # The code in other_arch appended /query. 
                # "self.kg_service_url + "/query""
                # If the env var includes parameters, we might need to be careful.
                # Let's assume the env var is the BASE url or full url.
                # Actually, looking at other_arch: os.getenv("ANNOTATION_SERVICE_URL")
                # and usage: self.kg_service_url + "/query"
                # So the env var should be the base URL.
                # But the user provided: ANNOTATION_SERVICE_URL=http://localhost:5000/query?limit=100&properties=true
                # This suggests the user provided the FULL URL with params.
                # I should handle this. Best to just use the URL provided if it looks complete, or append if not.
                # For safety, I'll assume the URL in env might be the full query URL if it has 'query' in it.
                # But to be consistent with other_arch code which did + "/query", I will try to support both or just use what works.
                # Let's stick to the implementation plan: "self.annotation_service_url" is used directly in my plan's code snippet.
                # "response = requests.post(self.annotation_service_url, ...)"
                # So I will use it directly.
                
                json=payload,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            json_response = response.json()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying knowledge graph: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return {"error": f"Failed to query knowledge graph: {str(e)}"}

    def process_annotation_query(
        self, query, user_id, query_type="annotation_biological", token=None
    ):
        # orchestrate the entire annotation pipeline from user query to final response
        try:
            logger.info(
                f"Starting annotation pipeline for query: '{query}', type: {query_type}"
            )

            # Route based on query type
            if query_type == "annotation_general":
                return self._handle_general_query(query, user_id)
            else:
                return self._handle_biological_query(query, user_id, token=token)

        except Exception as e:
            error_msg = f"Unexpected error in annotation pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "json_extraction": "unknown",
                    "cypher_conversion": "unknown",
                    "database_execution": "unknown",
                    "summarization": "unknown",
                },
            }

    def _handle_biological_query(self, query, user_id, token=None):
        try:
            # Extract relevant information
            logger.info("Extracting relevant information from the query.")
            relevant_information = self._extract_relevant_information(query)
            logger.info("Relevant information extraction successful")

            # Convert to initial JSON
            logger.info("Converting to annotation JSON...")
            initial_json = self._convert_to_annotation_json(
                relevant_information, query
            )
            logger.info("Initial JSON conversion successful")

            # Validate and update
            logger.info("Validating JSON structure...")
            validation = self._validate_and_update(initial_json)
            logger.info("JSON validation successful")

            if validation["validation_report"]["validation_status"] == "failed":
                logger.error("JSON validation failed")
                return {
                    "success": True,
                    "summary": "Query structure extracted but validation incomplete.",
                    "json_format": initial_json,
                    "resource": {"id": None, "type": "annotation"},
                }
            
            validated_json = validation["updated_json"]
            logger.info("JSON query structure validated")
            
            # Step 4: Execute query (DUAL MODE LOGIC)
            if self.use_external_api:
                # External API Mode
                logger.info("Executing in External API Mode")
                response = self.query_knowledge_graph(validated_json, token=token)
                
                # Check for error in response
                if "error" in response:
                    return {
                        "success": False,
                        "error": response["error"],
                        "summary": f"External API Error: {response['error']}"
                    }

                return {
                    "success": True,
                    "summary": response.get("answer", "No answer returned from service"),
                    "json_format": validated_json,
                    "resource": {"id": response.get("annotation_id"), "type": "annotation"}
                }
            else:
                # Local Neo4j Mode
                logger.info("Executing in Local Neo4j Mode")
                converter = JsonToCypherConverter()
                cypher_query = converter.convert_to_cypher(validated_json)
                logger.info(f"Generated Cypher: {cypher_query}")
                
                database_results = self.execute_cypher_query(cypher_query)
                
                if not database_results.get("success", False):
                     return {
                        "success": False,
                        "error": database_results.get("error"),
                        "summary": f"Database Error: {database_results.get('error')}"
                    }
                
                summary = self.summarize_results(query, database_results)
                
                return {
                    "success": True,
                    "summary": summary,
                    "json_format": validated_json,
                    "cypher_query": cypher_query,
                    "resource": {"id": None, "type": "annotation"}
                }
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process query: {str(e)}",
                "pipeline_status": {"json_extraction": "failed"},
            }

    def _handle_general_query(self, query, user_id):
        try:
            logger.info(f"Handling general query: '{query}'")

            # Generate simple database summary
            database_summary = self._generate_database_summary()

            # Use LLM to answer the query based on the summary
            summary_prompt = f"""
            Based on this database summary: {database_summary}
            
            Answer this question: {query}
            
            Provide a clear, informative response based on the available data.
            """

            summary = self.llm.generate(summary_prompt)
            logger.info("General query answered successfully")

            return {
                "success": True,
                "summary": summary,
                "cypher_query": None,
                "json_query": None,
                "database_results": {"data": {"summary": database_summary}},
                "error": None,
                "pipeline_status": {
                    "general_query_handling": "success",
                },
            }

        except Exception as e:
            error_msg = f"Error handling general query: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "general_query_handling": "failed",
                },
            }

    def _extract_relevant_information(self, query):
        try:
            logger.info("Extracting relevant information from the query.")
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(
                schema=self.enhanced_schema, query=query
            )
            extracted_info = self.llm.generate(prompt)
            logger.info(f"Extracted data: \n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            prompt = JSON_CONVERSION_PROMPT.format(
                query=query,
                extracted_information=relevant_information,
                schema=self.enhanced_schema,
            )
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
                "validation_status": "success",
            }

            # Create a deep copy to track changes
            updated_json = copy.deepcopy(initial_json)

            # Validate node properties
            if "nodes" not in updated_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")

            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})
                node_id = node.get("node_id")
                node_types[node_id] = node_type

                # Track removed properties
                for property_key in list(properties.keys()):
                    property_value = properties[property_key]

                    if not property_value and property_value != 0:
                        del properties[property_key]
                        validation_report["removed_properties"].append(
                            {
                                "node_type": node_type,
                                "node_id": node_id,
                                "property": property_key,
                                "original_value": property_value,
                            }
                        )
                    elif isinstance(property_value, str):
                        similar_values = self.neo4j.get_similar_property_values(
                            node_type, property_key, property_value
                        )

                        if similar_values:
                            selected_property = (
                                self._select_best_matching_property_value(
                                    property_value, similar_values
                                )
                            )

                            if selected_property.get("selected_value"):
                                new_value = selected_property.get("selected_value")
                                if new_value != property_value:
                                    validation_report["property_changes"].append(
                                        {
                                            "node_type": node_type,
                                            "node_id": node_id,
                                            "property": property_key,
                                            "original_value": property_value,
                                            "new_value": new_value,
                                            "similar_values": similar_values,
                                        }
                                    )
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
                s = node_types.get(edge["source"])
                t = node_types.get(edge["target"])
                rel = edge["type"]
                conn = f"{s}-{rel}-{t}"

                if conn not in self.schema_handler.processed_schema:
                    rev = f"{t}-{rel}-{s}"
                    if rev not in self.schema_handler.processed_schema:
                        raise ValueError(
                            f"Invalid source {s} and target {t} for predicate {rel}"
                        )
                    # Track direction changes
                    validation_report["direction_changes"].append(
                        {
                            "relation_type": rel,
                            "original": f"{s} → {t}",
                            "corrected": f"{t} → {s}",
                        }
                    )
                    # Swap source and target
                    temp_s = edge["source"]
                    edge["source"] = edge["target"]
                    edge["target"] = temp_s

            logger.info(
                f"Validated and updated JSON: \n{json.dumps(updated_json, indent=2)}"
            )

            return {
                "updated_json": updated_json,
                "validation_report": validation_report,
            }

        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            validation_report["validation_status"] = "failed"
            validation_report["error_message"] = str(e)
            return {
                "updated_json": initial_json,
                "validation_report": validation_report,
            }

    def _robust_json_parse(self, text):
        """
        Extracts and parses JSON from a string that might contain extra text or markdown.
        Also handles trailing commas which are common in LLM outputs.
        """
        try:
            # 1. Try to find JSON block using regex
            json_block_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_block_match:
                # If no braces found, maybe it's just a raw value or empty
                return {"selected_value": text.strip(), "confidence_score": 0.3}
                
            json_str = json_block_match.group(0)
            
            # 2. Basic cleanup: remove potential trailing commas before closing braces/brackets
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            raise

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(
                search_query=user_input_value, possible_values=possible_values
            )
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value raw: {selected_value}")
            
            # Fix: Use robust parsing
            try:
                selected_value_json = self._robust_json_parse(selected_value)
                logger.info(f"Selected value parsed: {selected_value_json}")
                return selected_value_json
            except Exception as e:
                logger.error(f"Failed to parse JSON from LLM: {e}")
                
                # Cleanup manually for fallback
                clean_value = selected_value.replace("```json", "").replace("```", "").strip()
                # If there's a lot of text, this isn't enough, but it's a start
                return {"selected_value": user_input_value, "confidence_score": 0.3}

        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise

    def execute_cypher_query(self, cypher_query):
        # Execute a Cypher query against the Neo4j database and return structured results
        try:
            logger.info(f"Executing Cypher query: {cypher_query}")

            driver = self.neo4j.get_driver()
            with driver.session() as session:
                logger.debug("Executing Neo4j query...")
                result = session.run(cypher_query)

                nodes = []
                relationships = []
                node_ids = set()
                rel_ids = set()
                data = {}  # Store scalar values for count queries

                # Extract data from the result
                records = []  # Store all records for multi-record queries
                for record in result:
                    record_data = {}
                    for key, value in record.items():
                        if hasattr(value, "labels"):  # This is a node
                            node_data = {
                                "id": str(value.id),
                                "labels": list(value.labels),
                                "properties": dict(value),
                            }
                            if str(value.id) not in node_ids:
                                nodes.append(node_data)
                                node_ids.add(str(value.id))

                        elif hasattr(value, "type"):  # This is a relationship
                            rel_data = {
                                "id": str(value.id),
                                "type": value.type,
                                "start_node": str(value.start_node.id),
                                "end_node": str(value.end_node.id),
                                "properties": dict(value),
                            }
                            if str(value.id) not in rel_ids:
                                relationships.append(rel_data)
                                rel_ids.add(str(value.id))

                        else:  # This is a scalar value (count, property, etc.)
                            data[key] = value
                            record_data[key] = value

                    if record_data:
                        records.append(record_data)

                # Count results
                counts = {
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships),
                    "result_records": len(list(result.data())),
                }

                # Check if this was a path query that should have returned relationships
                is_path_query = any(
                    rel in cypher_query.lower()
                    for rel in ["transcribed_to", "includes", "transcribed_from"]
                )
                if is_path_query and counts["total_relationships"] == 0:
                    # this indicates an invalid query or missing data
                    logger.warning(
                        f"Path query returned no relationships: {cypher_query}"
                    )
                logger.info(
                    f"Query executed successfully. Found {counts['total_nodes']} nodes and {counts['total_relationships']} relationships"
                )

                return {
                    "success": True,
                    "data": {
                        "nodes": nodes,
                        "relationships": relationships,
                        "counts": counts,
                        "records": records,
                        **data,
                    },
                    "error": None,
                    "cypher_query": cypher_query,
                }

        except Exception as e:
            error_msg = f"Error executing Cypher query: {str(e)}"
            logger.error(error_msg)

            return {
                "success": False,
                "data": {
                    "nodes": [],
                    "relationships": [],
                    "counts": {
                        "total_nodes": 0,
                        "total_relationships": 0,
                        "result_records": 0,
                    },
                },
                "error": error_msg,
                "cypher_query": cypher_query,
            }

    def summarize_results(self, query, results):
        # Use LLM to convert database results into user-friendly natural language responses
        try:
            logger.info(f"Starting result summarization for query: '{query}'")

            # Check if results are valid
            if not results.get("success", False):
                error_msg = results.get("error", "Unknown error occurred")
                logger.error(f"Cannot summarize failed query results: {error_msg}")
                return f"I'm sorry, but I encountered an error while searching the database: {error_msg}"

            # Extract data from results
            data = results.get("data", {})
            nodes = data.get("nodes", [])
            relationships = data.get("relationships", [])
            counts = data.get("counts", {})

            # Handle empty results
            if counts.get("total_nodes", 0) == 0:
                return f"I searched for information about '{query}', but I couldn't find any matching data in the database. Please try rephrasing your question or check if the gene/transcript/exon names are correct."

            # Prepare data for LLM summarization
            summary_data = {
                "original_query": query,
                "nodes_found": len(nodes),
                "relationships_found": len(relationships),
                "node_types": {},
                "key_properties": {},
                "relationships_info": [],
            }

            # Analyze nodes by type
            for node in nodes:
                node_type = (
                    node.get("labels", ["unknown"])[0]
                    if node.get("labels")
                    else "unknown"
                )
                if node_type not in summary_data["node_types"]:
                    summary_data["node_types"][node_type] = []

                # Extract key properties for summarization
                properties = node.get("properties", {})
                key_info = {}
                for prop, value in properties.items():
                    if prop in [
                        "gene_name",
                        "transcript_id",
                        "exon_id",
                        "gene_type",
                        "chr",
                    ]:
                        key_info[prop] = value

                if key_info:
                    summary_data["node_types"][node_type].append(key_info)

            # Analyze relationships
            for rel in relationships:
                rel_info = {
                    "type": rel.get("type", "unknown"),
                    "start_node": rel.get("start_node", "unknown"),
                    "end_node": rel.get("end_node", "unknown"),
                }
                summary_data["relationships_info"].append(rel_info)

            # Create LLM prompt for summarization
            summarization_prompt = self._create_summarization_prompt(
                query, summary_data
            )

            # Generate summary using LLM
            logger.info("Generating summary using LLM...")
            summary = self.llm.generate(summarization_prompt)

            logger.info(f"Successfully generated summary: {summary[:100]}...")
            return summary

        except Exception as e:
            error_msg = f"Error during result summarization: {str(e)}"
            logger.error(error_msg)

            # Fallback to basic summary
            try:
                data = results.get("data", {})
                nodes = data.get("nodes", [])
                counts = data.get("counts", {})

                if counts.get("total_nodes", 0) > 0:
                    return f"I found {counts['total_nodes']} items related to your query '{query}'. However, I encountered an issue while generating a detailed summary."
                else:
                    return f"I couldn't find any information for '{query}' in the database."
            except:
                return f"I'm sorry, but I encountered an error while processing your query '{query}'."

    def _create_summarization_prompt(self, query, summary_data):
        # Create a prompt for the LLM to generate user-friendly summaries.
        # Build node summary
        node_summary = ""
        for node_type, nodes_list in summary_data["node_types"].items():
            node_summary += f"\n- {node_type.capitalize()} nodes: {len(nodes_list)}"
            for node_info in nodes_list[:3]:  # Show first 3 nodes
                node_summary += f"\n  • {node_type.capitalize()}: "
                for prop, value in node_info.items():
                    node_summary += f"{prop}={value}, "
                node_summary = node_summary.rstrip(", ") + "\n"

        # Build relationship summary
        relationship_summary = ""
        if summary_data["relationships_found"] > 0:
            relationship_summary = f"\n**Relationships Found:**\n"
            for rel in summary_data["relationships_info"][
                :5
            ]:  # Show first 5 relationships
                relationship_summary += f"- {rel['type']}: connects nodes {rel['start_node']} and {rel['end_node']}\n"
        else:
            relationship_summary = "No relationships found."

        return RESULT_SUMMARIZATION_PROMPT.format(
            query=query,
            node_summary=node_summary,
            relationship_summary=relationship_summary,
        )
    
    def _generate_database_summary(self):
        try:
            stats_queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as total_nodes",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as total_relationships",
                "node_types": "MATCH (n) RETURN DISTINCT labels(n)[0] as node_type, count(n) as count ORDER BY count DESC",
                "relationship_types": "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type, count(r) as count ORDER BY count DESC",
            }

            summary_parts = []

            for key, query in stats_queries.items():
                try:
                    result = self.execute_cypher_query(query)
                    if result.get("success"):
                        data = result.get("data", {})
                        value = data.get(key)
                        records = data.get("records", [])

                        if value is not None:
                            # Single value (like count queries)
                            summary_parts.append(f"{key}: {value}")
                        elif records:
                            # Multiple records (like node types, relationship types)
                            if key == "node_types":
                                node_types = [
                                    f"{record.get('node_type', 'unknown')} ({record.get('count', 0)})"
                                    for record in records
                                ]
                                summary_parts.append(f"{key}: {', '.join(node_types)}")
                            elif key == "relationship_types":
                                rel_types = [
                                    f"{record.get('rel_type', 'unknown')} ({record.get('count', 0)})"
                                    for record in records
                                ]
                                summary_parts.append(f"{key}: {', '.join(rel_types)}")
                            else:
                                summary_parts.append(f"{key}: {records}")
                        else:
                            summary_parts.append(f"{key}: No data found")
                except Exception as e:
                     logger.warning(f"Failed to fetch stats for {key}: {e}")

            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating database summary: {str(e)}")
            return "Unable to generate database summary."
