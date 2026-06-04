import copy
import json
import logging
import os
import requests
from dotenv import load_dotenv
from app.tools.platform.annotation.neo4j_handler import Neo4jConnection
from app.tools.platform.annotation.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import (
    EXTRACT_RELEVANT_INFORMATION_PROMPT,
    JSON_CONVERSION_PROMPT,
    SELECT_PROPERTY_VALUE_PROMPT,
    RESULT_SUMMARIZATION_PROMPT,
)
from app.socket_manager import emit_to_user
from .json_to_cypher import JsonToCypherConverter

# Maps Neo4j node labels → the source databases they were built from.
NODE_LABEL_TO_SOURCE_DB = {
    "Gene":                   ["GENCODE v44"],
    "Transcript":             ["GENCODE v44"],
    "Exon":                   ["GENCODE v44"],
    "Protein":                ["UniProt"],
    "Variant":                ["ClinVar", "dbSNP"],
    "StructuralVariant":      ["dbVar", "ClinVar"],
    "RegulatoryElement":      ["ENCODE", "Roadmap Epigenomics"],
    "Enhancer":               ["ENCODE SCREEN", "Roadmap Epigenomics"],
    "Promoter":               ["ENCODE SCREEN"],
    "TFBS":                   ["JASPAR", "ENCODE"],
    "SuperEnhancer":          ["dbSUPER"],
    "TAD":                    ["3D Genome Browser"],
    "eQTL":                   ["GTEx v8"],
    "ProteinInteraction":     ["STRING v12"],
    "Pathway":                ["REACTOME", "GO"],
    "GOTerm":                 ["Gene Ontology"],
    "TissueExpression":       ["GTEx v8"],
    "ChromatinAccessibility": ["ENCODE", "Roadmap Epigenomics"],
    "TFRegulation":           ["TFLink", "JASPAR"],
}


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class Graph:
    def __init__(self, llm: LLMInterface, schema_handler: SchemaHandler) -> None:
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
        self.kg_service_url = os.getenv("ANNOTATION_SERVICE_URL", "")

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

        params = {"source": source, "limit": limit, "properties": True}
        payload = {"requests": json_query}

        try:
            logger.debug(
                f"Sending request to {self.kg_service_url} with payload: {payload}"
            )
            response = requests.post(
                self.kg_service_url + "/query",
                json=payload,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
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

    def validated_json(self, query, user_id):
        logger.info(f"Starting annotation query processing for question: '{query}'")

        # Extract relevant information
        relevant_information = self._extract_relevant_information(query)

        # Convert to initial JSON
        emit_to_user(user=user_id, message=f"Validating Constructed Json Format...")
        initial_json = self._convert_to_annotation_json(relevant_information, query)

        # Validate and update
        validation = self._validate_and_update(initial_json)

        # If validation failed, return the intermediate steps
        if validation["validation_report"]["validation_status"] == "failed":
            logger.error("Validation is failing *****sending the intial json format")
            return {
                "text": None,
                "json_format": initial_json,
                "resource": {"id": None, "type": "annotation"},
            }

        # Use the updated JSON for subsequent steps
        validated_json = validation["updated_json"]
        # validated_json["question"] = query
        """
            TODO
            add query along with job id to specifiy to what query is the json requested is related to.
            """
        return {
            "text": None,
            "json_format": validated_json,
            "resource": {"id": None, "type": "annotation"},
        }

    def generate_graph(self, query, validated_json, token):
        try:
            graph = self.query_knowledge_graph(validated_json, token)



            response = {
                "text": graph["answer"],
                "resource": {"id": graph["annotation_id"], "type": "annotation"},
            }
            # Store summary in Redis cache for 24 hours
            # redis_manager.create_graph(graph_id=graph_id, graph_summary=summary_text)

            logger.info("Completed query processing.")
            return response

        except Exception as e:
            logger.error(f"An error occurred during graph generation: {e}")
            return {
                "text": f"I apologize, but I wasn't able to generate the graph you requested. Could you please rephrase your question or provide additional details so I can better understand what you're looking for?"
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
                "failed_nodes": [],
                "validation_status": "success",
            }

            # Create a deep copy to track changes
            updated_json = copy.deepcopy(initial_json)

            # Validate node properties
            if "nodes" not in updated_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")

            # Pre-pass: collect all values that need Neo4j lookup grouped by (node_type, property_key)
            lookup_needed = {}  # (node_type, property_key) -> set of string values
            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})
                for property_key, property_value in properties.items():
                    if not property_value and property_value != 0:
                        continue
                    if node.get("is_list") and isinstance(property_value, (str, list)):
                        items = (
                            [i.strip() for i in property_value.split(",") if i.strip()]
                            if isinstance(property_value, str)
                            else property_value
                        )
                        lookup_needed.setdefault((node_type, property_key), set()).update(items)
                    elif isinstance(property_value, str):
                        lookup_needed.setdefault((node_type, property_key), set()).add(property_value)

            # Run one batch Neo4j query per (node_type, property_key)
            similarity_cache = {}  # (node_type, property_key, value) -> [(similar_value, score), ...]
            for (node_type, property_key), values in lookup_needed.items():
                batch = self.neo4j.get_similar_property_values_batch(node_type, property_key, list(values))
                for value, matches in batch.items():
                    similarity_cache[(node_type, property_key, value)] = matches

            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})
                node_id = node.get("node_id")
                node_types[node_id] = node_type
                if not node.get("is_list"):
                    node["status"] = True

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
                    elif node.get("is_list") and isinstance(property_value, (str, list)):
                        items = (
                            [i.strip() for i in property_value.split(",") if i.strip()]
                            if isinstance(property_value, str)
                            else property_value
                        )
                        validated_items = []
                        failed_items = []
                        item_suggestions = {}
                        for item in items:
                            similar_values = similarity_cache.get((node_type, property_key, item), [])
                            if similar_values:
                                selected_property = self._select_best_matching_property_value(
                                    item, similar_values
                                )
                                if selected_property.get("selected_value"):
                                    validated_items.append(selected_property["selected_value"])
                                else:
                                    failed_items.append(item)
                                    top = similar_values[0][0]
                                    if top.lower() != item.lower():
                                        item_suggestions[item] = top
                            else:
                                failed_items.append(item)
                        if failed_items:
                            # Only put confirmed items in the property.
                            # Failed items with suggestions go to pending_substitutions
                            # so _apply_pending_substitutions can append them after confirmation.
                            properties[property_key] = ", ".join(validated_items)
                            node["status"] = False
                            node["not_validated"] = failed_items
                            node["all_list_values"] = items
                            if item_suggestions:
                                node["suggestions"] = item_suggestions
                                node["pending_substitutions"] = item_suggestions
                            validation_report["failed_nodes"].append(
                                {"node_id": node_id, "reason": f"Could not find in database: {failed_items}"}
                            )
                        else:
                            properties[property_key] = ", ".join(validated_items)
                            node["status"] = True

                    elif isinstance(property_value, str):
                        similar_values = similarity_cache.get((node_type, property_key, property_value), [])

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
                                node["status"] = False
                                node["validation_error"] = f"No suitable match found for '{property_value}' in the database."
                                top = similar_values[0][0]
                                if top.lower() != property_value.lower():
                                    node["suggestion"] = top
                                validation_report["failed_nodes"].append(
                                    {"node_id": node_id, "reason": node["validation_error"]}
                                )
                        else:
                            node["status"] = False
                            node["validation_error"] = f"'{property_value}' not found in the database."
                            validation_report["failed_nodes"].append(
                                {"node_id": node_id, "reason": node["validation_error"]}
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

            for node in updated_json.get("nodes", []):
                node.pop("is_list", None)

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

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(
                search_query=user_input_value, possible_values=possible_values
            )
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value: {selected_value}")
            return selected_value
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

    def process_annotation_query(
        self, query, user_id, query_type="annotation_biological"
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
                return self._handle_biological_query(query, user_id)

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

    def _handle_biological_query(self, query, user_id):
        try:
            emit_to_user(user=user_id, message="Extracting relevant information from your query...")
            relevant_information = self._extract_relevant_information(query)

            emit_to_user(user=user_id, message="Validating constructed JSON format...")
            initial_json = self._convert_to_annotation_json(relevant_information, query)

            validation = self._validate_and_update(initial_json)

            # Hard validation failure (schema error, DB exception)
            if validation["validation_report"]["validation_status"] == "failed":
                error_msg = validation["validation_report"].get(
                    "error_message", "Annotation structure validation failed."
                )
                logger.error(f"Validation failed: {error_msg}")
                return {"success": False, "error": error_msg}

            json_format = validation["updated_json"]

            # Check for list-nodes with unconfirmed substitutions
            unconfirmed = []
            for node in json_format.get("nodes", []):
                for orig, sug in node.get("pending_substitutions", {}).items():
                    unconfirmed.append({
                        "original": orig,
                        "suggestion": sug,
                        "all_list_values": node.get("all_list_values", []),
                    })

            if unconfirmed:
                logger.info(f"Annotation needs confirmation for {len(unconfirmed)} item(s)")
                return {
                    "needs_confirmation": True,
                    "pending_json": json_format,
                    "unconfirmed_nodes": unconfirmed,
                }

            logger.info(f"Annotation JSON built: {len(json_format.get('nodes', []))} nodes")
            return {
                "success": True,
                "json_format": json_format,
                "validation_report": validation["validation_report"],
            }

        except Exception as e:
            logger.error(f"Biological query pipeline error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _handle_general_query(self, query, user_id):
        """Full NL → JSON → Cypher → execute pipeline. Returns actual data + provenance.
        Falls back to _handle_stats_query if Cypher conversion fails."""
        try:
            logger.info(f"Handling general data query: '{query}'")

            relevant_information = self._extract_relevant_information(query)
            initial_json = self._convert_to_annotation_json(relevant_information, query)
            validation = self._validate_and_update(initial_json)
            validated_json = validation["updated_json"]

            converter = JsonToCypherConverter()
            cypher_query = converter.convert_to_cypher(validated_json)

            database_results = self.execute_cypher_query(cypher_query)

            if not database_results.get("success", False):
                logger.warning("Cypher query failed, falling back to stats summary")
                return self._handle_stats_query(query, user_id)

            provenance = self._build_provenance(database_results)
            summary = self.summarize_results(query, database_results)

            return {
                "success": True,
                "summary": summary,
                "cypher_query": cypher_query,
                "database_results": database_results,
                "provenance": provenance,
                "error": None,
                "pipeline_status": {
                    "json_extraction": "success",
                    "cypher_conversion": "success",
                    "database_execution": "success",
                    "summarization": "success",
                },
            }

        except Exception as e:
            logger.warning(f"General data query failed ({e}), falling back to stats summary")
            return self._handle_stats_query(query, user_id)

    def _handle_stats_query(self, query, user_id):
        """Answers general/stats questions about the database (counts, schema, node types)."""
        try:
            logger.info(f"Handling stats query: '{query}'")
            database_summary = self._generate_database_summary()
            summary_prompt = f"""
            Based on this database summary: {database_summary}

            Answer this question: {query}

            Provide a clear, informative response based on the available data.
            """
            summary = self.llm.generate(summary_prompt)
            return {
                "success": True,
                "summary": summary,
                "cypher_query": None,
                "database_results": {"data": {"summary": database_summary}},
                "error": None,
                "pipeline_status": {"stats_query_handling": "success"},
            }
        except Exception as e:
            error_msg = f"Error handling stats query: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {"stats_query_handling": "failed"},
            }

    def _build_provenance(self, db_results):
        """Reads node labels from query results and maps them to source databases."""
        data = db_results.get("data", {})
        nodes = data.get("nodes", [])

        hit_labels = set()
        for node in nodes:
            for label in node.get("labels", []):
                hit_labels.add(label)

        source_dbs = []
        for label in hit_labels:
            source_dbs.extend(NODE_LABEL_TO_SOURCE_DB.get(label, []))

        return {
            "database": "Neo4j (Rejuve annotation graph)",
            "node_types_queried": sorted(hit_labels),
            "source_databases": sorted(set(source_dbs)),
        }

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
                    else:
                        summary_parts.append(f"{key}: Unable to retrieve")

                except Exception as e:
                    logger.warning(f"Failed to execute {key} query: {e}")
                    summary_parts.append(f"{key}: Error retrieving")

            return "Database Summary:\n" + "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Failed to generate database summary: {e}")
            return "Database Summary:\nUnable to retrieve database information due to an error."
