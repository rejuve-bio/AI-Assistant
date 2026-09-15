import logging

from app.prompts.annotation_prompts import RESULT_SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)


class JsonToCypherConverter:
    """
    Converts validated JSON queries to executable Cypher queries.
    """

    def __init__(self):
        self.supported_node_types = {
            "gene", "transcript", "exon", "protein",
            "promoter", "enhancer", "super_enhancer", "non_coding_rna",
            "regulatory_region", "snp", "structural_variant", "sequence_variant",
            "pathway", "reaction", "tad", "motif", "tfbs",
            "chromosome_chain", "chromosome",
            "anatomy", "tissue", "cell_type", "cell_line",
            "experimental_factor", "biological_process",
            "molecular_function", "molecular_interaction", "cellular_component",
            "developmental_stage", "disease", "phenotype",
            "small_molecule", "sequence_type",
        }
        self.supported_relationships = {
            "transcribes_to", "transcribed_from",
            "translates_to", "translation_of",
            "part_of", "expressed_in",
            "participates_in", "involved_in",
            "regulates", "negatively_regulates", "positively_regulates",
            "binds_to", "in_tad_region",
            "eqtl_association", "closest_gene",
            "upstream_of", "downstream_of", "located_in",
            "activity_by_contact", "accessible_in",
            "chromatin_state", "in_dnase_I_hotspot", "histone_modification",
            "interacts_with", "coexpressed_with",
            "associated_with", "ortholog_of", "alters_binding",
            "in_linkage_disequilibrium_with",
            "enables", "produced_by",
            "is_a", "capable_of", "has_xref", "equivalent_to",
            "child_pathway_of", "parent_pathway_of",
            "overlaps", "lower_resolution", "located_on_chain",
        }

    def convert_to_cypher(self, json_query):
        try:
            logger.info("Starting JSON to Cypher conversion")
            logger.debug(f"Input JSON: {json_query}")

            self._validate_json_structure(json_query)

            nodes = json_query.get("nodes", [])
            predicates = json_query.get("predicates", [])

            cypher_query = self._build_cypher_query(nodes, predicates)

            logger.info(f"Successfully converted to Cypher: {cypher_query}")
            return cypher_query

        except Exception as e:
            logger.error(f"Error converting JSON to Cypher: {str(e)}")
            raise ValueError(f"Failed to convert JSON to Cypher: {str(e)}")

    def _validate_json_structure(self, json_query):
        if not isinstance(json_query, dict):
            raise ValueError("JSON query must be a dictionary")

        if "nodes" not in json_query:
            raise ValueError("JSON query must contain 'nodes' key")

        if not isinstance(json_query["nodes"], list):
            raise ValueError("'nodes' must be a list")

        if "predicates" in json_query and not isinstance(
            json_query["predicates"], list
        ):
            raise ValueError("'predicates' must be a list")

        # Validate nodes
        for node in json_query["nodes"]:
            self._validate_node(node)

        # Validate predicates
        if "predicates" in json_query:
            for predicate in json_query["predicates"]:
                self._validate_predicate(predicate, json_query["nodes"])

    def _validate_node(self, node):
        required_keys = {"node_id", "type", "id", "properties"}
        if not all(key in node for key in required_keys):
            raise ValueError(f"Node missing required keys: {required_keys}")

        if node["type"] not in self.supported_node_types:
            raise ValueError(f"Unsupported node type: {node['type']}")

        if not isinstance(node["properties"], dict):
            raise ValueError("Node properties must be a dictionary")

    def _validate_predicate(self, predicate, nodes):
        required_keys = {"type", "source", "target"}
        if not all(key in predicate for key in required_keys):
            raise ValueError(f"Predicate missing required keys: {required_keys}")

        if predicate["type"] not in self.supported_relationships:
            raise ValueError(f"Unsupported relationship type: {predicate['type']}")

        # Validate source and target exist in nodes
        node_ids = {node["node_id"] for node in nodes}
        if predicate["source"] not in node_ids:
            raise ValueError(
                f"Predicate source '{predicate['source']}' not found in nodes"
            )
        if predicate["target"] not in node_ids:
            raise ValueError(
                f"Predicate target '{predicate['target']}' not found in nodes"
            )

    def _build_cypher_query(self, nodes, predicates):
        # Build MATCH clause and collect relationship variables
        match_clause, rel_vars = self._build_match_clause(nodes, predicates)

        # Build WHERE clause
        where_clause = self._build_where_clause(nodes)

        # Build RETURN clause (include relationship vars when present)
        return_clause = self._build_return_clause(nodes, rel_vars)

        # Combine clauses
        cypher_parts = [match_clause]
        if where_clause:
            cypher_parts.append(where_clause)
        cypher_parts.append(return_clause)

        return " ".join(cypher_parts)

    def _build_match_clause(self, nodes, predicates):
        if not predicates:
            # No relationships, just match nodes
            node_patterns = [self._build_node_pattern(node) for node in nodes]
            return f"MATCH {', '.join(node_patterns)}", []

        # Build path patterns with relationship variables
        path_patterns = []
        rel_vars = []
        for idx, predicate in enumerate(predicates, start=1):
            source_node = next(n for n in nodes if n["node_id"] == predicate["source"])
            target_node = next(n for n in nodes if n["node_id"] == predicate["target"])

            source_pattern = self._build_node_pattern(source_node)
            target_pattern = self._build_node_pattern(target_node)
            relationship = predicate["type"]

            rel_var = f"r{idx}"
            rel_vars.append(rel_var)

            path_patterns.append(
                f"{source_pattern}-[{rel_var}:{relationship}]->{target_pattern}"
            )

        return f"MATCH {', '.join(path_patterns)}", rel_vars

    def _build_property_conditions(self, properties):
        conditions = []
        for key, value in properties.items():
            if value and str(value).strip():
                conditions.append(f"{key}: '{self._sanitize_value(value)}'")
        return conditions

    def _build_node_pattern(self, node):
        node_id = node["node_id"]
        node_type = node["type"]
        if node["id"] and node["id"].strip():
            return f"({node_id}:{node_type} {{id: '{self._sanitize_value(node['id'])}'}})"
        properties = node["properties"]
        if properties and any(v for v in properties.values() if v):
            prop_conditions = self._build_property_conditions(properties)
            if prop_conditions:
                return f"({node_id}:{node_type} {{{', '.join(prop_conditions)}}})"
        return f"({node_id}:{node_type})"

    def _build_where_clause(self, nodes):
        conditions = []

        for node in nodes:
            node_id = node["node_id"]
            properties = node["properties"]

            for key, value in properties.items():
                if value and str(value).strip():
                    # Check if this property is already used in MATCH clause
                    if not (node["id"] and node["id"].strip()):
                        # Only add if not using database ID
                        conditions.append(
                            f"{node_id}.{key} = '{self._sanitize_value(value)}'"
                        )
        if conditions:
            return f"WHERE {' AND '.join(conditions)}"
        return ""

    def _build_return_clause(self, nodes, rel_vars):
        node_ids = [node["node_id"] for node in nodes]
        return_vars = node_ids + (rel_vars or [])
        return f"RETURN {', '.join(return_vars)}"

    def _sanitize_value(self, value):
        if not isinstance(value, str):
            value = str(value)
        # Escape single quotes
        value = value.replace("'", "\\'")
        # Remove newlines and excessive whitespace
        value = " ".join(value.split())

        return value


class CypherExecutor:
    """Runs a Cypher query against Neo4j and summarizes what comes back.

    Not currently wired into the annotation pipeline -- annotation builds and
    returns JSON only. Kept here, next to the converter that produces the query,
    for when execution is switched back on: construct with a Neo4j connection
    and an LLM, then call execute_cypher_query() and summarize_results().
    """

    def __init__(self, neo4j, llm):
        self.neo4j = neo4j
        self.llm = llm

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
                    for rel in ["transcribes_to", "part_of", "transcribed_from", "translates_to"]
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

    def generate_database_summary(self):
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
