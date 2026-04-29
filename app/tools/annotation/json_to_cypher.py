import logging

logger = logging.getLogger(__name__)


class JsonToCypherConverter:
    """
    Converts validated JSON queries to executable Cypher queries.
    """

    def __init__(self):
        self.supported_node_types = {"gene", "transcript", "exon"}
        self.supported_relationships = {
            "transcribed_to",
            "transcribed_from",
            "includes",
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

    def _build_node_pattern(self, node):
        node_id = node["node_id"]
        node_type = node["type"]

        # Handle database ID vs properties
        if node["id"] and node["id"].strip():
            # Database ID provided
            return (
                f"({node_id}:{node_type} {{id: '{self._sanitize_value(node['id'])}'}})"
            )
        else:
            # No database ID, use properties if any
            properties = node["properties"]
            if properties and any(v for v in properties.values() if v):
                # Has properties, build property conditions
                prop_conditions = []
                for key, value in properties.items():
                    if value and str(value).strip():
                        prop_conditions.append(
                            f"{key}: '{self._sanitize_value(value)}'"
                        )

                if prop_conditions:
                    return f"({node_id}:{node_type} {{{', '.join(prop_conditions)}}})"

            # No properties or all empty, just match by type
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
