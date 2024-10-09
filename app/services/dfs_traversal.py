from collections import defaultdict

def generate_schema_relationship():
    # GET THE SCHEMA FROM annotation
    
    # RETURN schema_relationship
    # refactor the schema rnship to get a schema_relationship
    pass

schema_relationship={
                    "gene": {
                        "transcribed to": "transcript",
                        "coexpressed_with": "gene",
                        "expressed_in": "ontology_term",
                        "genes_pathways": "pathway",
                        "regulates": ["gene","regulatory_region"],
                        "associated with":["enhancer","promoter","super_enhancer"],
                        "tfbs_snp": "snp",
                        "binds_to": "transcription_binding_site",
                        "in_tad_region": "tad",
                    },
                    "transcript": {
                        "transcribed_from": "gene",
                        "translates to": "protein",
                    },
                    "protein": {
                        "translation of": "transcript",
                        "interacts_with": "protein",
                        "go_gene_product": "go",
                    },
                    "ontology_term": {
                        "has_part": "ontology_term",
                        "part_of": "ontology_term",
                        "subclass_of": "ontology_term",
                    },
                    "cl": {
                        "capable_of": "go",
                        "part_of": "uberon",
                        "subclass_of": "cl",
                    },
                    "bto": {
                        "subclass_of": "bto",
                    },
                    "efo": {
                        "subclass_of": "efo",
                    },
                    "uberon": {
                        "subclass_of": "uberon",
                    },
                    "clo": {
                        "subclass_of": "clo",
                    },
                    "go": {
                        "subclass_of": "go",
                    },
                    "pathway": {
                        "parent_pathway_of": "pathway",
                        "child_pathway_of": "pathway",
                    },
                    "snp": {
                        "eqtl_association": "gene",
                        "closest_gene": "gene",
                        "upstream_gene": "gene",
                        "downstream_gene": "gene",
                        "in_gene": "gene",
                        "in_ld_with": "snp",
                        "activity_by_contact": "gene",
                        "chromatin_state": "uberon",
                        "in_dnase_I_hotspot": "uberon",
                        "histone_modification": "uberon",
                    },
                    "chromosome_chain": {
                        "lower_resolution": "chromosome_chain",
                    },
                    "position_entity": {
                        "located_on_chain": "chromosome_chain",
                    },
                    "regulatory_region": {
                        "regulates": "gene",
                    },
                }

 

def build_graph(schema_relationship):
    # schema_relationship = generate_schema_relationship()

    graph = defaultdict(list)
    for node, relationships in schema_relationship.items():
        for rel, target in relationships.items():
            if isinstance(target, list):
                for t in target:
                    graph[node].append((t, rel))
            else:
                graph[node].append((target, rel))
    # store the graph in file
    return graph


def extract_relations_between_nodes_dfs(current, target, path=None, relationships=None, visited=None):
        # read the file 
        # return the graph
        graph = build_graph(schema_relationship)
        if visited is None:
            visited = set()
        if path is None:
            path = []
        if relationships is None:
            relationships = []

        visited.add(current)
        path.append(current)

        if current == target:
            # Combine the nodes and relationships into a single explanatory string
            explanation = " -> ".join(f"{path[i]} -> {relationships[i]}" for i in range(len(relationships)))
            explanation += f" -> {path[-1]}"
            return explanation

        for neighbor, rel in graph[current]:
            if neighbor not in visited:
                explanation = extract_relations_between_nodes_dfs(
                    neighbor, target, path.copy(), relationships + [rel], visited.copy()
                )
                if explanation:
                    return explanation
        return None
        
    
def generate_json_from_schema_and_traversal(schema, prompt_answer, traversal_data=None):
    # Extract node properties from the schema
    node_properties = {entry.split()[0]: entry for entry in schema if entry.split()[0] != 'The'}
    nodes = []
    predicates = []
    node_id_map = {}
    node_id_counter = 1

    def create_node(node_type, prompt_props={}):
        """Creates a node object with valid properties."""
        nonlocal node_id_counter
        node_id = f"n{node_id_counter}"
        node_id_counter += 1

        node_schema_props = {k.split(":")[0]: None for k in node_properties.get(node_type, '').split(", ") if ":" in k}
        properties = {k: prompt_props.get(k) for k in node_schema_props if prompt_props.get(k) is not None}

        node_id_map[node_type] = node_id  # Map node type to node ID

        return {
            "node_id": node_id,
            "id": prompt_props.get('id', ''),
            "type": node_type,
            "properties": properties
        }

    def handle_source_node(source_node_data):
        """Handles the creation of the source node."""
        if source_node_data:
            source_node_type = source_node_data['type']
            if source_node_type in node_properties:
                return create_node(source_node_type, source_node_data['properties'])
            else:
                print(f"Warning: Source node type '{source_node_type}' not found in schema.")
        return None

    def handle_target_node(target_node_data):
        """Handles the creation of the target node."""
        if target_node_data:
            target_node_type = target_node_data['type']
            if target_node_type in node_properties:
                return create_node(target_node_type, target_node_data['properties'])
            else:
                print(f"Warning: Target node type '{target_node_type}' not found in schema.")
        return None

    def process_traversal_data(traversal_data, additional_properties):
        """Processes traversal data to create nodes and predicates."""
        if traversal_data:
            traversal_nodes_relations = traversal_data.split(' -> ')
            for i in range(0, len(traversal_nodes_relations) - 1, 2):
                source_type = traversal_nodes_relations[i]
                relation_type = traversal_nodes_relations[i + 1]
                target_type = traversal_nodes_relations[i + 2]

                if source_type not in node_id_map:
                    print(f"'{source_type}' not found in prompt_answer. Creating dynamically.")
                    new_node = create_node(source_type)
                    nodes.append(new_node)

                if target_type not in node_id_map:
                    print(f"'{target_type}' not found in prompt_answer. Creating dynamically.")
                    prompt_props = {k: v for k, v in additional_properties.items() if target_type in k}
                    new_node = create_node(target_type, prompt_props)
                    nodes.append(new_node)

                predicates.append({
                    "type": relation_type,
                    "source": node_id_map[source_type],
                    "target": node_id_map[target_type]
                })

    # Create the source node
    source_node_data = prompt_answer.get('source_node')
    source_node = handle_source_node(source_node_data)
    if source_node:
        nodes.append(source_node)

    # Handle the case where only the source node is provided
    if not prompt_answer.get('target_node'):
        return {
            "nodes": nodes,
            "predicates": []  # Empty dictionary for predicates
        }

    # If both source and target nodes are provided, continue with normal processing
    additional_properties = {k: v for k, v in prompt_answer.items() if k not in ['source_node', 'target_node']}
    process_traversal_data(traversal_data, additional_properties)

    # Construct the final JSON format
    result_json = {
        "nodes": nodes,
        "predicates": predicates if predicates else []  # Ensure predicates is empty if none are added
    }

    return result_json

