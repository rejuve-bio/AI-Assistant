from config.dfs_json_format import schema, nodes_template,predicates_template
from app.prompts.dfs_prompt import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class DFSHandler:

    def __init__(self,llm,schema) -> None:
        self.schema = self.process_schema(schema)
        self.llm= llm

    def process_schema(self, schema):
        """Processes the schema input into a usable format."""
        """accepts only preprossed_schema.txt """
        return [line.strip() for line in schema.split('\n') if line.strip()]# making easy to work with schema

    def intial_prompt(self,query):
        prompt = DFS_EXTRACT_NODES.format(schema=self.schema,query=query)
        extracted_info = self.llm.generate(prompt)
        logger.info(f"intial prompt for the given {query} is {extracted_info}")
        return extracted_info ## setting the scene

    def json_format(self,query):
        extracted_info = self.intial_prompt(query)
        source_node = extracted_info["source_node"]
        source_node = source_node["type"]
        
        if "target_node" in extracted_info:
            target_node = extracted_info["target_node"]
            target_node = target_node["type"]
            relations = self.extract_relations_between_nodes_dfs(source_node, target_node)
            logger.info(f"travesed relations in user query are {relations}")
            json_format = self.generate_json_from_schema_and_json_query(extracted_info, relations)
        else:
            json_format = self.generate_json_from_schema_and_json_query(extracted_info)
        return json_format 
    # to make sure that the explorer can recored its findings


    # def extract_relations_between_nodes_dfs(self,current, target, path=None, relationships=None, visited=None):
    #     from app import schema_handler
    #     # read the file 
    #     # return the graph
    #     graph = schema_handler.schema_graph
    #     if visited is None:
    #         visited = set()
    #     if path is None:
    #         path = []
    #     if relationships is None:
    #         relationships = []

    #     visited.add(current)
    #     path.append(current)

    #     if current == target:
    #         # Combine the nodes and relationships into a single explanatory string
    #         explanation = " -> ".join(f"{path[i]} -> {relationships[i]}" for i in range(len(relationships)))
    #         explanation += f" -> {path[-1]}"
    #         return explanation
 
    #     for neighbor, rel in graph[current]:
    #         if neighbor not in visited:
    #             explanation = self.extract_relations_between_nodes_dfs(
    #                 neighbor, target, path.copy(), relationships + [rel], visited.copy()
    #             )
    #             if explanation:
    #                 return explanation
    #     return None
        ## the depth first search algorithm where we go through the graph, to make sure the query makes sense
        ## maybe the breadth first search can be more effective
            
    

    def extract_relations_between_nodes_bfs(self, start, target):
        from app import schema_handler
        from collections import deque
        graph = schema_handler.schema_graph

        visited = set()
        queue = deque()
        # Each queue element is a tuple: (current_node, path_taken, relationships_collected)
        queue.append((start, [start], []))
        visited.add(start)

        while queue:
            current_node, path, relationships = queue.popleft()

            if current_node == target:
                # Construct the explanation string
                explanation = " -> ".join(f"{path[i]} -> {relationships[i]}" for i in range(len(relationships)))
                explanation += f" -> {path[-1]}"
                return explanation

            # Explore all neighbors of the current node
            # Adjust the next line based on how your graph stores neighbors/relationships
            for neighbor, rel in graph[current_node]:  # Assumes graph structure provides (neighbor, relationship)
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_rels = relationships + [rel]
                    queue.append((neighbor, new_path, new_rels))

        return None  # If no path exists
    
    def generate_json_from_schema_and_json_query(self,prompt_answer, traversal_data=None, schema=schema, nodes_template=nodes_template, predicates_template=predicates_template):
        # Extract node properties from the schema
        node_properties = {}
        nodes = []
        predicates = []
        node_id_map = {}
        node_id_counter = 1

        # Parsing schema for node types and their properties
        for schema_item in schema:
            node_type, props = schema_item.split(" {")
            props = props.rstrip("}")
            node_properties[node_type.strip()] = props

        def extract_additional_properties(node_type, prompt_answer):
            """Extracts additional properties based on node type from prompt_answer."""
            additional_props = {}
            schema_keys = [prop.split(":")[0] for prop in node_properties.get(node_type, '').split(", ")]

            # Check if any prompt_answer items map to schema properties for the given node type
            for key, value in prompt_answer.items():
                if key in schema_keys:
                    additional_props[key] = value
            return additional_props

        def create_node(node_data):
            """Creates a node object dynamically based on the template provided."""
            nonlocal node_id_counter
            node_id = node_data.get('id', '')  # Set to '' if id is empty

            # Extract properties for the node from schema
            node_type = node_data.get('type', '')
            prompt_props = node_data.get('properties', {})
            schema_props = {
                k.split(":")[0]: None for k in node_properties.get(node_type, '').split(", ") if ":" in k
            }

            # Add any additional properties passed in prompt_answer to this node
            additional_props = extract_additional_properties(node_type, prompt_answer)
            properties = {k: prompt_props.get(k, additional_props.get(k)) for k in schema_props if prompt_props.get(k) or additional_props.get(k)}

            # Create a new node based on the nodes template
            new_node = {}
            for key in nodes_template.keys():
                if key == "properties":
                    new_node[key] = properties  # Set properties dynamically
                elif key == "node_id":
                    new_node[key] = f"n{node_id_counter}"  # Generate and assign the node_id
                    node_id_counter += 1
                elif key == "id":
                    new_node[key] = node_id  # Assign the id (or '')
                else:
                    # Set values for 'type' directly from the node data
                    new_node[key] = node_data.get(key, nodes_template[key])

            # Store the node ID in the mapping using the type as the key
            node_id_map[node_type] = new_node['node_id']  # Map node type to node ID
            return new_node 
        ## where the cypher query is built from the DFS search above

        def handle_node(node_data):
            """Handles the creation of a node based on the prompt answer and schema."""
            if node_data:
                node_type = node_data.get('type', '')
                if node_type in node_properties:
                    return create_node(node_data)
                else:
                    print(f"Warning: Node type '{node_type}' not found in schema.")
            return None

        def process_traversal_data(traversal_data):
            """Processes traversal data to create nodes and predicates dynamically."""
            if traversal_data:
                traversal_nodes_relations = traversal_data.split(' -> ')
                for i in range(0, len(traversal_nodes_relations) - 1, 2):
                    source_type = traversal_nodes_relations[i]
                    relation_type = traversal_nodes_relations[i + 1]
                    target_type = traversal_nodes_relations[i + 2]

                    # Create source node if not already created
                    if source_type not in node_id_map:
                        source_node = create_node({"type": source_type})
                        nodes.append(source_node)

                    # Create target node if not already created
                    if target_type not in node_id_map:
                        # Extract properties relevant to the target node
                        target_props = {k: v for k, v in prompt_answer.items() if target_type in k}
                        target_node = create_node({"type": target_type, "properties": target_props})
                        nodes.append(target_node)

                    # Dynamically add predicate relationship based on the predicates template
                    new_predicate = {}
                    for key in predicates_template.keys():
                        if key == "source":
                            new_predicate[key] = node_id_map[source_type]  # Assign the source node_id
                        elif key == "target":
                            new_predicate[key] = node_id_map[target_type]  # Assign the target node_id
                        else:
                            new_predicate[key] = relation_type if key == "type" else predicates_template[key]
                    predicates.append(new_predicate)

        # Create the source node
        source_node_data = prompt_answer.get('source_node')
        source_node = handle_node(source_node_data)
        if source_node:
            nodes.append(source_node)

        # Handle the target node if provided
        target_node_data = prompt_answer.get('target_node')
        if target_node_data:
            target_node = handle_node(target_node_data)
            if target_node:
                nodes.append(target_node)

        # Process the traversal data
        process_traversal_data(traversal_data)

        # Construct the final JSON format
        result_json = {
            "nodes": nodes,
            "predicates": predicates if predicates else []  # Ensure predicates is empty if none are added
        }

        return result_json

            