from app.services.dfs_traversal import extract_relations_between_nodes_dfs, generate_json_from_schema_and_traversal

class DFSHandler:

    def __init__(self,llm,schema) -> None:
        self.schema = self.process_schema(schema)
        self.llm= llm

    def process_schema(self, schema):
        """Processes the schema input into a usable format."""
        """accepts only preprossed_schema.txt """
        return [line.strip() for line in schema.split('\n') if line.strip()]

    def intial_prompt(self,query):
        prompt = f"""
                You are an assistant that extracts graph nodes and relationships from user queries based on a given schema.
                {self.schema}
                
                **Allowed Relationships:**

                **Instructions:**
                Given the user query below:
                {query}

                1. Identify and extract the relevant nodes from the query.
                2. Extract the source node and target node from the user query.
                3. Always return the response in the following strict JSON format only:
                {{
                    "source_node": {{
                        "type": "<type_of_source_node>",
                        "id":"",
                        "properties": <source_node_properties>
                    }},
                    "target_node": {{"type": "<type_of_target_node>","id":"", "properties": <target_node_properties>}}  // Include this line only if a target node exists
                }}
                4. also identify any schema properties mention with the exact name mentioned in the schema
                If no id is given place "id" with "" value string don't put null,none or anything   
                """
        extracted_info = self.llm.generate(prompt)
        print("intial prompt", extracted_info)
        return extracted_info

    def json_format(self,query):
        extracted_info = self.intial_prompt(query)
        source_node = extracted_info["source_node"]
        source_node = source_node["type"]
        
        if "target_node" in extracted_info:
            target_node = extracted_info["target_node"]
            target_node = target_node["type"]
            relations = extract_relations_between_nodes_dfs(source_node, target_node)
            print("travesed relations are", relations)
            json_format = generate_json_from_schema_and_traversal(self.schema, extracted_info, relations)
        else:
            json_format = generate_json_from_schema_and_traversal(self.schema, extracted_info)
        return json_format

        