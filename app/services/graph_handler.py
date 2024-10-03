# Handles graph-related operations like processing nodes, edges, generating responses ...
import json
import re
import traceback
import os
import requests
from llm_handler import Gemini,Open_AI_Model
from helpers import schema_description, sample_json_format,schema_relationship
from summarizer import Graph_Summarizer
from ai_assistant import QueryHandler
from collections import defaultdict
import json

ANNOTATION_URL = os.getenv("ANNOTATION_URL")

class Annotation_service_GraphHandler(QueryHandler):

    def __init__(self, query):
        super().__init__(query)
        self.llm = Open_AI_Model() 
        self.schema_description = schema_description
        self.sample_json_format = sample_json_format

    def extract_nodes_and_edges(self):
        try:       
            prompt = f"""
            You are an assistant that extracts graph nodes and relationships from user queries based on a given schema.

            {schema_description}

            **Allowed Relationships:**
            {schema_relationship}

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
            """
            nodes = self.llm(prompt)
            match = re.search(r'```json\n(\{.+\})\n```', nodes, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else nodes
            nodes = json.loads(cleaned_json)
            print(".......extracting source and target nodes........")
            print(nodes)
            return nodes        
        except Exception as e:
            traceback.print_exc()
            print(f"Error in extract_nodes_and_edges: {str(e)}")

    def graph(self):
        graph = defaultdict(list)
        for node, relationships in schema_relationship.items():
            for rel, target in relationships.items():
                if isinstance(target, list):
                    for t in target:
                        graph[node].append((t, rel))
                else:
                    graph[node].append((target, rel))
        return graph
    
    def dfs_traverse(self,graph, current, target, path=None, relationships=None, visited=None):
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
                explanation = self.dfs_traverse(
                    graph, neighbor, target, path.copy(), relationships + [rel], visited.copy()
                )
                if explanation:
                    return explanation
        return None

    def construct_json(self,source_node,target_node=None,relations=None):
        try:
            prompt = f"""
                The user question is:
                {query}
                Given the following extracted information from a query:

                {source_node}
                {target_node}
                {relations}

                Convert this information into the following JSON format:

                {sample_json_format}

                Please follow these rules when creating the output:
                1. For each node, include node_id, id, type, and properties fields. These are mandatory and should be the only fields for nodes.
                2. Generate node_ids by auto-incrementing for each node (n1, n2, n3, ...).
                3. If a specific ID is given in the extracted information, use it in the 'id' field. Otherwise, leave it as empty string.
                4. Add the label of the node in the type filed
                5. Include all relevant properties from the extracted information in the properties field of each node.
                6. For predicates (relationships), include type, source, and target fields. The source and target should reference the node_ids.
                7. Ensure that the output JSON structure matches the sample format provided.
                8. Make sure all relationships in the extracted information are represented as predicates in the output.
                9. Make sure all nodes in the relation ship are available in the nodes list
                Provide only the resulting JSON as your response, without any additional explanation or commentary.
                """
            json_format = self.llm(prompt)

            match = re.search(r'```json\n(\{.+\})\n```', json_format, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else None
            # json_format = json.loads(cleaned_json)
            print(".......finished json format query........")
            print(json_format)
            return cleaned_json
        except:
            traceback.print_exc()

    def final_round(self):
        nodes = self.extract_nodes_and_edges()
        graph = self.graph()
        source_node = nodes["source_node"]
        source_type = source_node["type"]

        if "target_node" not in nodes:
            # Only source node is present, construct the JSON with just the source node
            json_response = self.construct_json(source_node=source_node)
        else:
            # Both source and target nodes are present
            target_node = nodes["target_node"]
            target_type = target_node["type"]
            relations = self.dfs_traverse(graph, source_type, target_type)
            print(".......finished traversing a relationship........")
            print(relations)
            # Construct the JSON format including both nodes and their relationship
            json_response = self.construct_json(source_node=source_node, target_node=target_node, relations=relations)
        return json_response


    def query_annotation_service(self):
        requested_data = self.final_round()
        # json_query = {"requests":requested_data}
        # print(json_query)
        # headers = {'Content-Type': 'application/json'}
        # response = requests.post(ANNOTATION_URL, headers=headers, data=json.dumps(json_query))
        # if response.status_code == 200:
        #     self.graph = response.content
        #     print("........annotation returned succefully...........")
        #     return self.graph
        # else:
        #     print(f"Failed with status code {response.status_code}: {response.text}")

    def summarizer_prompt(self):
        try:
            from sample import annotation_return, query,json_format
            self.graph = annotation_return
            self.query = query
            summary = Graph_Summarizer()
            graph_summary = summary.open_ai_summarizer(self.graph,query,json_format)
            print(graph_summary)
            return graph_summary#, self.graph
        except Exception as e:
            print(f"Error in summarizer_prompt: {str(e)}")
            return None

 

# a = Annotation_service_GraphHandler('')
# a.summarizer_prompt()








# Example query
query = "Show me all genes stored in the database."
query = "Retrieve a list of all transcripts available."
query = "Can you provide a summary of all proteins in the database?"
query = "List all enhancers in the knowledge base" 
query = "Display all enhancers that are recorded in the system."

#* Specific node with id
query = "What properties does the gene ensg00000232448 has?"
# query = "Give me the properties of the transcript enst00000437504."
# query = "What are the attributes of the protein p78504?"
# query = "Fetch the information associated with the exon ense00003875467."
# query = "What detail is available for the enhancer chr1_203097061_203097400_grch38?"

#* Specific nodes with a property
# query = "Which genes are classified as protein coding?"
# query = "List all transcripts that are of type lncRNA."
# query = "Show me exons located on chromosome 20."
# query = "Which proteins have accessions A8K4X5?"

#* Single edge type (2 nodes - 1 edge)
# query = "List out exons found in enst00000691353 transcript"
# query = "Give me the enhancers that have association with the ensg00000227906 gene"
# query = "A transcript with the ID enst00000381261, which encodes a protein, has undergone changes. Based on this transcript, what is the resulting protien?"
# query = "What transcripts belong to the gene ensg00000232448?"


#* Two different edges (3 nodes - 2 edges)
# query = "What proteins can we get from the gene ensg00000132639?"
# query = "Which promoter is involved in producing transcript enst00000658520?"
# query = "Can you provide a list of proteins synthesized by genes on chromosome x?"
# query = "What is the promoter involved in the formation of transcript enst00000658520?" 
# query = "Which proteins are translated from transcripts that include exons located in chr20:1000000-1050000?"
# query = "What proteins can we get from ensg0043403 gene?"
# query = "what transcript can we get from a protein with accessions ABK453L and D3DW15?"
# query = "What genes on chromosome chrx, have transcripts that include exons and give me the protein they produce?"
# query = "Give me all enhancers associated with ensg00000132639 gene"

#* Three different eges (4 nodes - edges)
# query = "What enhancers are involved in the formation of the protein p78504?"

a = Annotation_service_GraphHandler(query)
a.final_round()

