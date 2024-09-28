# Handles graph-related operations like processing nodes, edges, generating responses ...
import json
import re
import traceback
import os
import requests
from llm_handler import Gemini
from helpers import schema_description, sample_json_format,nodes_edges_prompt,json_constructor_prompt,relationship_extractor,sample_graph,schema_relationship
from summarizer import Graph_Summarizer
from ai_assistant import QueryHandler
from collections import defaultdict
import json

ANNOTATION_URL = os.getenv("ANNOTATION_URL")

class Annotation_service_GraphHandler(QueryHandler):

    def __init__(self, query):
        super().__init__(query) 
        self.schema_description = schema_description
        self.sample_json_format = sample_json_format

    def extract_nodes_and_edges(self):
        try:       
            # self.nodes_and_edges = self.llm(nodes_edges_prompt)
            self.nodes = self.llm(relationship_extractor)
            match = re.search(r'```json\n(\{.+\})\n```', self.nodes, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else None
            self.nodes = json.loads(cleaned_json)
            return self.nodes        
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
            json_format = self.llm(json_constructor_prompt)
            match = re.search(r'```json\n(\{.+\})\n```', json_format, re.DOTALL | re.IGNORECASE)
            cleaned_json = match.group(1) if match else None
            self.json_format = json.loads(cleaned_json)
            print(".......finished json format query........")
            return self.query_annotation_service
        except:
            traceback.print_exc()

    def final_round(self):
        nodes = self.extract_nodes_and_edges()
        graph = self.graph()
        source_node = nodes['source_node']
        source_type = nodes['type']
        
        if 'target_node' not in source_node:
            # Only source node is present, construct the JSON with just the source node
            json_response = self.construct_json(source_node=source_node)
        else:
            # Both source and target nodes are present
            target_node = nodes['target_node']
            target_type = target_node['type']
            relations = self.dfs_traverse(graph, source_type, target_type)
            print(relations)
            # Construct the JSON format including both nodes and their relationship
            json_response = self.construct_json(source_node=source_node, target_node=target_node, relations=relations)
        
        return json_response


    def query_annotation_service(self):
        headers = {'Content-Type': 'application/json'}

        response = requests.post(ANNOTATION_URL, headers=headers, data=json.dumps(self.query_json_format))
        if response.status_code == 200:
            self.graph = response.content
            print("........annotation returned succefully...........")
            return self.graph
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
   
    # query_and_refactoring_json_format(self):
    #     try:
    #         self.query_annotation_service()
    #     except Exception as e:
            # if there is schema issue do this all over again or ... ?
            # print(f"Error in query_and_refactoring_json_format: {str(e)}")

    def summarizer_prompt(self):
        try:
            from app.services.sample import annotation_return, query
            self.graph = annotation_return
            self.query = query
            summary = Graph_Summarizer(self.query, self.query_json_format)
            graph_summary = summary.summarizer(self.graph)
            return graph_summary, self.graph
        except Exception as e:
            print(f"Error in summarizer_prompt: {str(e)}")
            return None

    def generate_annotation_result(self):
        try:
            query = self.final_round()
            return query
        except Exception:
            traceback.print_exc()


query ="What is gene ENSG00000237491 transcribed_to?"

a = Annotation_service_GraphHandler(query)
print(a.generate_annotation_result())

