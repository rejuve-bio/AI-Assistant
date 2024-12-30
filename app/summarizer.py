
from collections import defaultdict
import re
import traceback
import json
import tiktoken
import logging
import os
import requests
from dotenv import load_dotenv
from app.prompts.summarizer_prompts import SUMMARY_PROMPT, SUMMARY_PROMPT_BASED_ON_USER_QUERY,SUMMARY_PROMPT_CHUNKING,SUMMARY_PROMPT_CHUNKING_USER_QUERY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
class Graph_Summarizer: 
    '''
    Handles graph-related operations like processing nodes, edges, generating responses ...
    '''
    def __init__(self,llm) -> None:
        self.llm = llm
        self.llm = llm
      
        if self.llm.__class__.__name__ == 'GeminiModel':
            self.max_token=2000
        elif self.llm.__class__.__name__ == 'OpenAIModel':
            self.max_token=100000     
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.kg_service_url = os.getenv('ANNOTATION_SERVICE_URL')

    def clean_and_format_response(self,desc):
        desc = desc.strip()
        desc = re.sub(r'\n\s*\n', '\n', desc)
        desc = re.sub(r'^\s*[\*\-]\s*', '', desc, flags=re.MULTILINE)
        lines = desc.split('\n')

        formatted_lines = []
        for line in lines:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', line)
            for sentence in sentences:
                formatted_lines.append(sentence + '\n')
        formatted_desc = ' '.join(formatted_lines).strip()
        return formatted_desc


    def group_edges_by_source(self,edges):
        """Group edges by source_node."""
        grouped_edges = defaultdict(list)
        for edge in edges:
            source_node_id = edge["source"].split(' ')[-1]  # Extract ID
            grouped_edges[source_node_id].append(edge)
        return grouped_edges

    def generate_node_description(self,node):
        """Generate a description for a node with available attributes."""
        desc_parts = []

        for key, value in node.items():
            # Attempt to parse JSON-like strings into lists
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, list):
                        # Limit to top 3 items
                        top_items = parsed_value[:3]
                        if top_items:
                            desc_parts.append(f"{key.capitalize()}: {', '.join(top_items)}")
                        continue  # Move to the next attribute after processing
                except json.JSONDecodeError:
                    pass  # If not a JSON string, treat it as a regular string

            # For non-list attributes, simply add them to the description
            desc_parts.append(f"{key.capitalize()}: {value}")
        return " | ".join(desc_parts)

    def generate_grouped_descriptions(self,edges, nodes,batch_size=50):
        grouped_edges = self.group_edges_by_source(edges)
        descriptions = []

        # Process each source node and its related target nodes
        for source_node_id, related_edges in grouped_edges.items():
            source_node = nodes.get(source_node_id, {})
            source_desc = self.generate_node_description(source_node)

            # Collect descriptions for all target nodes linked to this source node
            target_descriptions = []
            for edge in related_edges:
                target_node_id = edge["target"].split(' ')[-1]
                target_node = nodes.get(target_node_id, {})
                target_desc = self.generate_node_description(target_node)

                # Add the relationship and target node description
                label = edge["label"]
                target_descriptions.append(f"{label} -> Target Node ({edge['target']}): {target_desc}")

            # Combine the source node description with all target node descriptions
            source_and_targets = (f"Source Node ({source_node_id}): {source_desc}\n" +
                                "\n".join(target_descriptions))
            descriptions.append(source_and_targets)

            # If batch processing is required, we can break or yield after each batch
            # if len(descriptions) >= batch_size:
            #   break   Process the next batch in another iteration

        return descriptions

    def nodes_description(self,nodes):
        nodes_descriptions = []
        for source_node_id in nodes:
            source_node = nodes.get(source_node_id, {})
            source_desc = self.generate_node_description(source_node)
            nodes_descriptions.append(source_desc)
        return nodes_descriptions
    
    def num_tokens_from_string(self, encoding_name: str):
        """Calculates the number of tokens in each description and groups them into batches under a token limit."""
        encoding = tiktoken.get_encoding(encoding_name)
        accumulated_tokens = 0
        grouped_batched_descriptions = []
        self.current_batch = []  
        for i, desc in enumerate(self.description):          
            desc_tokens = len(encoding.encode(desc))
            if accumulated_tokens + desc_tokens <= self.max_token:
                self.current_batch.append(desc)
                accumulated_tokens += desc_tokens
            else:
                grouped_batched_descriptions.append(self.current_batch)
                self.current_batch = [desc]
                accumulated_tokens = desc_tokens  
        if self.current_batch:
            grouped_batched_descriptions.append(self.current_batch)         
        return grouped_batched_descriptions


    
    def graph_description(self,graph, limited_nodes = 100):
       
        limited_node_ids = set()
        for i in range(min(limited_nodes, len(graph['nodes']))):
            limited_node_ids.add(graph['nodes'][i]['data']['id'])

        limited_nodes_data = [node for node in graph['nodes'] if node['data']['id'] in limited_node_ids]
        limited_edges_data = []
        for edge in graph['edges']:
            if edge['data']['source'] in limited_node_ids and edge['data']['target'] in limited_node_ids:
                limited_edges_data.append(edge)

        limited_graph = {
            "nodes": limited_nodes_data,
            "edges": limited_edges_data
        }
        nodes = {node['data']['id']: node['data'] for node in limited_graph['nodes']}

        if len(limited_graph['edges']) > 0:
            edges = [{'source': edge['data']['source'],
                    'target': edge['data']['target'],
                    'label': edge['data']['label']} for edge in limited_graph['edges']]
    
            self.description = self.generate_grouped_descriptions(edges, nodes, batch_size=10)
            self.descriptions = self.num_tokens_from_string("cl100k_base")
        else:
            self.descriptions = self.nodes_description(nodes)
        
        return self.descriptions


    def annotate_by_id(self,query, graph_id, token):
        logger.info("querying annotation by graph id...")
        
        try:
            logger.debug(f"Sending request to {self.kg_service_url}")
            response = requests.get(
                "api url",
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            json_response = response.json()
          
            response =  {
                        "annotation_id": response.get("annotation_id", []),
                        "summary": response.get("summary", []),
                    }
            return response
        except:
            traceback.print_exc()


    def get_graph_info(self, graph_id, token):
        logger.info("querying the graph...")
        
        try:
            logger.debug(f"Sending request to {self.kg_service_url}")
            response = requests.get(
                self.kg_service_url+'/annotation/'+graph_id,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            json_response = response.json()
          
            graph =  {
                        "nodes": graph.get("nodes", []),
                        "edges": graph.get("edges", []),
                        "node_count": graph.get("node_count",[]),
                        "edge_count": graph.get("edge_count",[]),
                    }
            return graph
        except:
            traceback.print_exc()

    def summary(self,graph=None,user_query=None,graph_id=None, token = None):

        try:
            if graph_id:
                summary = self.get_graph_info(graph_id, token)
                return summary

            if graph:
                graph = self.graph_description(graph)

                prev_summery=[]
                for i, batch in enumerate(self.descriptions):  
                    if prev_summery:
                        if user_query:
                            prompt = SUMMARY_PROMPT_CHUNKING_USER_QUERY.format(description=batch,user_query=user_query,prev_summery=prev_summery)
                        else:
                            prompt = SUMMARY_PROMPT_CHUNKING.format(description=batch,prev_summery=prev_summery)
                    else:
                        if user_query:
                            prompt = SUMMARY_PROMPT_BASED_ON_USER_QUERY.format(description=batch,user_query=user_query)
                            print("prompt", prompt)
                        else:
                            prompt = SUMMARY_PROMPT.format(description=batch)
                            print("prompt", prompt)

                    response = self.llm.generate(prompt)
                    prev_summery = [response]  
                    return response
                # cleaned_desc = self.clean_and_format_response(prev_summery)
                # return cleaned_desc
        except:
            traceback.print_exc()
   

              