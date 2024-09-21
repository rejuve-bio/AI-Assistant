
# Group edges by source_node and then process each group
from collections import defaultdict
import re
from llm_handler import Gemini


class Graph_Summarizer:
    
    def __init__(self) -> None:
        self.llm = Gemini()

    def clean_and_format_response(self,desc):
        """Cleans the response from the Gemini model and formats it with multiple lines."""
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
            source_node_id = edge["source_node"].split(' ')[-1]  # Extract ID
            grouped_edges[source_node_id].append(edge)
        return grouped_edges

    def generate_node_description(self,node):
        """Generate a description for a node with available attributes."""
        desc_parts = []
        attributes = {
            'type': 'type',
            'protein_name': 'protein_name',
            'synonyms': 'synonyms',
            'accessions': 'accessions',
            'source': 'source',
            'gene_name': 'gene_name',
            'gene_type': 'gene_type',
            'start': 'start',
            'end': 'end',
            'chr': 'chr',
            'transcript_id': 'transcript_id',
            'transcript_name': 'transcript_name',
            'transcript_type': 'transcript_type'
        }

        for key, label in attributes.items():
            if key in node:
                if key in ['synonyms', 'accessions']:
                    # Slice to get top 3 elements if available
                    top_items = node[key][:3]
                    if top_items:
                        desc_parts.append(f"{label}: {', '.join(top_items)}")
                else:
                    desc_parts.append(f"{label}: {node[key]}")

        return " | ".join(desc_parts)

    def generate_grouped_descriptions(self,edges, nodes, batch_size=10):
        grouped_edges = self.group_edges_by_source(edges)
        descriptions = []

        # Process each source node and its related target nodes
        for source_node_id, related_edges in grouped_edges.items():
            source_node = nodes.get(source_node_id, {})
            source_desc = self.generate_node_description(source_node)

            # Collect descriptions for all target nodes linked to this source node
            target_descriptions = []
            for edge in related_edges:
                target_node_id = edge["target_node"].split(' ')[-1]
                target_node = nodes.get(target_node_id, {})
                target_desc = self.generate_node_description(target_node)

                # Add the relationship and target node description
                label = edge["label"]
                target_descriptions.append(f"{label} -> Target Node ({edge['target_node']}): {target_desc}")

            # Combine the source node description with all target node descriptions
            source_and_targets = (f"Source Node ({source_node_id}): {source_desc}\n" +
                                "\n".join(target_descriptions))
            descriptions.append(source_and_targets)

            # If batch processing is required, we can break or yield after each batch
            if len(descriptions) >= batch_size:
                break  # Process the next batch in another iteration

        return descriptions
    
    def graph_description(self,graph):
        nodes = {node['data']['id']: node['data'] for node in graph['nodes']}
        edges = [{'source_node': edge['data']['source_node'], 'target_node': edge['data']['target_node'], 'label': edge['data']['label']} for edge in graph['edges']]
        self.description = self.generate_grouped_descriptions(edges, nodes, batch_size=10)

    def summarizer_prompt(self,user_query,graph):
        # make the description to the llm in chunk not once

        self.graph_description(graph)
        prompt = (
        f"you are a biology expert assistant"
        f"Given a user query question {user_query}"
        f"Given the following data, please summarize the key points clearly:\n"
        f"Data:\n{self.description}\n"
        f"Instructions: Please provide a clear and concise summary of the following data, highlighting the core information and the associations related with the suer query.\n"
        f"Include the key and must details related with the user question only."
        f"Summary:")

        response = self.llm(prompt)
        cleaned_desc = self.clean_and_format_response(response)
        return cleaned_desc
    