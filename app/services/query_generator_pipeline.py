import json
from typing import Any, Dict

import requests


class LLMInterface:
    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the generate method")

class GeminiModel(LLMInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts":[{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1
            }
        }
        response = requests.post(f"{self.api_url}?key={self.api_key}", headers=headers, json=data)
        response.raise_for_status()
        print("----------------Response --------------------------------")
        content = response.json()['candidates'][0]['content']['parts'][0]['text']
    
            # Extract JSON from the code block
        json_content = self._extract_json_from_codeblock(content)
        print(json_content)
        # Parse the extracted JSON
        return json.loads(json_content)

    def _extract_json_from_codeblock(self, content: str) -> str:
        # Find content between ```json and ``` tags
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            # Extract the JSON content and remove the opening ```json tag
            json_content = content[start+7:end].strip()
            return json_content
        else:
            # If no code block is found, return the original content
            return content
    
class QueryProcessor:
    def __init__(self, llm: LLMInterface, schema: str):
        self.llm = llm
        self.schema = schema
    
    def process_query(self, query: str) -> Dict[str, Any]:
        # Parse the query to extract the relevant information from the schema
        # ...
        
        prompt = self._construct_prompt(query)
        
        response = self.llm.generate(prompt)
        
        return response
    
    def _construct_prompt(self, query: str) -> str:
        return f"""
Given the following schema and query, extract the relevant nodes, edges, and relationships that could lead to answering the query:

Schema:
{self.schema}

Query: {query}

Based on the schema and the query, please:
1. Identify the relevant node types involved in the query.
2. Determine the relationships between these nodes that are necessary to answer the query.
3. Construct a possible path through the graph that connects these nodes using the defined relationships.
4. If the query mentions specific IDs (e.g., gene names like IGF3), include these in your response.

Please provide your response in the following JSON format:
{{
    "relevant_nodes": [
        {{"type": "node_type1", "id": "specific_id_if_given_else_empty_string"}},
        {{"type": "node_type2", "id": "specific_id_if_given_else_empty_string"}},
        ...
    ],
    "relevant_relationships": ["relationship1", "relationship2", ...],
    "proposed_path": [
        {{"start_node": {{"type": "node_type1", "id": "specific_id_if_given_else_empty_string"}}, "relationship": "relationship1", "end_node": {{"type": "node_type2", "id": "specific_id_if_given_else_empty_string"}}}},
        ...
    ],
    "explanation": "A brief explanation of how this path relates to the query and how it can be used to find the answer."
}}

Ensure that all nodes and relationships in your response are consistent with the provided schema. Do not introduce any nodes or relationships that are not defined in the schema.
"""

class JsonFormatConverter:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def convert(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._construct_prompt(extracted_info)
        return self.llm.generate(prompt)

    def _construct_prompt(self, extracted_info: Dict[str, Any]) -> str:
        sample_format = """
{
  "nodes": [
    {
      "node_id": "n1",
      "id": "",
      "type": "label",
      "properties": {
        "key": "value"
      }
    },
    {
      "node_id": "n2",
      "id": "",
      "type": "label",
      "properties": {}
    }
    ...
  ],
  "predicates": [
    {
      "type": "predicate",
      "source": "n1",
      "target": "n2"
    }
    ...
  ]
}
"""
        return f"""
Given the following extracted information from a query:

{json.dumps(extracted_info, indent=2)}

Convert this information into the following JSON format:

{sample_format}

Please follow these rules when creating the output:
1. For each node, include node_id, id, type, and properties fields. These are mandatory and should be the only fields for nodes.
2. Generate node_ids by auto-incrementing for each node (n1, n2, n3, ...).
3. If a specific ID is given in the extracted information (e.g., IGF3 for a gene), use it in the 'id' field. Otherwise, leave it as an empty string.
4. Add the label of the node in the type filed
5. Include all relevant properties from the extracted information in the properties field of each node.
6. For predicates (relationships), include type, source, and target fields. The source and target should reference the node_ids.
7. Ensure that the output JSON structure matches the sample format provided.
8. Make sure all relationships in the extracted information are represented as predicates in the output.
9. Make sure all nodes in the relation ship are available in the nodes list
Provide only the resulting JSON as your response, without any additional explanation or commentary.
"""

class QueryExtractionSystem:
    def __init__(self, llm, schema):
        self.schema = schema
        self.query_processor = QueryProcessor(llm, schema)
        self.json_converter = JsonFormatConverter(llm)
        
              
    
    def process_query(self, query):
        extracted_info = self.query_processor.process_query(query)
        converted_json = self.json_converter.convert(extracted_info)
        print("=================Final=================")
        return converted_json


if __name__ == '__main__':
    gemini_api_key = ""
    schema_text = """
    Schema
    Node properties:
    exon {exon_id: STRING, start: STRING, end: STRING, id: STRING, exon_number: STRING, chr: STRING}
    gene {gene_name: STRING, gene_type: STRING, synonyms: LIST}
    transcript {gene_name: STRING}
    protein {source: STRING, id: STRING, source_url: STRING, synonyms: LIST, protein_name: STRING}
    Relationship properties:
    transcribed_to {source: STRING, source_url: STRING}
    transcribed_from {source: STRING, source_url: STRING}
    translates_to {source: STRING, source_url: STRING}
    translation_of {source: STRING, source_url: STRING}
    The relationships:
    (:gene)-[:transcribed_to]->(:transcript)
    (:transcript)-[:transcribed_from]->(:gene)
    (:transcript)-[:translates_to]->(:protein)
    (:protein)-[:translation_of]->(:transcript)
    """
    
     # Example query
    # query = "Give me genes that can produce protein"
    query = "What proteins can we get from IGF3 gene"
    gemini_llm = GeminiModel(gemini_api_key)
    gemini_system = QueryExtractionSystem(gemini_llm, schema_text)
    gemini_result = gemini_system.process_query(query)
    print("Gemini Result:")
    print(json.dumps(gemini_result, indent=2))