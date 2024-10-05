import json
from typing import Any, Dict

from app.services.llm_models import LLMInterface


class LLMPromptHandler:
    def __init__(self, llm: LLMInterface, schema: str):
        self.llm = llm
        self.schema = schema
    
    def process_initial_query(self, query: str) -> Dict[str, Any]:
        prompt = self._construct_initial_prompt(query)
        return self.llm.generate(prompt)
    
    def convert_to_json_format(self, extracted_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        prompt = self._construct_json_conversion_prompt(extracted_info, query)
        return self.llm.generate(prompt)
    
    def process_kg_response(self, original_query: str, json_query: Dict[str, Any], kg_response: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._construct_kg_response_prompt(original_query, json_query, kg_response)
        return self.llm.generate(prompt)
    
    
    def _construct_initial_prompt(self, query: str) -> str:
       return f"""
Schema:
{self.schema}

Query: {query}

### TASK:
Let's think step by step to extract the relevant information needed to answer the query.

1. Identify relevant nodes and their properties based on the schema.
2. Identify necessary relationships between the nodes.
3. Construct a path using relationships from the schema (connect from one node to the other to achive the query).
4. Include any specific IDs mentioned in the query.
5. Double check if the direction is reveresed it is strict (start)-[predicate]->(end) so Never change this direction

### STRICT RULES:
- Use only node types and relationships specified in the schema.
- Do not invent or reverse relationships.
- Ensure all nodes in relationships are included in the list.
- Only add property keys if mentioned int the query if not don't add it
- Never grab the property from the schema and include in the nodes

### RESPONSE FORMAT:
Please provide your response in the following format:

**Relevant Nodes:**
- Node Type: `node_type1`
  - ID: `specific_id_or_empty_string`
  - Properties: 
    - Property:

- Node Type: `node_type2`
  - ID: `specific_id_or_empty_string`
  - Properties: 

**Relevant Relationships:** # only if a connection of path is needed to acheive the query
For each relationship, please specify the details as follows:

1. **Relationship 1:**
   - **Start Node:**
     - Type: `node_type1`
     - ID: `id_or_empty_string`
   - **Predicate:** `relationship_from_schema`
   - **End Node:**
     - Type: `node_type2`
     - ID: `id_or_empty_string`

2. **Relationship 2:**
   - **Start Node:**
     - Type: `node_type3`
     - ID: `id_or_empty_string`
   - **Predicate:** `another_relationship_from_schema`
   - **End Node:**
     - Type: `node_type4`
     - ID: `id_or_empty_string`

(Continue for all relevant relationships)
"""
    
    
    def _construct_json_conversion_prompt(self, extracted_info: Dict[str, Any], query: str) -> str:
        sample_format = """
{
  "nodes": [
    {
      "node_id": "label_1",
      "id": "id_or_empty_string",
      "type": "label",
      "properties": {
        "key": "value"
      }
    },
    {
      "node_id": "label_2",
      "id": "id_or_empty_string",
      "type": "label",
      "properties": {}
    }
    ...
  ],
  "predicates": [
    {
      "type": "predicate",
      "source": "label_1",
      "target": "label_2"
    }
    ...
  ]
}
"""
        return f"""
### TASK:
Convert the extracted information into the target JSON format based on the schema.

Query: {query}
Extracted information:
{json.dumps(extracted_info, indent=2)}

Schema:
{self.schema}

### Conversion rules:
1. Generate unique `node_ids` for each node in the format "label_X".
2. Include **ALL nodes** mentioned in the extracted information in the "nodes" list.
3. Ensure all nodes that appear in the predicates (relationships) are also included in the "nodes" list, even if they were not explicitly extracted.
4. Ensure all predicates (relationships) **exactly match** those defined in the schema.
5. **Do NOT add** any information not present in the extracted information or schema.

### Response format (JSON):
{sample_format}
"""   

    def _construct_kg_response_prompt(self, original_query: str, json_query: Dict[str, Any], kg_response: Dict[str, Any]) -> str:
        return f"""
Original Query: {original_query}

JSON Query sent to Knowledge Graph:
{json.dumps(json_query, indent=2)}

Knowledge Graph Response:
{json.dumps(kg_response, indent=2)}

Task: Based on the original query and the knowledge graph response, provide a concise and informative answer.
Follow these guidelines:
1. Directly address the original query.
2. Use only the information provided in the knowledge graph response.
3. If the knowledge graph response doesn't contain enough information to fully answer the query, state this clearly.
4. Format the response in a clear, easy-to-read manner.
5. If appropriate, use bullet points or numbered lists for clarity.

Your response:
"""