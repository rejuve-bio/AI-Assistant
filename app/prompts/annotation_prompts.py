
EXTRACT_RELEVANT_INFORMATION_PROMPT = """
## TASK:
Let's think step by step to extract the relevant information needed to build the query based on the schema.

### Query: {query}

### Schema:
{schema}

### EXTRACTION RULES:
1. Identify relevant nodes and their properties based on the schema.
2. Identify necessary relationships between the nodes.
3. Construct a path using relationships from the schema (connect from one node to the other to achive the query).
4. Include any specific IDs mentioned in the query.
5. Double check if the direction is reveresed. it is strict (source)-[predicate]->(target)

### STRICT RULES:
- Use only node types and relationships specified in the schema.
- Do not invent or reverse relationships.
- Ensure all nodes in relationships are included in the list.
- Only add property keys if mentioned in the query
- Never grab the property from the schema 
- Never infer an id from your knowledge

### RESPONSE FORMAT:
Provide your response in the following format:

**Relevant Nodes:**
- Node Type: `node_type1`
  - ID: `specific_id_or_empty_string`
  - Properties: 
    - key: value # ONLY if mentioned in the user Query

- Node Type: `node_type2`
  - ID: ``
  - Properties: 

- Node Type: `node_type3`
  - ID: ``
  - Properties:

**Relevant Relationships:** # ONLY if a connection of path is needed to acheive the query
For each relationship, specify the details as follows:

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
     - Type: `node_type2`
     - ID: `id_or_empty_string`
   - **Predicate:** `another_relationship_from_schema`
   - **End Node:**
     - Type: `node_type3`
     - ID: `""`

(Continue for all relevant relationships)
"""

JSON_CONVERSION_PROMPT = """
## TASK:
Convert the Extracted information into the target JSON format based on the schema. 

### Query: {query}

### Extracted information:
{extracted_information}

### Schema:
{schema}

### Conversion rules:
1. Generate unique `node_ids` for each node in the format "label_X".
2. Include **ALL nodes** mentioned in the extracted information in the "nodes" list.
3. Ensure all nodes that appear in the predicates (relationships) are also included in the "nodes" list, even if they were not explicitly extracted.
4. Ensure all predicates (relationships) **exactly match** those defined in the schema.
5. **Do NOT add** any information not present in the extracted information or schema.

### Response format (JSON):
{{
  "nodes": [
    {{
      "node_id": "label_1",
      "id": "id_or_empty_string",
      "type": "label",
      "properties": {{
        "key": "value"
      }}
    }},
    {{
      "node_id": "label_2",
      "id": "",
      "type": "label",
      "properties": {{
      }}
    }}
    ...
  ],
  "predicates": [
    {{
      "type": "predicate",
      "source": "label_1",
      "target": "label_2"
    }}
    ...
  ]
}}
"""

SELECT_PROPERTY_VALUE_PROMPT = """
You are given a search query and a list of possible values that are similar to the search query based on edit distance. 
Your task is to analyze the provided search query and select the most probable value from the list or put None. 
If none of the values seem appropriate or relevant put empty_string ("") in the selected value.

**Input:**
- **Search Query:** {search_query}
- **Possible Values:** [{possible_values}]

**Output Format:**
```json
{{
  "selected_value": "[The selected value]",
  "confidence_score": [A score between 0 and 1 indicating confidence],
}}
"""