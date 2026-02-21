EXTRACT_RELEVANT_INFORMATION_PROMPT = """
## TASK:
Let's think step by step to extract the relevant information needed to build the query based on the schema.

### Query: {query}

### Schema:
{schema}

### EXTRACTION RULES:
1. Identify relevant nodes and their properties based on the schema.
2. Identify necessary relationships between the nodes.
3. Construct a path using relationships from the schema (connect from one node to the other to achieve the query).
4. Include any specific IDs mentioned in the query.
5. Double check if the direction is correct. It is strict (source)-[predicate]->(target)

### STRICT RULES:
1. Use ONLY node types and relationships specified in the provided ### Schema ### segment.
2. Common types include: gene, transcript, exon, protein, pathway, snp.
3. Common relationships include: transcribed_to, transcribed_from, includes, translates_to, interacts_with, genes_pathways.
4. Do not invent or reverse relationships. Direction is critical.
5. Ensure all nodes in relationships are included in the nodes list.
6. Only add property keys if mentioned in the user query.
7. Never infer an ID from your internal knowledge.

### CRITICAL ID vs PROPERTIES RULES:
- **Database ID**: If the query asks for a specific database ID (like "ensg00000186092"), put it in the `id` field
- **Property Value**: If the query asks for a name/identifier (like "BRCA1", "ENST00000441515"), put it in `properties`
- **Examples:**
  - "Find gene BRCA1" → `id: ""`, `properties: {{"gene_name": "BRCA1"}}`
  - "Find gene with ID ensg00000186092" → `id: "ensg00000186092"`, `properties: {{}}`
  - "Find transcript ENST00000441515" → `id: ""`, `properties: {{"transcript_id": "ENST00000441515"}}`

### RELATIONSHIP INFERENCE RULES:
- **Only add relationships when the query explicitly asks for connected information**
- **Examples:**
  - "Find gene BRCA1" → NO relationships needed (just the gene)
  - "Show me transcripts of gene BRCA1" → ADD transcribed_to relationship
  - "Find transcript ENST00000441515" → NO relationships needed (just the transcript)
  - "Show me exons of transcript ENST00000441515" → ADD includes relationship

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

**Relevant Relationships:** # ONLY if a connection or path is needed to achieve the query
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
1. Generate unique `node_ids` for each node in the format "label_X" (e.g., "gene_1", "transcript_1", "exon_1").
2. Include **ALL nodes** mentioned in the extracted information in the "nodes" list.
3. Ensure all nodes that appear in the predicates (relationships) are also included in the "nodes" list, even if they were not explicitly extracted.
4. Ensure all predicates (relationships) **exactly match** those defined in the schema (e.g., transcribed_to, transcribed_from, includes, translates_to, interacts_with).
5. **Do NOT add** any information not present in the extracted information or schema.
6. Use the correct node types defined in the schema (gene, transcript, protein, etc.).

### CRITICAL ID vs PROPERTIES RULES:
- **Database ID**: If the extracted info has a database ID (like "ensg00000186092"), put it in the `id` field
- **Property Value**: If the extracted info has a name/identifier (like "BRCA1", "ENST00000441515"), put it in `properties`
- **Examples:**
  - Gene name "BRCA1" → `id: ""`, `properties: {{"gene_name": "BRCA1"}}`
  - Database ID "ensg00000186092" → `id: "ensg00000186092"`, `properties: {{}}`
  - Transcript ID "ENST00000441515" → `id: ""`, `properties: {{"transcript_id": "ENST00000441515"}}`

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
```
"""

RESULT_SUMMARIZATION_PROMPT = """
You are a helpful biological database assistant. A user asked: "{query}"

The database search returned the following results:

**Nodes Found:**
{node_summary}

**Relationships Found:**
{relationship_summary}

**Instructions:**
Please provide a clear, natural language summary that:
1. Directly answers the user's question: "{query}"
2. Explains what was found in simple terms
3. Highlights the most important information
4. Uses biological terminology appropriately
5. Is conversational and helpful
6. Keeps the response under 200 words

**Response:**
"""