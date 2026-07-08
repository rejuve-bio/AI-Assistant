ORGANISM_DETECTION_PROMPT = """You are a biological query classifier. Determine whether the following query is about human biology or Drosophila melanogaster (fruit fly) biology.

## ABSOLUTE OVERRIDES — answer "fly" immediately if ANY of these are present, regardless of other gene names or terms:

**Organism words:**
- "Drosophila", "dmel", "fruit fly", "melanogaster", "D. melanogaster"

**FlyBase identifiers:**
- Any identifier starting with: FBgn, FBal, FBtr, FBbt, FBdv, FBrf (e.g. FBgn0000490, FBgn999xyz)

**Fly-exclusive gene names (absent in human):**
- wg, hh, dpp, en, eve, ftz, vg, arm, sev, boss, ci, Dl, N (as Notch fly symbol), nkd, ptc, smo, cos2, fu, su(fu), puc, bsk, hep, Mst, yki, sd, ex, ft, ds, fj, mer, mats, wts, lats, sav, hpo

**Fly-specific anatomy and tissue terms:**
- fat body, wing disc, eye disc, leg disc, imaginal disc, ommatidia, ommatidium
- bristle, chaeta, salivary gland polytene, polytene chromosome
- cardia, proventriculus, Malpighian tubule, trachea (fly context), hemocyte
- dorsal vessel, garland cell, ring gland, corpus allatum, corpus cardiacum

**Fly-specific cell types:**
- hemocyte, plasmatocyte, crystal cell, lamellocyte, oenocyte, tracheal cell, neuropil

**Fly developmental and biological terms:**
- larval stage, pupal stage, wandering larva, embryonic stage (with number), imaginal
- metamorphosis (fly context), oogenesis (fly context), follicle cell, nurse cell, pole cell

**Fly allele notation:**
- Gene symbols with bracket suffixes: w[1118], p53[A], brca2[P1], ry[506]

---

CRITICAL RULE: A query may mix fly-specific and human-sounding gene/term names (e.g. cross-species comparisons, or fly genes like p53 that share names with human genes). If ANY absolute override signal above is present anywhere in the query, answer "fly". Ambiguous gene names like p53, Rb, E2f, brca2, Myc do NOT override a fly signal.

Default to "human" ONLY when none of the above fly signals appear anywhere in the query.

Query: {query}

Respond with ONLY one word: "human" or "fly". No explanation.
"""


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

### ANNOTATION TARGET RULE:
- In compound queries like "Annotate [X] and find/tell/show [Y]" or "Annotate [X] for/about/in [Y]":
  - [X] is the annotation target → create nodes ONLY for [X]
  - [Y] is a research purpose, context, or separate sub-task → do NOT create nodes for [Y]
- The annotation target is what comes immediately after "annotate" (or "find"/"show" in annotation context)
- Research context words like "longevity", "aging", "cancer" appearing after "for", "about", "its role in", "investigating it for" are NOT annotation nodes
- Examples:
  - "Annotate FOXO3 and tell me about longevity" → extract FOXO3 only (longevity is research context)
  - "Annotate TP53 and find papers about its role in cancer" → extract TP53 only (cancer is a literature topic, not an annotation target)
  - "Annotate TP53 and its association with cancer" → extract both (explicit relationship requested)
  - "Annotate FOXO3 and BRCA1" → extract both (both are annotation targets)

### STRICT RULES:
- Use only node types and relationships specified in the schema above.
- Do not invent or reverse relationships.
- Ensure all nodes in relationships are included in the list.
- Only add property keys if mentioned in the user query
- Never grab the property from the schema
- Never infer an id from your knowledge
- **NEVER create nodes for entities not explicitly named in the user query — do NOT add similar or related entities based on your own knowledge**

### LIST DETECTION RULES:
- If the query provides multiple values for the same node type (e.g. a list of gene names), treat them as a SINGLE list node — do NOT create one node per value.
- Mark list nodes with `is_list: true` and set the property value as a comma-separated list.
- Examples:
  - "genes BRCA1, TP53, EGFR" → ONE node, `is_list: true`, `gene_name: BRCA1, TP53, EGFR`
  - "any gene in PTEN, RB1, CDK2" → ONE node, `is_list: true`, `gene_name: PTEN, RB1, CDK2`

### CRITICAL ID vs PROPERTIES RULES:
- **Database ID**: If the query asks for a specific database ID (like "ensg00000186092"), put it in the `id` field
- **Property Value**: If the query asks for a name/identifier (like "BRCA1", "ENST00000441515"), put it in `properties`
- **Examples:**
  - "Find gene BRCA1" → `id: ""`, `properties: {{"gene_name": "BRCA1"}}`
  - "Find gene with ID ensg00000186092" → `id: "ensg00000186092"`, `properties: {{}}`
  - "Find transcript ENST00000441515" → `id: ""`, `properties: {{"transcript_id": "ENST00000441515"}}`

### RELATIONSHIP INFERENCE RULES:
- **Only add relationships when the query EXPLICITLY names a second entity type or uses words like "related to", "connected to", "transcripts of", "exons of", "regulates", etc.**
- Words like "annotate", "find", "show", "get", "what is" do NOT imply relationships — return only the named node
- **Examples:**
  - "Find gene BRCA1" → NO relationships (just the gene)
  - "Annotate gene BRCA1" → NO relationships (just the gene)
  - "Show me transcripts of gene BRCA1" → ADD transcribed_to relationship
  - "What TAD does FTO sit in?" → ADD in_tad_region relationship
  - "Find transcript ENST00000441515" → NO relationships (just the transcript)

### RESPONSE FORMAT:
Provide your response in the following format:

**Relevant Nodes:**
- Node Type: `node_type1`
  - ID: `specific_id_or_empty_string`
  - Is List: false
  - Properties:
    - key: value # ONLY if mentioned in the user Query

- Node Type: `node_type2` (list node example)
  - ID: ``
  - Is List: true
  - Properties:
    - key: "value1, value2, value3"

- Node Type: `node_type3`
  - ID: ``
  - Is List: false
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
1. Generate unique `node_ids` for each node in the format "label_X" (e.g., "gene_1", "transcript_1", "gene_list_A").
2. Include **ONLY nodes explicitly named in the user query** — never add nodes based on your own knowledge or similarity.
3. Ensure all nodes that appear in the predicates (relationships) are also included in the "nodes" list.
4. Ensure all predicates (relationships) **exactly match** those defined in the schema above.
5. **Do NOT add** any information not present in the extracted information or schema.
6. Use the correct node types from the schema above.
7. If a node is a list (`is_list: true`), set the property value as a comma-separated string (e.g., "BRCA1, TP53, EGFR") and include `"is_list": true` on the node. Do NOT use a JSON array.

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
      "is_list": false,
      "properties": {{
        "key": "value"
      }}
    }},
    {{
      "node_id": "label_list_A",
      "id": "",
      "type": "label",
      "is_list": true,
      "properties": {{
        "key": "value1, value2, value3"
      }}
    }}
    ...
  ],
  "predicates": [
    {{
      "type": "predicate",
      "source": "label_1",
      "target": "label_list_A"
    }}
    ...
  ]
}}
"""

EXTRACT_AND_CONVERT_PROMPT = """
## TASK:
Extract relevant biological entities from the query and directly output the annotation JSON. Do this in one step.

### Query: {query}

### Schema:
{schema}

### EXTRACTION RULES:
1. Identify relevant nodes and their properties based on the schema.
2. Identify necessary relationships between the nodes.
3. Include any specific IDs mentioned in the query.
4. Check relationship direction — it is strict: (source)-[predicate]->(target)

### YOUR ROLE:
You are the ANNOTATION AGENT. Your only job is to extract entities to annotate into a graph. You do NOT handle literature search, clinical trials, or general Q&A — those are separate agents. Only extract what needs to go into the annotation graph.

### COMPOUND QUERY RULE:
Queries often combine an annotation request with a separate task. ONLY extract the annotation target — ignore the rest.
- "Annotate TP53 and find papers about its role in cancer" → extract TP53 only. "cancer" is for the literature agent.
- "Annotate FOXO3 and tell me what clinical trials are investigating it for longevity" → extract FOXO3 only. "longevity" is research context.
- "Annotate FOXO3 for longevity" → extract FOXO3 only. "longevity" describes research purpose, not an annotation target.
- "Annotate TP53 and its association with cancer" → extract TP53 AND cancer (explicit relationship between two named entities).
- "Annotate FOXO3 and BRCA1" → extract both (both are annotation targets).
Trigger phrases that signal a SEPARATE task (NOT annotation): "find papers", "tell me about", "what clinical trials", "for longevity", "about aging", "its role in", "investigating it for".

### STRICT RULES:
- Use only node types and relationships specified in the schema.
- Do not invent or reverse relationships.
- Only add property keys if mentioned in the user query — never infer from your knowledge.
- Never infer an id from your knowledge.
- NEVER create nodes for entities not explicitly named in the query.
- If the query provides multiple values for the same node type, treat them as ONE list node with `is_list: true` and a comma-separated property value.
- Only add relationships when the query EXPLICITLY names a second entity type or uses relational words like "related to", "transcripts of", "regulates", etc.

### CRITICAL ID vs PROPERTIES:
- Database ID (e.g. "ensg00000186092") → put in `id` field
- Name/identifier (e.g. "BRCA1") → put in `properties`

### Response format (JSON only, no extra text):
{{
  "nodes": [
    {{
      "node_id": "label_1",
      "id": "id_or_empty_string",
      "type": "label",
      "is_list": false,
      "properties": {{
        "key": "value"
      }}
    }}
  ],
  "predicates": [
    {{
      "type": "predicate",
      "source": "label_1",
      "target": "label_2"
    }}
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

SELECT_PROPERTY_VALUES_BATCH_PROMPT = """You are a biological database validation assistant.

For each search query below you are given candidate values found via string-similarity search in a biological database.
For each query decide:
1. Is any candidate the same biological entity as the query?
2. If yes — is the difference trivial (case, whitespace, punctuation, obvious typo) that it can be silently auto-corrected, or is it different enough that the user should confirm?

Rules:
- auto_accept: true  → same entity, trivial difference only (e.g. "BRAC1"→"BRCA1", "tp53"→"TP53", "Alzheimers Disease"→"Alzheimer's Disease")
- auto_accept: false → plausible match but genuinely different-looking (e.g. random string → real gene name the user probably didn't know)
- null               → no plausible match at all

Items to evaluate (JSON):
{items_json}

Respond with ONLY a valid JSON object. No explanation, no markdown fences.
Example: {{"BRAC1": {{"value": "BRCA1", "auto_accept": true}}, "hgf6d7": {{"value": "ZNF697", "auto_accept": false}}, "xyz999": null}}
"""
