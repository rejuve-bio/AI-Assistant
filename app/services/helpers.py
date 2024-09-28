schema_description = """
                    **Schema Description:**

                    Nodes:
                    gene
                    Properties:
                    gene_name: STRING
                    gene_type: STRING
                    synonyms: LIST
                    Example:
                    gene_name = ENSG00000237491
                    gene_type = protein_coding
                    synonyms = [bA204H22.1, dJ1099D15.3, C20orf94]

                    transcript
                    Properties:
                    gene_name: STRING
                    Example:
                    gene_name = AKAP17A

                    protein
                    Properties:
                    protein_name: STRING
                    synonyms: LIST
                    Example:
                    protein_name = MKKS, ANKE1
                    synonyms = []

                    ontology_term

                    pathway

                    snp

                    chromosome_chain

                    position_entity

                    transcription_binding_site

                    tad

                    regulatory_region

                    enhancer

                    promoter

                    super_enhancer

                    non_coding_rna

                    CL

                    Uberon

                    BTO

                    EFO

                    CLO

                    GO
                    """
schema_relationship={
                    "gene": {
                        "transcribed_to": "transcript",
                        "coexpressed_with": "gene",
                        "expressed_in": "ontology_term",
                        "genes_pathways": "pathway",
                        "regulates": ["gene","regulatory_region"],
                        "associated_with":["enhancer","promoter","super_enhancer"],
                        "tfbs_snp": "snp",
                        "binds_to": "transcription_binding_site",
                        "in_tad_region": "tad",
                    },
                    "transcript": {
                        "transcribed_from": "gene",
                        "translates_to": "protein",
                    },
                    "protein": {
                        "translation_of": "transcript",
                        "interacts_with": "protein",
                        "go_gene_product": "go",
                    },
                    "ontology_term": {
                        "has_part": "ontology_term",
                        "part_of": "ontology_term",
                        "subclass_of": "ontology_term",
                    },
                    "cl": {
                        "capable_of": "go",
                        "part_of": "uberon",
                        "subclass_of": "cl",
                    },
                    "bto": {
                        "subclass_of": "bto",
                    },
                    "efo": {
                        "subclass_of": "efo",
                    },
                    "uberon": {
                        "subclass_of": "uberon",
                    },
                    "clo": {
                        "subclass_of": "clo",
                    },
                    "go": {
                        "subclass_of": "go",
                    },
                    "pathway": {
                        "parent_pathway_of": "pathway",
                        "child_pathway_of": "pathway",
                    },
                    "snp": {
                        "eqtl_association": "gene",
                        "closest_gene": "gene",
                        "upstream_gene": "gene",
                        "downstream_gene": "gene",
                        "in_gene": "gene",
                        "in_ld_with": "snp",
                        "activity_by_contact": "gene",
                        "chromatin_state": "uberon",
                        "in_dnase_I_hotspot": "uberon",
                        "histone_modification": "uberon",
                    },
                    "chromosome_chain": {
                        "lower_resolution": "chromosome_chain",
                    },
                    "position_entity": {
                        "located_on_chain": "chromosome_chain",
                    },
                    "regulatory_region": {
                        "regulates": "gene",
                    },
                }
nodes_edges_prompt = '''
                    Given the user query
                    
                    {self.query}
                    
                    you are supposed to identify and create a graph based on the following schema and refering to the description on the schema
                    
                    {self.schema_description}

                    1. dont use the examples in the description just return the result only from the query
                    2. return the result in json format in key value pairs where the keys are the schema name and value is the user query we will set to query the graph
                    3. only return a relationship related with user question
                    3. dont explain anything just return the result in json format
                    4. return me in a cypher query format
                    '''

relationship_extractor = """
                        You are an assistant that extracts graph nodes and relationships from user queries based on a given schema.

                        {schema}

                        **Allowed Relationships:**
                        {schema_relationships}

                        **Instructions:**
                        Given the user query below:
                        {query}

                        1. Identify and extract the relevant nodes from the query.
                        2. Extract the source node and target node from the user query.
                        3. Always return the response in the following strict JSON format only:

                        {{
                            "source_node": {{
                                "type": "<type_of_source_node>",
                                "properties": <source_node_properties>
                            }},
                            "target_node": {{"type": "<type_of_target_node>", "properties": <target_node_properties>}}  // Include this line only if a target node exists
                        }}
                        """
json_constructor_prompt = """
            The user question is:
            {self.query}
            Given the following extracted information from a query:
            {self.nodes_and_edges}
            Convert this information into the following JSON format:
            {self.sample_json_format}
            Please follow these rules when creating the output:
            1. For each node, include node_id, id, type, and properties fields. These are mandatory and should be the only fields for nodes.
            2. Generate node_ids by auto-incrementing for each node (n1, n2, n3, ...).
            3. If a specific ID is given in the extracted information, use it in the 'id' field. Otherwise, leave it as an empty string.
            4. Add the label of the node in the type filed
            5. Include all relevant properties from the extracted information in the properties field of each node.
            6. For predicates (relationships), include type, source, and target fields. The source and target should reference the node_ids.
            7. Ensure that the output JSON structure matches the sample format provided.
            8. Make sure all relationships in the extracted information are represented as predicates in the output.
            9. Make sure all nodes in the relation ship are available in the nodes list
            10.Dont mention id in the dicitionary
            Provide only the resulting JSON as your response, without any additional explanation or commentary.
            """

sample_json_format = """
                {
                "nodes": [
                    {
                    "node_id": "n1",
                    "type": "label",
                    "properties": {
                        "key": "value"
                    }
                    },
                    {
                    "node_id": "n2",
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
json_constructor_prompt = """
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
        3. If a specific ID is given in the extracted information, use it in the 'id' field. Otherwise, leave it as an empty string.
        4. Add the label of the node in the type filed
        5. Include all relevant properties from the extracted information in the properties field of each node.
        6. For predicates (relationships), include type, source, and target fields. The source and target should reference the node_ids.
        7. Ensure that the output JSON structure matches the sample format provided.
        8. Make sure all relationships in the extracted information are represented as predicates in the output.
        9. Make sure all nodes in the relation ship are available in the nodes list
        10.Dont mention id in the dicitionary
        Provide only the resulting JSON as your response, without any additional explanation or commentary.
        """