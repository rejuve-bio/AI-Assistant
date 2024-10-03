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
                        "transcribed to": "transcript",
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

sample_json_format = """
                {
                "nodes": [
                    {
                    "node_id": "n1",
                    "id":"",
                    "type": "label",
                    "properties": {
                        "key": "value"
                    }
                    },
                    {
                    "node_id": "n2",
                    "id":"",
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