from collections import defaultdict
def graph():
    graph = defaultdict(list)
    for node, relationships in schema_relationship.items():
        for rel, target in relationships.items():
            if isinstance(target, list):
                for t in target:
                    graph[node].append((t, rel))
            else:
                graph[node].append((target, rel))
    return graph
    
schema_description = """
                    Node properties:

                    - **gene**
                    - `id`: STRING Example: "ensg00000101349"
                    - `gene_name`: STRING Example: "PAK5"
                    - `gene_type`: STRING Available options: ['protein_coding', 'lncRNA', 'misc_RNA', 'processed_pseudogene', 'miRNA']
                    - `synonyms`: STRING Example: "["PAK-7", "p21 (RAC1) activated kinase 7", "p21CDK"
                    - `start`: INTEGER Min: 1591604, Max: 13008972
                    - `end`: INTEGER Min: 1602520, Max: 13169103
                    - `chr`: STRING Available options: ['chr20', 'chrX', 'chrY']

                    - **transcript**
                    - `id`: STRING Example: "enst00000353224"
                    - `start`: INTEGER Min: 1591604, Max: 13009349
                    - `end`: INTEGER Min: 1600303, Max: 13169103
                    - `chr`: STRING Available options: ['chr20', 'chrX', 'chrY']
                    - `transcript_id`: STRING Example: "ENST00000353224.10"
                    - `transcript_name`: STRING Example: "PAK5-201"
                    - `transcript_type`: STRING Available options: ['protein_coding', 'lncRNA', 'protein_coding_CDS_not_defined', 'misc_RNA', 'retained_intron', 'processed_pseudogene', 'nonsense_mediated_decay', 'miRNA']
                    - `label`: STRING Available options: ['transcript']

                    - **exon**
                    - `id`: STRING Example: "ense00001901152"
                    - `start`: INTEGER Min: 9557608, Max: 13009349
                    - `end`: INTEGER Min: 9557734, Max: 13009384
                    - `chr`: STRING Available options: ['chr20']
                    - `exon_number`: INTEGER Min: 1, Max: 26
                    - `exon_id`: STRING Example: "ENSE00001901152"

                    - **protein**
                    - `id`: STRING Example: "q9nu02"
                    - `protein_name`: STRING Example: "ANKE1"
                    - `accessions`: STRING Example: "["B3KUQ0", "Q9H6Y9"]"

                    - **promoter**
                    - `id`: STRING Example: "chr1_959246_959305_grch38"
                    - `start`: INTEGER Example: "959246"
                    - `end`: INTEGER Example: "959305"
                    - `chr`: STRING Example: "chr1"

                    - **snp**
                    - `id`: STRING Example: "rs367896724"
                    - `start`: INTEGER Example: "10177"
                    - `end`: INTEGER Example: "10177"
                    - `chr`: STRING Example: "chr1"
                    - `ref`: STRING Example: "A"
                    - `caf_ref`: FLOAT Example: "0.5747"
                    - `alt`: STRING Example: "AC"
                    - `caf_alt`: FLOAT Example: "0.4253"

                    - **enhancer**
                    - `id`: STRING Example: "chr1_203028401_203028890_grch38"
                    - `start`: INTEGER Example: "203028401"
                    - `end`: INTEGER Example: "203028890"
                    - `chr`: STRING Example: "chr1"

                    - **pathway**
                    - `id`: STRING Example: "r-hsa-164843"
                    - `pathway_name`: STRING Example: "2-LTR circle formation"

                    - **super_enhancer**
                    - `id`: STRING Example: "chr1_119942741_120072457_grch38"
                    - `start`: INTEGER Example: "119942741"
                    - `end`: INTEGER Example: "120072457"
                    - `chr`: STRING Example: "chr1"
                    - `se_id`: STRING Example: "SE_00001"

                    Relationship properties:

                    The relationships:
                    (:gene)-[:transcribed to]->(:transcript)
                    (:transcript)-[:translates to]->(:protein)
                    (:transcript)-[:transcribed from]->(:gene)
                    (:transcript)-[:includes]->(:exon)
                    (:protein)-[:translation_of]->(:transcript)
                    """

additional_nodes = """
                    ontology_term               

                    chromosome_chain

                    position_entity

                    transcription_binding_site

                    tad

                    regulatory_region              

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
                        "transcribed from": "gene",
                        "translates to": "protein",
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
                        "property1": "<value>",
                        "property2": "<value>"
                    }
                    },
                    {
                    "node_id": "n2",
                    "id":"",
                    "type": "label",
                    "properties": {
                        "property1": "<value>",
                        "property2": "<value>"
                    }
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