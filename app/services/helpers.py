schema_description = '''
                    Nodes: gene, transcript, protein

                    if node is gene it might include this properies such as:
                    gene {gene_name: STRING, gene_type: STRING, synonyms: LIST}
                    example gene_name = ENSG00000237491, gene_type = protein_coding, synonyms = [bA204H22.1, dJ1099D15.3, C20orf94]

                    if node is transcript it might include this properies such as:
                    transcript {gene_name: STRING}
                    example gene_name = AKAP17A

                    if node is protein it might include this properies such as:
                    protein {synonyms: LIST, protein_name: STRING}
                    example protein_name= MKKS,ANKE1, synonyms = []

                    Relationship properties are only transcribed_to ,transcribed_from, translates_to ,translation_of

                    The relationships are only this way dont create any other relationships :
                    (:gene)-[:transcribed_to]->(:transcript)
                    (:transcript)-[:transcribed_from]->(:gene)
                    (:transcript)-[:translates_to]->(:protein)
                    (:protein)-[:translation_of]->(:transcript)
                    '''

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
sample_graph  = {"nodes":[
  {
            "data": {
                "id": "enst00000378392",
                "type": "transcript",
                "gene_name": "ANKEF1",
                "transcript_id": "ENST00000378392.6",
                "start": "10034987",
                "transcript_name": "ANKEF1-202",
                "end": "10058303",
                "source": "GENCODE",
                "chr": "chr20",
                "transcript_type": "protein_coding",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        },
        {
            "data": {
                "id": "enst00000378380",
                "type": "transcript",
                "gene_name": "ANKEF1",
                "transcript_id": "ENST00000378380.4",
                "start": "10035049",
                "transcript_name": "ANKEF1-201",
                "end": "10058303",
                "source": "GENCODE",
                "chr": "chr20",
                "transcript_type": "protein_coding",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        },
        {
            "data": {
                "id": "enst00000488991",
                "type": "transcript",
                "gene_name": "ANKEF1",
                "transcript_id": "ENST00000488991.1",
                "start": "10035117",
                "transcript_name": "ANKEF1-204",
                "end": "10055827",
                "source": "GENCODE",
                "chr": "chr20",
                "transcript_type": "protein_coding_CDS_not_defined",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        }
  ],
  "edges": [
        {
            "data": {
                "id": 1152921504606847846,
                "label": "transcribed_to",
                "source_node": "gene ensg00000101349",
                "target_node": "transcript enst00000353224",
                "source_data": "GENCODE",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        },
        {
            "data": {
                "id": 1155173304420533094,
                "label": "transcribed_to",
                "source_node": "gene ensg00000101349",
                "target_node": "transcript enst00000378423",
                "source_data": "GENCODE",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        },
        {
            "data": {
                "id": 6917529027641082780,
                "label": "transcribed_to",
                "source_node": "gene ensg00000101349",
                "target_node": "transcript enst00000378429",
                "source_data": "GENCODE",
                "source_url": "https://www.gencodegenes.org/human/"
            }
        },]
}