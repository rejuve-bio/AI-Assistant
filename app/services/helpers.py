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
