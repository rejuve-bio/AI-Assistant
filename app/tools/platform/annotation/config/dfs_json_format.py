schema = [
    "gene {id: STRING, gene_name: STRING, gene_type: STRING, synonyms: STRING, start: INTEGER, end: INTEGER, chr: STRING}",
    "transcript {id: STRING, start: INTEGER, end: INTEGER, chr: STRING, transcript_id: STRING, transcript_name: STRING, transcript_type: STRING, label: STRING}",
    "exon {id: STRING, start: INTEGER, end: INTEGER, chr: STRING, exon_number: INTEGER, exon_id: STRING}",
    "protein {id: STRING, protein_name: STRING, accessions: STRING, start: INTEGER}",
    "promoter {id: STRING, start: INTEGER, end: INTEGER, chr: STRING}",
    "enhancer {id: STRING, start: INTEGER, end: INTEGER, chr: STRING}",
]

nodes_template = {
    "node_id": "",
    "id": "",
    "type": "",
    "properties": {}
}

predicates_template = {
    "type": "",
    "source": "",
    "target": ""
}