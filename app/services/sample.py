query = "Provide information about gene associated protein with MKKS"

json_format = {
"nodes": [
  {
    "node_id": "n1",
    "id": "",
    "type": "gene",
    "properties": {}
  },
  {
    "node_id": "n2",
    "id": "",
    "type": "transcript",
    "properties": {}
  },
  {
    "node_id": "n3",
    "id": "",
    "type": "protein",
      "properties": {
      "protein_name": "MKKS"
    }
  },
  {
    "node_id": "n4",
    "id": "",
    "type": "protein",
      "properties": {
      "protein_name": "MKKS"
    }
  }
],
"predicates": [
  {
    "type": "transcribed to",
    "source": "n1",
    "target": "n2"
  },
  {
    "type": "translates to",
    "source": "n2",
    "target": "n3"
  }
]
 }


annotation_return = {
    "nodes": [
        {
            "data": {
                "id": "q9npj1",
                "type": "protein",
                "protein_name": "MKKS",
                "accessions": "[\"A8K7B0\", \"D3DW18\"]"
            }
        },
        {
            "data": {
                "id": "ensg00000125863",
                "type": "gene",
                "gene_name": "MKKS",
                "gene_type": "protein_coding",
                "synonyms": "[\"HMCS\", \"MKS\", \"MKKS centrosomal shuttling protein\", \"McKusick-Kaufman/Bardet-Biedl syndromes putative chaperonin\", \"Bardet-Biedl syndrome 6 protein\", \"KMS\", \"HGNC:7108\", \"alternative protein MKKS\", \"MKKS\", \"molecular chaperone MKKS\", \"BBS6\"]",
                "start": 10401009,
                "end": 10434222,
                "chr": "chr20"
            }
        },
        {
            "data": {
                "id": "enst00000347364",
                "type": "transcript",
                "transcript_id": "ENST00000347364.7",
                "start": 10401009,
                "transcript_name": "MKKS-201",
                "end": 10434222,
                "label": "transcript",
                "chr": "chr20",
                "transcript_type": "protein_coding"
            }
        },
        {
            "data": {
                "id": "enst00000651692",
                "type": "transcript",
                "transcript_id": "ENST00000651692.1",
                "start": 10401621,
                "transcript_name": "MKKS-203",
                "end": 10434214,
                "label": "transcript",
                "chr": "chr20",
                "transcript_type": "protein_coding"
            }
        },
        {
            "data": {
                "id": "enst00000399054",
                "type": "transcript",
                "transcript_id": "ENST00000399054.6",
                "start": 10405184,
                "transcript_name": "MKKS-202",
                "end": 10431922,
                "label": "transcript",
                "chr": "chr20",
                "transcript_type": "protein_coding"
            }
        }
    ],
    "edges": [
        {
            "data": {
                "id": 1152921504606887443,
                "label": "transcribed_to",
                "source_node": "gene ensg00000125863",
                "target_node": "transcript enst00000347364"
            }
        },
        {
            "data": {
                "id": 1152922604118515325,
                "label": "translates_to",
                "source_node": "transcript enst00000347364",
                "target_node": "protein q9npj1"
            }
        },
        {
            "data": {
                "id": 1155173304420572691,
                "label": "transcribed_to",
                "source_node": "gene ensg00000125863",
                "target_node": "transcript enst00000651692"
            }
        },
        {
            "data": {
                "id": 1152922604118515326,
                "label": "translates_to",
                "source_node": "transcript enst00000651692",
                "target_node": "protein q9npj1"
            }
        },
        {
            "data": {
                "id": 1159676904047943187,
                "label": "transcribed_to",
                "source_node": "gene ensg00000125863",
                "target_node": "transcript enst00000399054"
            }
        },
        {
            "data": {
                "id": 1152922604118515328,
                "label": "translates_to",
                "source_node": "transcript enst00000399054",
                "target_node": "protein q9npj1"
            }
        }
    ]
}