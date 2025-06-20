
def extract_tool_info(tool_name, dataset_id=None, src=None ):
    tools ={

    "bed to gff": {
        "tool_id": "bed2gff1",
        "input_format": "bed",
        "output_format": "gff",
        "tool_input": {
                        "input": {"src": src, "id": dataset_id},
                        "output_format": "gff",
                        "strand": True,
                        "feature_type": "gene"  # Tweakable: e.g., "exon", "CDS"
                    }
    },
    "genbank to gff3": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/iuc/bp_genbank2gff3/bp_genbank2gff3/1.1",
        "input_format": "gbk",
        "output_format": "gff3",
        "tool_input": {
                        "genbank": {"src": src, "id": dataset_id},
                        "noinfer": True,
                        "sofile": {
                            "sofile": "__none__"
                        },
                        "ethresh": "1",  # Tweakable: error threshold (string, e.g., "0.5")
                        "model": "--CDS",  # Tweakable: e.g., "--gene", "--mRNA"
                        "typesource": "contig"  # Tweakable: e.g., "chromosome"
                    }
    },
    "ab1 to fastq": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/ecology/ab1_fastq_converter/ab1_fastq_converter/1.20.0",
        "input_format": "ab1",
        "output_format": "fastq",
        "tool_input": {
                        "input": {"src": src, "id": dataset_id},
                        "tr": {
                                "trim": "false",  # Tweakable: "true"
                                "cutoff": 0.05,  # Tweakable: quality cutoff (float, e.g., 0.01)
                                "minseq": 20,  # Tweakable: min sequence length (int, e.g., 50)
                                "offset": 33.0  # Tweakable: quality offset (float, e.g., 64.0)
                            }
                        }
    },
    "bed to bigBed": {
        "tool_id": "bed_to_bigBed",
        "input_format": "bed",
        "output_format": "bigbed",
        "tool_input": {
                        "input": {"src": src, "id": dataset_id},
                        "settings": {
                            "settingsType": "preset",
                            "blockSize": 256,  # Tweakable: e.g., 128, 512
                            "itemsPerSlot": 512,  # Tweakable: e.g., 256, 1024
                            "unc": False  # Tweakable: True
                        }
                        }
    },
    "gff to bed": {
        "tool_id": "gff2bed1",
        "input_format": "gff",
        "output_format": "bed",
        "tool_input": {"input": {"src": src, "id": dataset_id}}
    },
    "gtf to bed12": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/iuc/gtftobed12/gtftobed12/357",
        "input_format": "gtf",
        "output_format": "bed12",
        "tool_input": {
                        "gtf_file": {"src": src, "id": dataset_id},
                        "advanced_options_selector": "advanced",
                        "ignoreGroupsWithoutExons": True,  # Tweakable: False
                        "simple": False,  # Tweakable: True
                        "allErrors": False,  # Tweakable: True
                        "impliedStopAfterCds": False,  # Tweakable: True
                        "includeVersion": False,  # Tweakable: True
                        "infoOut": False  # Tweakable: True
                    }
    },
    "fasta to tabular": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/fasta_to_tabular/fasta2tab/1.1.1",
        "input_format": "fasta",
        "output_format": "tabular",
        "tool_input": {
                        "input": {"src": src, "id": dataset_id},
                        "descr_columns": 2,  # Tweakable: e.g., 1, 3
                        "keep_first": 0  # Tweakable: e.g., 10 (chars to keep from header)
                    }
    },
    "tabular to fasta": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/tabular_to_fasta/tab2fasta/1.1.1",
        "input_format": "tabular",
        "output_format": "fasta",
        "tool_input": {"input": {"src": src, "id": dataset_id},
                       "title_col": [1],  # Tweakable: e.g., [2], [3]
                       "seq_col": 2  # Tweakable: e.g., 1, 3
                        }
    },
    "tabular to fastq": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/tabular_to_fastq/tabular_to_fastq/1.1.5+galaxy2",
        "input_format": "tabular",
        "output_format": "fastq",
        "tool_input": {
                        "input": {
                            "input_file": {"src": src, "id": dataset_id},
                            "identifier": "c1",  # Tweakable: e.g., "c2"
                            "sequence": "c2",  # Tweakable: e.g., "c1"
                            "quality": "c3"  # Tweakable: e.g., "c4"
                        }
                    }
    },
    "gff to gtf": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/devteam/gffread/gffread/2.2.1.4+galaxy0",
        "input_format": "gff",
        "output_format": "gtf",
        "tool_input": {
                        "input": {
                            "input_file": dataset_id,
                            "reference_genome": {
                                "source": "history",
                                "genome_fasta": "fasta_dataset_id"
                            },
                            "gffs": {
                                "gff_fmt": "gff"   # Tweakable: e.g., "gtf", "bed"
                            }
                        }
                    }
    },
    "twoBit to fasta": {
        "tool_id": "toolshed.g2.bx.psu.edu/repos/iuc/ucsc_twobittofa/ucsc-twobittofa/472",
        "input_format": "2bit",
        "output_format": "fasta",
        "tool_input": {
                        "input": {
                            "twobit_input": dataset_id
                        }
                    }
    }

    }
    return tools[tool_name]
