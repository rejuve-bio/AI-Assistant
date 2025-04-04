TOOL_PROMPT = """ 
**Role**:
You are a senior bioinformatician specializing in Tool automation and information extraction. 
Your task is to analyze raw Galaxy tool metadata and produce a structured JSON summary that can power an intelligent agent for automating Galaxy operations.

**Goal**: Convert the input metadata into a detailed JSON object with the following structure:

```json
{{
  "tool_id": "exact_tool_id_from_galaxy",
  "name": "Official Tool Name",
  "description": "1–2 sentence summary of the tool's functionality. Mention biological use cases where relevant.",
  "input_types": ["List of Galaxy data types consumed (e.g., fastq, bam, fasta)"],
  "output_types": ["List of Galaxy data types produced"],
  "categories": ["Galaxy categories or ['general'] if none"],
  "main_parameters": ["Names of required parameters"],
  "optional_parameters": ["Names of optional parameters"],
  "parameters_detail": {{
    "parameter_name": {{
      "type": "text|integer|float|boolean|data|select|... (as found in tool metadata)",
      "accepted_formats": ["only for type: data", e.g., fastq, bam, etc.],
      "optional": true|false,
      "description": "Optional short explanation if available in metadata"
    }}
  }}
}}
```

**Formatting Rules**:
- Input and output types must be Galaxy-recognized types.
- For `parameters_detail`, always include `type` and `optional`. Include `accepted_formats` only for `type: data`.
- Do not include markdown, extra comments, or any formatting outside of the JSON block.

**Instructions**: Analyze the following raw Galaxy tool metadata and return only the structured JSON summary in the format above:

**Raw Tool Metadata**:  
{input}

"""

WORKFLOW_PROMPT = """
**Role**: 
You are a senior bioinformatician and automation architect working with Galaxy workflows.
Your task is to analyze raw Galaxy workflow metadata and produce a structured JSON summary that can be used by an intelligent agent to automate workflow execution, validation, and editing using the BioBlend API.

**Goal**: 
Convert the input metadata into a structured JSON object with detailed information about workflow structure, tools used, parameters, inputs, and outputs.

```json
{{
  "workflow_id": "exact_workflow_id_from_galaxy",
  "name": "Official Workflow Name",
  "description": "1–2 sentence summary of the workflow's purpose and biological application.",
  "inputs": {{
    "input_name": {{
      "type": "data",
      "accepted_formats": ["fastq", "bam", ...]
    }}
  }},
  "outputs": {{
    "output_name": {{
      "type": "data",
      "format": "bam|vcf|fasta|..."
    }}
  }},
  "steps": {{
    "step_index": {{
      "tool_id": "tool_id_used_in_step",
      "name": "Tool Name",
      "description": "Short description of what the step does.",
      "inputs": {{
        "input_name": {{
          "source_step": "step_index_or_input_label",
          "type": "data",
          "format": "fastq|bam|..."
        }}
      }},
      "outputs": {{
        "output_name": {{
          "format": "bam|vcf|..."
        }}
      }},
      "parameters": {{
        "parameter_name": {{
          "value": "default_value_if_present",
          "optional": true|false
        }}
      }}
    }}
  }},
  "categories": ["Workflow categories if present or inferred, e.g., NGS", "Alignment", "RNA-seq"]
}}
```

**Formatting Rules**:
- All I/O data types should be Galaxy-recognized (e.g., fastq, bam).
- Step indices should be retained as in the workflow structure (0, 1, 2...).
- For each step, describe the connection to previous steps or workflow inputs.
- Keep the JSON clean, with no markdown, comments, or extra formatting outside the JSON block.


**Instructions**: Analyze the following raw workflow metadata and return only the completed JSON summary in the required format:

**Raw Workflow Metadata**:  
{input}

"""


DATASET_PROMPT = """
**Role**:
You are a senior bioinformatician specializing in Galaxy dataset management and metadata curation. Your task is to analyze raw Galaxy dataset metadata and produce a structured JSON summary for automation agents handling data operations.

**Goal**: Convert the input metadata into a detailed JSON object with the following structure:

```json
{{
  "dataset_id": "exact_dataset_id_from_galaxy",
  "name": "Human-readable Dataset Name",
  "description": "1-sentence summary of data content and purpose",
  "data_type": "Galaxy data type (e.g., fastq, bam, fasta)",
  "format": "Specific format variant (e.g., fastqsanger, bam_index)",
  "source": {{
    "type": "input|tool_output|workflow_output",
    "tool_id": "tool_id_if_generated_by_tool",
    "workflow_id": "workflow_id_if_workflow_output",
    "step_index": "workflow_step_number_if_applicable"
  }},
  "metadata": {{
    "creation_date": "ISO 8601 timestamp",
    "file_size": "size_in_bytes",
    "file_format": "detailed_format_info",
    "annotations": ["user-provided tags or notes"]
  }},
  "provenance": {{
    "tool_id": "source_tool_if_applicable",
    "tool_version": "tool_version_used",
    "parameters": {{
      "parameter_name": "parameter_value_used_in_generation"
    }}
  }},
  "accessible": true|false,
  "url": "direct_download_url_if_available"
}}
```

**Formatting Rules**:
- Use official Galaxy data types and formats
- Include full provenance chain when available
- For source type: 
  - Use 'input' for original uploads
  - 'tool_output' for tool-generated data
  - 'workflow_output' for workflow-generated data
- Preserve all timestamps in ISO 8601 format
- Include byte sizes as integers
- Keep URLs only if publicly accessible

**Instructions**: Analyze the following raw dataset metadata and return only the structured JSON summary in the specified format:

**Raw Dataset Metadata**:
{input}
"""
