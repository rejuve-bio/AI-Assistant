Full README.md content:

```markdown
# Code Execution Agent - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [What is the Code Execution Agent?](#what-is-the-code-execution-agent)
3. [Service Integration (Like RAG, Annotation, etc.)](#service-integration)
4. [Supported File Types](#supported-file-types)
5. [Architecture and Components](#architecture-and-components)
6. [The Hybrid Approach](#the-hybrid-approach)
7. [ReAct Framework Integration](#react-framework-integration)
8. [Complete Workflow](#complete-workflow)
9. [Module Details](#module-details)
10. [Configuration](#configuration)
11. [Testing the Code Execution Agent](#testing-the-code-execution-agent)
12. [Usage Examples](#usage-examples)
13. [Safety and Security](#safety-and-security)
14. [Error Handling and Self-Correction](#error-handling-and-self-correction)
15. [Artifact Management](#artifact-management)
16. [Comparison with Other Services](#comparison-with-other-services)

---

## Overview

The Code Execution Agent is a secure, intelligent Python code execution service integrated into the AI Assistant. It accepts various document types (CSV, HTML, XML, PDF, URLs), uses Large Language Models (LLMs) to generate Python code based on user instructions, executes that code safely in a sandboxed environment, and returns results including plots, statistical analyses, and data transformations.

**Key Capabilities:**
- Dynamic code generation using LLMs (Gemini/OpenAI)
- Multi-format data ingestion and normalization
- Statistical analysis and visualization
- Bioinformatics-specific operations
- Self-correcting error handling via ReAct framework
- Publication-ready outputs

---

## What is the Code Execution Agent?

The Code Execution Agent transforms the AI Assistant from a document retrieval system into an **interactive analytical partner**. Unlike other services that retrieve or annotate existing knowledge, the Code Execution Agent:

1. **Executes Code**: Generates and runs Python code to perform calculations, statistical tests, and visualizations
2. **Processes Raw Data**: Works directly with uploaded CSV, HTML, XML, PDF files and URLs
3. **Adapts Dynamically**: Uses LLMs to understand user intent and generate appropriate code
4. **Corrects Errors**: Implements ReAct framework to iteratively fix code errors
5. **Returns Artifacts**: Produces plots, tables, and analysis results ready for publication

**What Makes It Unique:**
- **Intelligence**: Uses LLMs to understand complex research questions and generate appropriate code
- **Flexibility**: Can handle diverse data analysis tasks without pre-programmed workflows
- **Safety**: Executes code in a sandboxed environment with timeouts and resource limits
- **Completeness**: Provides end-to-end analysis from raw data to publication-ready figures

---

## Service Integration

### How It Works as a Service (Like RAG)

The Code Execution Agent is integrated as a **first-class service** in the AI Assistant ecosystem, working alongside:

- **RAG**: Retrieval-Augmented Generation for document search
- **Annotation**: Knowledge graph annotation and querying
- **Galaxy**: Galaxy workflow integration
- **Hypothesis**: Hypothesis generation from variants/phenotypes

#### Integration Points

**1. Unified API Endpoint (`/query`)**
```
All services, including Code Execution, use the same /query endpoint
The classifier automatically routes queries to the appropriate service
```

**2. LangGraph Workflow Integration**
```python
# In app/main.py
code_exec_agent node → Handles code execution requests
_classify_query() → Identifies code_exec intent
_route_query() → Routes to code_exec_agent
```

**3. Query Classification**
The system uses an LLM-based classifier to determine which service to use:
- `code_exec`: Requests for calculations, plots, analysis on data files
- `rag`: General information, document search
- `annotation`: Gene/protein/variant queries
- `galaxy`: Galaxy tool/workflow questions
- `hypothesis`: Hypothesis generation requests

**4. Response Format**
All services return consistent response structures:
```json
{
  "response": "...",
  "resource": {
    "type": "code_exec",
    "id": "<run_id>"
  },
  "artifacts": [...]
}
```

#### Automatic Service Selection

**Example Query Flow:**
1. User asks: *"Compute correlation heatmap from this CSV"*
2. Classifier analyzes query → identifies `code_exec` intent
3. Router sends query to `code_exec_agent` node
4. Code Execution Agent processes the request
5. Returns results with artifacts

**When Code Execution is Selected:**
- Keywords: "compute", "calculate", "plot", "analyze", "generate graph", "statistical test"
- File uploads: CSV, HTML, XML files
- URLs with data files
- Requests involving data analysis or visualization

---

## Supported File Types

### CSV Files (`.csv`)
- **What it does**: Loads tabular data into pandas DataFrames
- **Features**:
  - Automatic encoding detection (UTF-8, Latin-1, etc.)
  - Delimiter detection (comma, tab, semicolon)
  - Handles large files with efficient memory usage
  - Preserves data types
- **Use Cases**: Gene expression data, clinical data, experimental results

### HTML Files (`.html`, `.htm`)
- **What it does**: Extracts tables from HTML using pandas `read_html`
- **Features**:
  - Multi-table extraction
  - Metadata extraction via BeautifulSoup
  - Handles complex nested structures
- **Use Cases**: Supplemental data from journal websites, HTML reports

### XML Files (`.xml`)
- **What it does**: Parses XML into normalized table structures
- **Features**:
  - Converts XML elements to records
  - Handles nested hierarchies
  - Preserves metadata
- **Use Cases**: Assay data, structured annotations, metadata files

### PDF Files (`.pdf`)
- **What it does**: Extracts tables and text from PDF documents
- **Features**:
  - Table extraction using pdfplumber/camelot
  - Text extraction for context
  - Handles multi-page documents
- **Use Cases**: Published paper results, supplementary data tables

### URLs
- **What it does**: Downloads and processes remote files
- **Features**:
  - Content-type detection
  - Automatic format routing
  - Provenance tracking (source URL recorded)
- **Use Cases**: Public datasets, shared research data, online repositories

### File Upload Endpoints

**1. `/upload_content`** (Supports CSV, HTML, XML, PDF)
```bash
POST /upload_content
Content-Type: multipart/form-data

files: @data.csv
question: "Compute correlation heatmap"
```

**2. `/query` with URLs**
```bash
POST /query
Content-Type: application/x-www-form-urlencoded

question: "Analyze this data"
urls: https://example.com/data.csv
context: {"resource": "code_exec"}
```

**3. `/query` with File Paths**
```bash
POST /query
Content-Type: application/x-www-form-urlencoded

question: "Generate PCA plot"
files: storage/data_files/csv/{content_id}.csv
context: {"resource": "code_exec"}
```

---

## Architecture and Components

### High-Level Architecture

```
User Query + Files/URLs
    ↓
Query Classifier (LLM)
    ↓
[code_exec selected]
    ↓
Code Execution Handler
    ├── Load Documents (loaders.py)
    ├── Profile Data (handler.py)
    ├── Plan with ReAct (LLM + prompts)
    ├── Execute Code (sandbox.py)
    ├── Collect Artifacts (artifacts.py)
    └── Return Results
```

### Component Overview

**Core Orchestrator: `handler.py`**
- `CodeExecutionHandler`: Main coordinator class
- `execute()`: Orchestrates the entire pipeline
- `plan_with_react()`: LLM-driven code generation
- `profile_data()`: Data schema and statistics extraction

**Data Loaders: `loaders.py`**
- `normalize_inputs()`: Routes files/URLs to appropriate loaders
- `load_csv()`, `load_html()`, `load_xml()`, `load_pdf()`, `load_url()`
- Returns standardized structure: `{tables: [...], texts: [...], metadata: {...}}`

**Secure Execution: `sandbox.py`**
- `run_python()`: Subprocess-based code execution
- `SandboxLimits`: Timeout, memory, network restrictions
- Blocks dangerous imports, captures stdout/stderr
- DataFrame serialization for safe data passing

**Helper Modules:**
- `plotting.py`: Pre-built plotting functions
- `stats.py`: Statistical test functions
- `bio_ops.py`: Bioinformatics-specific operations
- `transform.py`: Data cleaning and transformation utilities

**Artifact Management: `artifacts.py`**
- `ensure_run_dirs()`: Creates organized output directories
- `build_manifest()`: Generates run manifests with provenance
- `cleanup_expired()`: TTL-based artifact cleanup

---

## The Hybrid Approach

### What is the Hybrid Approach?

The Code Execution Agent uses a **Hybrid Approach** that combines:
1. **Pre-built Helper Functions** (plotting.py, stats.py, bio_ops.py, transform.py)
2. **LLM-Generated Custom Code** (dynamically generated Python code)

### How It Works

**Step 1: Helper Function Discovery**
The LLM prompt (`REACT_CODE_EXEC_PROMPT`) includes documentation of all available helper functions:
- Function signatures
- Usage examples
- Return value formats

**Step 2: LLM Decision Making**
When generating code, the LLM:
- **Checks** if a helper function exists for the task
- **Prioritizes** helper functions when appropriate
- **Falls back** to custom code when helpers don't fit

**Step 3: Code Generation**
```python
# Example: LLM chooses to use helper
from app.code_exec.plotting import save_correlation_heatmap
result = save_correlation_heatmap(df, path)

# Example: LLM chooses custom code
import matplotlib.pyplot as plt
# Custom visualization with specific requirements
```

**Step 4: Execution**
The sandbox allows imports from `app.code_exec.*` modules, making helpers accessible to generated code.

### Why the Hybrid Approach is Recommended

**1. Reliability**
- Helper functions are **tested and optimized**
- Handle edge cases (missing values, data types, etc.)
- Consistent output formats
- Production-ready code

**2. Flexibility**
- LLM can still write custom code for unique requirements
- Not restricted to only helper functions
- Can combine helpers with custom logic

**3. Best of Both Worlds**
- **Common tasks**: Use tested helpers (faster, more reliable)
- **Unique tasks**: Use LLM-generated custom code (flexible, adaptive)

**4. Maintainability**
- Helpers can be updated without changing LLM prompts extensively
- Custom code adapts to new requirements automatically
- Clear separation of concerns

**5. Performance**
- Helpers are optimized for common operations
- Custom code only when needed
- Reduces LLM token usage for standard tasks

### Example: Hybrid in Action

**User Query**: *"Create a correlation heatmap and then a custom scatter plot with error bars"*

**LLM-Generated Code:**
```python
# Step 1: Use helper for correlation heatmap
from app.code_exec.plotting import save_correlation_heatmap
save_correlation_heatmap(df, os.path.join(context['output_dir'], 'correlation.png'))

# Step 2: Custom code for scatter with error bars (helper doesn't support error bars)
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 6))
plt.errorbar(df['x'], df['y'], yerr=df['y_err'], fmt='o')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.savefig(os.path.join(context['output_dir'], 'scatter_with_errors.png'), dpi=200)
plt.close()
```

**Benefits:**
- Reliable correlation heatmap (helper)
- Flexible custom scatter plot (LLM-generated)
- Best approach for each task

---

## ReAct Framework Integration

### What is ReAct?

**ReAct (Reasoning + Acting)** is an iterative framework where the agent:
1. **Thinks** (reasons about the problem)
2. **Acts** (executes code)
3. **Observes** (sees results/errors)
4. **Corrects** (adjusts approach)
5. **Repeats** (until success or max attempts)

### Implementation in Code Execution Agent

**1. Planning Phase (ReAct: Think)**
```python
# In handler.py, plan_with_react()
- LLM receives: user instructions + data profile + helper function documentation
- LLM generates: reasoning + code plan + executable snippets
- Returns: {"steps": [...], "code_snippets": [...], "reasoning": "..."}
```

**2. Execution Phase (ReAct: Act)**
```python
# In handler.py, execute()
- Code snippets executed sequentially in sandbox
- Each snippet runs with full context (DataFrames, output_dir, etc.)
- Results captured (stdout, stderr, artifacts)
```

**3. Observation Phase (ReAct: Observe)**
```python
# Error detection
- Checks stderr for error indicators (Error, Traceback, Exception, etc.)
- Captures execution output
- Monitors for success/failure
```

**4. Correction Phase (ReAct: Correct)**
```python
# In handler.py, execute() - ReAct error correction loop
if execution_failed and attempt < max_retries:
    # Construct fix prompt
    fix_prompt = f"""
    The following Python code failed with an error.
    ORIGINAL CODE: {code}
    ERROR MESSAGE: {stderr}
    CONTEXT: {data_profile}
    
    Return ONLY the corrected Python code.
    """
    # LLM generates fixed code
    fixed_code = self.llm.generate(fix_prompt)
    # Retry with fixed code
    attempt += 1
```

### ReAct Error Correction Example

**Scenario**: User uploads CSV with encoding issues

```
Attempt 1:
Code: pd.read_csv('data.csv')
Error: UnicodeDecodeError: invalid start byte
ReAct: LLM analyzes error → "Need to detect encoding"

Attempt 2:
Code: 
  import chardet
  with open('data.csv', 'rb') as f:
      encoding = chardet.detect(f.read())['encoding']
  df = pd.read_csv('data.csv', encoding=encoding)
Error: Column 'GeneID' not found
ReAct: LLM analyzes → "Check actual column names from profile"

Attempt 3:
Code: 
  # Use column name from profile: 'gene_id'
  df = pd.read_csv('data.csv', encoding=encoding)
  df.rename(columns={'gene_id': 'GeneID'}, inplace=True)
Success: Data loaded, analysis proceeds
```

### Benefits of ReAct

1. **Self-Correcting**: Agent fixes its own errors automatically
2. **Adaptive**: Adjusts to unexpected data formats/issues
3. **Robust**: Handles edge cases without human intervention
4. **Transparent**: Each correction attempt is logged

---

## Complete Workflow

### End-to-End User Journey

**1. User Uploads File**
```
User → POST /upload_content
  files: @data.csv
  question: "Compute correlation and plot heatmap"
  
System → Saves file to storage/data_files/csv/{content_id}.csv
         Returns: {content_id, file_path, ...}
```

**2. User Queries with Code Execution Intent**
```
User → POST /query
  question: "Compute correlation and plot heatmap"
  context: {"resource": "code_exec"}
  files: storage/data_files/csv/{content_id}.csv
```

**3. Query Classification**
```
Classifier (LLM) → Analyzes query
  Input: "Compute correlation and plot heatmap"
  Output: query_type = "code_exec"
  
Router → Sends to code_exec_agent node
```

**4. Document Loading**
```
CodeExecutionHandler.load_documents()
  ├── Detects file type: .csv
  ├── Calls load_csv()
  ├── Auto-detects encoding/delimiter
  └── Returns: {tables: [DataFrame], texts: [], metadata: {}}
  
SocketIO → "Parsing documents..."
```

**5. Data Profiling**
```
CodeExecutionHandler.profile_data()
  ├── Analyzes DataFrame shape, dtypes
  ├── Counts missing values
  ├── Computes numeric summaries
  └── Returns: {profiles: [{shape, dtypes, missing, numeric_summary}]}
  
SocketIO → "Profiling data..."
```

**6. ReAct Planning**
```
CodeExecutionHandler.plan_with_react()
  ├── Constructs prompt with:
  │   ├── User instructions
  │   ├── Data profile (column names, types, stats)
  │   ├── Helper function documentation
  │   └── Code generation rules
  ├── Calls LLM.generate(prompt)
  ├── Parses JSON response
  └── Returns: {steps: [...], code_snippets: [...], reasoning: "..."}
  
SocketIO → "Planning analysis with AI..."
```

**7. Code Execution (with ReAct Error Correction)**
```
For each code_snippet:
  ├── Execute in sandbox
  │   ├── Serialize DataFrames to CSV
  │   ├── Inject context (output_dir, run_id, etc.)
  │   ├── Allow helper imports (app.code_exec.*)
  │   ├── Block dangerous imports
  │   └── Capture stdout/stderr
  │
  ├── Check for errors
  │   ├── If error found:
  │   │   ├── Call LLM to fix code (ReAct correction)
  │   │   ├── Retry execution (up to max_retries=3)
  │   │   └── Log attempts
  │   └── If success:
  │       └── Continue to next snippet
  │
  └── Collect outputs
  
SocketIO → "Executing step 1/3: Load data..."
          "Executing step 2/3: Compute correlation..."
          "Executing step 3/3: Generate heatmap..."
```

**8. Artifact Collection**
```
CodeExecutionHandler.execute()
  ├── Scans output directory for generated files
  │   ├── Figures: .png, .svg, .pdf
  │   ├── Tables: .csv, .html
  │   └── Logs: .txt
  ├── Builds artifact list
  └── Records in manifest
  
SocketIO → "Rendering artifacts..."
```

**9. Manifest Generation**
```
artifacts.build_manifest()
  ├── Creates run manifest:
  │   ├── run_id
  │   ├── inputs: {files, urls, instructions}
  │   ├── params: {timeout, memory}
  │   ├── artifacts: [{name, format, path, size}]
  │   └── environment: {llm, limits}
  └── Saves to tmp/artifacts/{run_id}/manifest.json
```

**10. Response Generation**
```
Return to user:
{
  "text": "Completed 3 step(s): ...\nGenerated 2 artifact(s): ...",
  "manifest": {...},
  "artifacts": [
    {"name": "correlation_heatmap", "format": "png", "path": "..."},
    {"name": "summary_stats", "format": "csv", "path": "..."}
  ],
  "resource": {"type": "code_exec", "id": "<run_id>"}
}
  
SocketIO → "Execution completed."
```

### Real-Time Progress Updates

The agent provides SocketIO updates throughout the workflow:
- "Parsing documents..."
- "Profiling data..."
- "Planning analysis with AI..."
- "Executing step 1/3: ..."
- "Step 1 had an error, asking AI to fix it..."
- "Rendering artifacts..."
- "Execution completed."

---

## Module Details

### `handler.py` - Core Orchestrator

**Class: `CodeExecutionHandler`**
- **Purpose**: Coordinates the entire code execution pipeline
- **Key Methods**:
  - `__init__(llm)`: Initializes with LLM instance (Gemini/OpenAI)
  - `execute()`: Main entry point, orchestrates full pipeline
  - `load_documents()`: Wraps loaders.normalize_inputs()
  - `profile_data()`: Extracts data schema and statistics
  - `plan_with_react()`: LLM-driven code generation
  - `run_in_sandbox()`: Executes code securely

**Responsibilities:**
- Document loading coordination
- Data profiling
- LLM interaction for code generation
- ReAct error correction loop
- Artifact collection and manifest creation
- User progress updates via SocketIO

### `loaders.py` - Data Ingestion

**Function: `normalize_inputs(files, urls)`**
- Routes files/URLs to appropriate loaders based on extension/content-type
- Returns standardized structure: `{tables: [...], texts: [...], metadata: {...}}`

**Loaders:**
- `load_csv()`: Encoding/delimiter detection, DataFrame creation
- `load_html()`: Table extraction via pandas.read_html, metadata via BeautifulSoup
- `load_xml()`: XML parsing to normalized records
- `load_pdf()`: Table and text extraction via pdfplumber/camelot
- `load_url()`: Content-type detection, routing to appropriate loader

**Returns:**
```python
{
  "tables": [
    {
      "dataframe": pd.DataFrame,
      "source": "file_path or url",
      "shape": [rows, cols],
      "columns": [...]
    }
  ],
  "texts": [...],  # Extracted text blocks
  "metadata": {...}  # File metadata
}
```

### `sandbox.py` - Secure Execution

**Function: `run_python(code, context, limits)`**
- Executes Python code in isolated subprocess
- **Security Features**:
  - Timeout enforcement (wall-time)
  - Memory limits (hints)
  - Network blocking (default)
  - Import restrictions (blocks socket, requests, etc.)
  - File access limited to output directory

**DataFrame Handling:**
- Serializes DataFrames to CSV before subprocess
- Reconstructs DataFrames within sandbox
- Makes helper modules importable via sys.path manipulation

**Allowed Libraries:**
- pandas, numpy, matplotlib, seaborn
- scipy, statsmodels (for statistics)
- json, os (with restrictions)
- app.code_exec.* (helper modules)

**Blocked Operations:**
- Network access (socket, requests, urllib)
- System operations (subprocess, shutil)
- Dangerous file operations

### `plotting.py` - Visualization Helpers

**Available Functions:**
- `save_histogram(df, column, path, bins=30)`: Histogram plots
- `save_scatter(df, x, y, path)`: Scatter plots
- `save_correlation_heatmap(df, path)`: Correlation matrix heatmap
- `save_boxplot(df, column, by, path)`: Box/violin plots
- `save_timeseries(df, x, y, path)`: Time-series line charts
- `save_pca_preview(df, path, n_components=2)`: PCA visualization

**Features:**
- Publication-ready styling
- Automatic figure sizing and DPI
- Agg backend (no GUI required)
- Proper figure cleanup (plt.close())

### `stats.py` - Statistical Analysis Helpers

**Available Functions:**
- `t_test_independent()`: Independent samples t-test
- `mann_whitney()`: Mann-Whitney U test (non-parametric)
- `anova_oneway()`: One-way ANOVA
- `kruskal_wallis()`: Kruskal-Wallis test
- `fdr_bh()`: Benjamini-Hochberg FDR correction
- `correlations()`: Correlation matrix computation
- `linear_regression()`: Linear regression modeling

**Features:**
- Proper statistical test implementations
- FDR correction for multiple testing
- Consistent return formats

### `bio_ops.py` - Bioinformatics Helpers

**Available Functions:**
- `differential_like()`: Differential expression analysis with volcano plot components
- `ora_enrichment()`: Over-Representation Analysis (ORA) for gene sets
- `gene_id_map()`: Gene symbol to ID mapping
- `dose_response_ic50()`: 4-parameter logistic curve fitting and IC50 estimation

**Features:**
- Domain-specific analyses
- Standard bioinformatics workflows
- Publication-ready outputs

### `transform.py` - Data Transformation Helpers

**Available Functions:**
- `missingness_report()`: Missing value analysis
- `drop_duplicates()`: Duplicate removal
- `join_tables()`: DataFrame merging
- `scale_columns()`: Normalization/scaling (standard, minmax)
- `unit_convert()`: Unit conversion

**Features:**
- Data cleaning utilities
- Standardization operations
- Consistent DataFrame handling

### `artifacts.py` - Artifact Management

**Functions:**
- `ensure_run_dirs(root, run_id)`: Creates organized directory structure
  - `{root}/{run_id}/figures/`
  - `{root}/{run_id}/tables/`
  - `{root}/{run_id}/logs/`
- `build_manifest()`: Creates run manifest with provenance
- `cleanup_expired()`: TTL-based cleanup of old artifacts

**Artifact Structure:**
```python
{
  "id": "run_id",
  "created_at": "timestamp",
  "inputs": {"files": [...], "urls": [...], "instructions": "..."},
  "params": {"timeout": 60, "max_memory_mb": 1024},
  "artifacts": [
    {
      "id": "artifact_id",
      "type": "figure|table|log",
      "name": "plot_name",
      "format": "png|csv|html",
      "path_or_url": "file_path",
      "bytes_size": 12345,
      "metadata": {...}
    }
  ],
  "environment": {"llm": "Gemini", "limits": {...}}
}
```

---

## Configuration

### Environment Variables

**`CODE_EXEC_TIMEOUT`** (default: 60 seconds)
- Maximum execution time per code snippet
- Prevents infinite loops
- Configurable per request via `options`

**`CODE_EXEC_MAX_MEM_MB`** (default: 1024 MB)
- Memory limit hint for sandbox execution
- Helps prevent memory exhaustion
- Configurable per request

**`CODE_EXEC_ALLOW_NETWORK`** (default: false)
- Controls network access during execution
- Should remain false for security
- Only enable if needed for specific use cases

**`ARTIFACT_TTL_MINUTES`** (default: 120 minutes)
- Time-to-live for artifacts
- Old artifacts automatically cleaned up
- Prevents disk space issues

### Configuration via API

**Request Options:**
```json
{
  "timeout_seconds": 120,
  "max_memory_mb": 2048,
  "allow_network": false,
  "output_formats": ["png", "svg"]
}
```

**Example:**
```bash
curl -X POST http://localhost:5002/query \
  -d "question=Analyze data" \
  -d "context={\"resource\":\"code_exec\"}" \
  -d "options={\"timeout_seconds\":120,\"max_memory_mb\":2048}" \
  -d "files=data.csv"
```

---

## Testing the Code Execution Agent

### Prerequisites

1. **Service Running**: AI Assistant must be running (via Docker or local)
2. **Authentication Token**: Valid Bearer token for API requests
3. **Test Data**: Sample CSV/HTML/XML/PDF files or URLs
4. **Testing Tool**: Postman, curl, or HTTP client

### Testing Methods

#### Method 1: Postman Testing

**1. Setup Postman Collection**

Create a new collection: "Code Execution Agent Tests"

**2. Environment Variables**
```
base_url: http://localhost:5002
token: YOUR_BEARER_TOKEN
```

**3. Test Cases**

**Test 1: Upload CSV File**
```
POST {{base_url}}/upload_content
Headers:
  Authorization: Bearer {{token}}
Body (form-data):
  files: [Select file: data.csv]
  question: "Compute correlation heatmap"
```

**Test 2: Query with Uploaded File**
```
POST {{base_url}}/query
Headers:
  Authorization: Bearer {{token}}
  Content-Type: application/x-www-form-urlencoded
Body:
  question: "Generate correlation heatmap and PCA plot"
  context: {"resource":"code_exec"}
  files: storage/data_files/csv/{content_id}.csv
```

**Test 3: Query with URL**
```
POST {{base_url}}/query
Headers:
  Authorization: Bearer {{token}}
  Content-Type: application/x-www-form-urlencoded
Body:
  question: "Analyze this dataset and compute summary statistics"
  context: {"resource":"code_exec"}
  urls: https://example.com/data.csv
```

**Test 4: Statistical Analysis**
```
POST {{base_url}}/query
Headers:
  Authorization: Bearer {{token}}
Body (form-data):
  files: [Select file: clinical_data.csv]
  question: "Perform t-test between treatment and control groups"
  context: {"resource":"code_exec"}
```

#### Method 2: curl Testing

**Test 1: Upload and Analyze**
```bash
# Step 1: Upload CSV
curl -X POST http://localhost:5002/upload_content \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@test_data.csv" \
  -F "question=Compute correlation heatmap"

# Step 2: Use returned file_path in query
curl -X POST http://localhost:5002/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=Plot histogram for all numeric columns" \
  -d "context={\"resource\":\"code_exec\"}" \
  -d "files=storage/data_files/csv/{content_id}.csv"
```

**Test 2: Direct URL Analysis**
```bash
curl -X POST http://localhost:5002/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=Extract tables from PDF and create bar chart" \
  -d "context={\"resource\":\"code_exec\"}" \
  -d "urls=https://example.com/results.pdf"
```

**Test 3: Statistical Test**
```bash
curl -X POST http://localhost:5002/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=Perform ANOVA test on expression data grouped by condition" \
  -d "context={\"resource\":\"code_exec\"}" \
  -d "files=storage/data_files/csv/expression_data.csv"
```

#### Method 3: Python Testing Script

```python
import requests
import json

BASE_URL = "http://localhost:5002"
TOKEN = "YOUR_BEARER_TOKEN"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/x-www-form-urlencoded"
}

# Test 1: Upload file
with open("test_data.csv", "rb") as f:
    files = {"files": f}
    data = {"question": "Compute correlation heatmap"}
    response = requests.post(
        f"{BASE_URL}/upload_content",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data=data
    )
    result = response.json()
    file_path = result.get("file_path")
    print(f"File uploaded: {file_path}")

# Test 2: Analyze uploaded file
query_data = {
    "question": "Generate PCA plot and correlation heatmap",
    "context": json.dumps({"resource": "code_exec"}),
    "files": file_path
}
response = requests.post(
    f"{BASE_URL}/query",
    headers=headers,
    data=query_data
)
result = response.json()
print(json.dumps(result, indent=2))
```

### Expected Response Format

**Success Response:**
```json
{
  "response": {
    "text": "Completed 3 step(s): ...\nGenerated 2 artifact(s): ...",
    "manifest": {
      "id": "run_id",
      "inputs": {...},
      "artifacts": [...]
    },
    "artifacts": [
      {
        "name": "correlation_heatmap",
        "format": "png",
        "path": "tmp/artifacts/{run_id}/figures/correlation_heatmap.png"
      }
    ],
    "resource": {
      "type": "code_exec",
      "id": "run_id"
    }
  }
}
```

**Error Response:**
```json
{
  "response": {
    "text": "Error during code execution: ...",
    "error": "Error message",
    "resource": {
      "type": "code_exec",
      "id": "run_id"
    }
  }
}
```

### Testing Checklist

- [ ] CSV file upload and analysis
- [ ] HTML file upload and table extraction
- [ ] XML file processing
- [ ] PDF table extraction
- [ ] URL-based data loading
- [ ] Correlation analysis and visualization
- [ ] Statistical test execution (t-test, ANOVA)
- [ ] PCA visualization
- [ ] Bioinformatics operations (differential analysis)
- [ ] Error handling (invalid file, wrong column names)
- [ ] ReAct error correction (test with problematic data)
- [ ] Artifact generation and retrieval
- [ ] Multi-step analysis workflows

---

## Usage Examples

### Example 1: Basic CSV Analysis

**User Query**: *"Load this CSV and compute summary statistics"*

**Upload:**
```bash
POST /upload_content
files: @gene_expression.csv
```

**Query:**
```bash
POST /query
question: "Load this CSV and compute summary statistics"
context: {"resource":"code_exec"}
files: storage/data_files/csv/{content_id}.csv
```

**Generated Code (LLM):**
```python
import pandas as pd
import numpy as np

# Load data
df = context['tables'][0]['dataframe']
print(f"Data shape: {df.shape}")
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
```

**Result**: Text summary with statistics printed to stdout

### Example 2: Visualization

**User Query**: *"Create a correlation heatmap and scatter plot"*

**Generated Code (LLM uses hybrid approach):**
```python
import os
from app.code_exec.plotting import save_correlation_heatmap, save_scatter

df = context['tables'][0]['dataframe']
output_dir = context['output_dir']

# Use helper for correlation heatmap
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
result = save_correlation_heatmap(df, heatmap_path)
print(f"Heatmap saved: {result['path']}")

# Use helper for scatter plot
scatter_path = os.path.join(output_dir, 'scatter_plot.png')
result = save_scatter(df, 'gene_A', 'gene_B', scatter_path)
print(f"Scatter plot saved: {result['path']}")
```

**Result**: Two PNG files generated in artifacts directory

### Example 3: Statistical Test

**User Query**: *"Perform t-test between treatment and control groups on expression values"*

**Generated Code (LLM):**
```python
from app.code_exec.stats import t_test_independent

df = context['tables'][0]['dataframe']

# Perform t-test
result = t_test_independent(
    df=df,
    group_col='condition',
    value_col='expression',
    group_a='treatment',
    group_b='control'
)

print(f"T-test Results:")
print(f"  t-statistic: {result['t']:.4f}")
print(f"  p-value: {result['p']:.4f}")
print(f"  Group A (n={result['n_a']}): treatment")
print(f"  Group B (n={result['n_b']}): control")

if result['p'] < 0.05:
    print("\nResult: Statistically significant difference (p < 0.05)")
else:
    print("\nResult: No significant difference (p >= 0.05)")
```

**Result**: Statistical test results with interpretation

### Example 4: Multi-Step Analysis

**User Query**: *"Load data, compute differential expression, and create a volcano plot"*

**Generated Code (LLM - multiple steps):**
```python
# Step 1: Load and inspect
df = context['tables'][0]['dataframe']
print(f"Loaded {df.shape[0]} genes, {df.shape[1]} samples")

# Step 2: Differential expression analysis
from app.code_exec.bio_ops import differential_like

de_results = differential_like(
    df=df,
    group_col='condition',
    value_cols=['gene_1', 'gene_2', 'gene_3'],  # Example genes
    group_a='treatment',
    group_b='control',
    method='ttest',
    fdr_alpha=0.05
)

# Step 3: Create volcano plot (custom code - helper doesn't exist)
import matplotlib.pyplot as plt
import numpy as np

de_df = de_results['dataframe']
significant = de_df[de_df['significant'] == True]

plt.figure(figsize=(10, 8))
plt.scatter(de_df['log2FC'], -np.log10(de_df['p_value']), 
           alpha=0.5, c='gray', label='Not significant')
plt.scatter(significant['log2FC'], -np.log10(significant['p_value']),
           alpha=0.7, c='red', label='Significant (FDR < 0.05)')
plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', label='p=0.05')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 p-value')
plt.title('Volcano Plot - Differential Expression')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(context['output_dir'], 'volcano_plot.png'), dpi=300)
plt.close()

print(f"\nDifferential Expression Results:")
print(f"  Total genes tested: {len(de_df)}")
print(f"  Significant genes (FDR < 0.05): {len(significant)}")
```

**Result**: Differential analysis results + volcano plot visualization

### Example 5: PDF Table Extraction

**User Query**: *"Extract the results table from this PDF and create a bar chart"*

**Upload:**
```bash
POST /upload_content
files: @paper_results.pdf
```

**Query:**
```bash
POST /query
question: "Extract the results table from this PDF and create a bar chart"
context: {"resource":"code_exec"}
files: storage/pdfs/{content_id}.pdf
```

**Generated Code (LLM):**
```python
# Data already loaded from PDF via loaders
df = context['tables'][0]['dataframe']  # First table extracted from PDF

print(f"Extracted table from PDF:")
print(df.head())

# Create bar chart
import matplotlib.pyplot as plt
import os

if 'value' in df.columns and 'category' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(df['category'], df['value'])
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Results from PDF')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(context['output_dir'], 'bar_chart.png'), dpi=200)
    plt.close()
    print("Bar chart created successfully")
```

**Result**: Table extracted from PDF + bar chart visualization

---

## Safety and Security

### Sandbox Isolation

**Subprocess Execution:**
- Code runs in isolated subprocess (not in main process)
- Prevents crashes from affecting the main application
- Isolated memory space

**Time Limits:**
- Wall-time timeout prevents infinite loops
- Configurable per request (default: 60 seconds)
- Automatic termination if exceeded

**Memory Limits:**
- Memory cap prevents resource exhaustion
- Configurable per request (default: 1024 MB)
- System-level limits can be enforced

### Import Restrictions

**Blocked Imports:**
```python
# Network access
socket, requests, urllib, ftplib, http

# System operations
subprocess, shutil, os.system

# File system (outside sandbox)
open() restricted to output_dir only
```

**Allowed Imports:**
```python
# Data science
pandas, numpy, matplotlib, seaborn
scipy, statsmodels

# Helper modules
app.code_exec.plotting
app.code_exec.stats
app.code_exec.bio_ops
app.code_exec.transform
```

### File System Restrictions

**Safe File Operations:**
- Code can only write to designated `output_dir`
- `safe_open()` wrapper ensures writes stay within bounds
- Read access limited to context-provided files

**Protected Directories:**
- Main application code
- System directories
- Other users' data
- Configuration files

### Network Restrictions

**Default: No Network Access**
- Prevents data exfiltration
- Prevents downloading malicious code
- Prevents external API calls (unless explicitly allowed)

**If Network Allowed:**
- Only enable for trusted use cases
- Monitor all network activity
- Log all external connections

### Best Practices

1. **Validate Inputs**: Check file types, sizes before processing
2. **Sample Large Data**: Don't process extremely large files directly
3. **Monitor Resource Usage**: Track execution time, memory consumption
4. **Log Everything**: Record all code execution attempts
5. **Clean Up Artifacts**: Use TTL-based cleanup to prevent disk issues

---

## Error Handling and Self-Correction

### Error Detection

**Automatic Detection:**
The system checks `stderr` for common error indicators:
- `Error`, `Exception`, `Traceback`
- Specific Python exceptions: `NameError`, `AttributeError`, `KeyError`, `ValueError`, `TypeError`

**Error Types Handled:**

1. **Encoding Errors**
   - Problem: `UnicodeDecodeError`
   - Solution: LLM generates code with encoding detection

2. **Column Name Errors**
   - Problem: `KeyError` - column not found
   - Solution: LLM checks data profile, uses correct column names

3. **Data Type Errors**
   - Problem: `TypeError` - wrong data type
   - Solution: LLM adds type conversion code

4. **Missing Library Errors**
   - Problem: `ImportError`
   - Solution: LLM uses alternative approach or helper functions

5. **Logic Errors**
   - Problem: Incorrect calculations, wrong functions
   - Solution: LLM analyzes error, corrects logic

### ReAct Error Correction Flow

```
Execute Code
    ↓
[Error Detected?]
    ↓ Yes
Construct Fix Prompt
    ├── Original code
    ├── Error message
    └── Data context
    ↓
Call LLM for Fix
    ↓
Extract Fixed Code
    ↓
Retry Execution (up to max_retries=3)
    ↓
[Success?]
    ↓ No → Log final error
    ↓ Yes
Continue to Next Step
```

### Example: Self-Correction in Action

**Initial Code (with error):**
```python
df = pd.read_csv('data.csv')
corr = df.corr()  # Error if non-numeric columns exist
```

**Error:**
```
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

**LLM Fix (Automatic):**
```python
df = pd.read_csv('data.csv')
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print(f"Computed correlation for {len(numeric_df.columns)} numeric columns")
```

**Result**: Code corrected and executed successfully

---

## Artifact Management

### Artifact Structure

```
tmp/artifacts/
  └── {run_id}/
      ├── manifest.json       # Run metadata and provenance
      ├── figures/
      │   ├── correlation_heatmap.png
      │   ├── pca_plot.png
      │   └── volcano_plot.png
      ├── tables/
      │   ├── summary_stats.csv
      │   └── de_results.csv
      └── logs/
          └── execution.log
```

### Manifest Schema

```json
{
  "id": "abc123...",
  "created_at": "2024-01-15T10:30:00Z",
  "inputs": {
    "files": ["storage/data_files/csv/data.csv"],
    "urls": [],
    "instructions": "Compute correlation heatmap"
  },
  "params": {
    "timeout": 60,
    "max_memory_mb": 1024
  },
  "artifacts": [
    {
      "id": "art1",
      "type": "figure",
      "name": "correlation_heatmap",
      "format": "png",
      "path_or_url": "tmp/artifacts/{run_id}/figures/correlation_heatmap.png",
      "bytes_size": 45678,
      "metadata": {
        "step": "generated",
        "filename": "correlation_heatmap.png"
      }
    }
  ],
  "environment": {
    "llm": "Gemini",
    "limits": {
      "timeout": 60,
      "max_memory_mb": 1024
    }
  }
}
```

### TTL Cleanup

**Automatic Cleanup:**
- Old artifacts automatically deleted after TTL expires
- Default TTL: 120 minutes (configurable)
- Prevents disk space issues
- Runs periodically via `cleanup_expired()`

**Manual Cleanup:**
```python
from app.code_exec.artifacts import cleanup_expired

# Clean artifacts older than 60 minutes
cleanup_expired(root="tmp/artifacts", ttl_minutes=60)
```

---

## Comparison with Other Services

### Code Execution vs. RAG

| Aspect | Code Execution | RAG |
|-------|----------------|-----|
| **Purpose** | Execute code, analyze data | Retrieve and search documents |
| **Input** | CSV/HTML/XML/PDF/URL (data files) | PDFs, text documents |
| **Processing** | Code generation → execution | Vector search → retrieval |
| **Output** | Plots, statistics, analysis results | Text summaries, relevant passages |
| **LLM Usage** | Code generation | Answer synthesis |
| **File Types** | Data files (CSV, HTML, XML) + PDFs | PDFs (for indexing) |
| **Use Case** | "Compute correlation from this CSV" | "What does this PDF say about gene X?" |

**When to Use:**
- **Code Execution**: When you need to **analyze or compute** something from data
- **RAG**: When you need to **find information** from documents

### Code Execution vs. Annotation

| Aspect | Code Execution | Annotation |
|-------|----------------|------------|
| **Purpose** | Data analysis and computation | Knowledge graph querying |
| **Input** | Raw data files | Gene/protein/variant queries |
| **Processing** | Python code execution | Neo4j graph traversal |
| **Output** | Analysis results, plots | Graph query results, annotations |
| **Use Case** | "Plot expression data" | "What are the known interactions of gene X?" |

**When to Use:**
- **Code Execution**: For **data analysis** and **computations**
- **Annotation**: For **querying existing knowledge graphs**

### Code Execution vs. Galaxy

| Aspect | Code Execution | Galaxy |
|-------|----------------|--------|
| **Purpose** | In-app code execution | Galaxy workflow integration |
| **Input** | Uploaded files, URLs | Galaxy tool/workflow requests |
| **Processing** | LLM-generated Python code | Galaxy API calls |
| **Output** | Plots, statistics | Galaxy job results |
| **Use Case** | "Analyze this CSV now" | "Run Galaxy workflow X" |

**When to Use:**
- **Code Execution**: For **quick analysis** within the AI Assistant
- **Galaxy**: For **complex workflows** requiring Galaxy tools

### Code Execution vs. Hypothesis

| Aspect | Code Execution | Hypothesis |
|-------|----------------|------------|
| **Purpose** | Data analysis | Hypothesis generation |
| **Input** | Data files | Variants, phenotypes |
| **Processing** | Code execution | Graph construction, reasoning |
| **Output** | Analysis results | Hypothesis graphs |
| **Use Case** | "Test this hypothesis with data" | "Generate hypothesis from variants" |

**When to Use:**
- **Code Execution**: For **testing hypotheses** with data
- **Hypothesis**: For **generating hypotheses** from variants

### Complementary Services

These services work together:
1. **RAG** finds relevant papers → **Code Execution** analyzes supplemental data
2. **Annotation** finds gene interactions → **Code Execution** tests with expression data
3. **Hypothesis** generates hypothesis → **Code Execution** tests it with experimental data

---

## Additional Information

### LLM Integration

**Supported LLMs:**
- Gemini (Google)
- OpenAI (GPT-4, GPT-3.5)
- Any LLM implementing `LLMInterface`

**LLM Responsibilities:**
1. **Query Classification**: Determines if query is `code_exec` intent
2. **Code Generation**: Creates Python code from user instructions
3. **Error Fixing**: Corrects code errors via ReAct

**LLM Prompt Structure:**
```
1. System prompt (REACT_CODE_EXEC_PROMPT)
2. Helper function documentation
3. User instructions
4. Data profile (column names, types, statistics)
5. Code generation rules
```

### SocketIO Integration

**Real-Time Updates:**
- Progress messages sent via SocketIO
- User sees live updates during execution
- Error states communicated in real-time

**Update Messages:**
- "Parsing documents..."
- "Profiling data..."
- "Planning analysis with AI..."
- "Executing step 1/3: ..."
- "Step 1 had an error, asking AI to fix it..."
- "Rendering artifacts..."
- "Execution completed."

### Future Enhancements

**Planned Features:**
- Multi-file support (currently processes one file at a time)
- More helper functions (additional plots, statistics)
- Interactive plots (Plotly integration)
- Result caching for repeated queries
- Execution history and replay

---

## Troubleshooting

### Common Issues

**Issue 1: "No code was generated"**
- **Cause**: LLM didn't understand the request
- **Solution**: Rephrase query, be more specific

**Issue 2: "Column not found"**
- **Cause**: Column name mismatch
- **Solution**: ReAct framework should auto-correct, but check data profile first

**Issue 3: "Timeout exceeded"**
- **Cause**: Code execution too slow
- **Solution**: Increase timeout via `options`, or simplify query

**Issue 4: "Import error"**
- **Cause**: Trying to import blocked module
- **Solution**: Use allowed libraries or helper functions

**Issue 5: "File not found"**
- **Cause**: Incorrect file path
- **Solution**: Verify file was uploaded successfully, use correct path from upload response

### Debugging Tips

1. **Check Logs**: Review application logs for detailed error messages
2. **Inspect Artifacts**: Check generated code in artifacts directory
3. **Verify Data Profile**: Ensure data loaded correctly before code generation
4. **Test Helper Functions**: Test helper functions independently if issues persist
5. **Simplify Query**: Break complex requests into simpler steps

---

## Conclusion

The Code Execution Agent provides a powerful, flexible, and secure way to perform data analysis directly within the AI Assistant. By combining:

- **LLM Intelligence**: Understanding user intent and generating appropriate code
- **Helper Functions**: Reliable, tested functions for common tasks
- **ReAct Framework**: Self-correcting error handling
- **Secure Sandbox**: Safe code execution with resource limits
- **Multi-Format Support**: Handling diverse data types



