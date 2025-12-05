## Code Execution Agent – Overview and Testing Guide

### What it does (short)
- Executes Python for data analysis using PythonREPLTool (LangChain) driven by your LLM.
- Loads user-provided data (CSV, HTML, XML, PDF, URL) and performs stats, transforms, and visualizations.
- Saves generated artifacts (e.g., plots) and returns inline base64 previews plus file paths.
- Handles rate limits with retry, iteration/time caps, and structured responses.

### Hybrid approach (LLM + helper functions)
- The agent can both generate and execute arbitrary Python and import first-class helpers:
  - `app.code_exec.plotting` (visualizations)
  - `app.code_exec.stats` (statistics)
  - `app.code_exec.transform` (data cleaning/transform)
  - `app.code_exec.bio_ops` (bioinformatics)
- The prompt encourages using helpers when they fit, and falling back to custom code when needed.
- `sys.path` is set so `app.code_exec.*` modules are importable inside the REPL.

### Supported inputs
- CSV, HTML, XML: via `/upload_content` (multipart form-data)
- PDF: default to RAG; use `context {"resource":"code_exec"}` to analyze with code_exec
- URL: via `/query` with `context {"resource":"code_exec"}`

### Routing rules (no conflict with RAG)
- PDFs
  - Default: RAG (semantic Q&A) – `resource="content"`
  - Analysis/plots: CodeExec – `resource="code_exec"` or analysis keywords
- CSV/HTML/XML
  - CodeExec triggers if `resource="code_exec"` or the question contains analysis keywords (compute, analyze, plot, mean, sum, chart, correlation, summary, etc.)
- URL
  - CodeExec only when `resource="code_exec"` is set

> Note on implicit routing
> - If a user uploads a CSV/HTML/XML and asks, for example, “Compute summary statistics and plot a bar chart of means” without setting `context`, the keyword match automatically routes to CodeExec.
> - PDFs default to RAG, but similar analysis keywords will route to CodeExec unless you explicitly want RAG behavior.
> - URLs require `context {"resource":"code_exec"}` to use CodeExec.

### Key endpoints
- POST `/upload_content` (files)
- POST `/query` (general; can call content QA or code_exec depending on `context.resource`)
- GET `/user_status/documents/` (list content and `content_id`)

### Auth
- All requests require header: `Authorization: Bearer YOUR_TOKEN`

### Options
Optional JSON to tune execution limits (safeguards are applied by default):
```json
{
  "timeout_seconds": 120,
  "max_memory_mb": 2048,
  "max_iterations": 20
}
```

---

## Postman examples by input type

### 1) CSV → CodeExec
Compute summary statistics and plot a bar chart.
Postman setup:
- Method: POST
- URL: http://localhost:5002/upload_content
- Headers:
  - Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Compute summary statistics and plot a bar chart of means.
  - context: {"resource":"code_exec"}
  - files: [choose file] data.csv
  - options: {"timeout_seconds":120,"max_memory_mb":2048,"max_iterations":20}
Expected response (shape):
```json
{
  "results": [
    {
      "filename": "data.csv",
      "response": { "resource": { "content_id": "...", "file_path": "storage/data_files/csv/...csv" }, "text": "CSV file uploaded successfully." },
      "answer": {
        "text": "...summary and status...",
        "outputs": [{ "stdout": "...", "stderr": "", "step": 1 }],
        "artifacts": [
          {
            "name": "mean_values_bar_chart",
            "format": "png",
            "path": "tmp/artifacts/<run_id>/figures/mean_values_bar_chart.png",
            "data": "data:image/png;base64,..."
          }
        ]
      }
    }
  ]
}
```

### 2) HTML → CodeExec
Extract tables and plot means.
Postman setup:
- Method: POST
- URL: http://localhost:5002/upload_content
- Headers: Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Extract all tables and plot a bar chart of mean numeric columns.
  - context: {"resource":"code_exec"}
  - files: [choose file] page.html

### 3) XML → CodeExec
Parse records and compute summary stats.
Postman setup:
- Method: POST
- URL: http://localhost:5002/upload_content
- Headers: Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Parse records from this XML and summarize numeric fields.
  - context: {"resource":"code_exec"}
  - files: [choose file] data.xml

### 4) PDF → CodeExec (analysis)
Extract tables and compute means (forces code_exec; avoids RAG).
Postman setup:
- Method: POST
- URL: http://localhost:5002/upload_content
- Headers: Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Extract tables and compute mean of numeric columns; plot a bar chart.
  - context: {"resource":"code_exec"}
  - files: [choose file] report.pdf

### 5) PDF → RAG (semantic QA)
Summarize without code execution (default behavior).
Postman setup (upload + ask):
- Method: POST
- URL: http://localhost:5002/upload_content
- Headers: Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Summarize the Results section in 5 bullet points.
  - files: [choose file] report.pdf

Optional follow-up using content_id:
1) Find the PDF content_id
   - Method: GET
   - URL: http://localhost:5002/user_status/documents/
   - Headers: Authorization: Bearer YOUR_TOKEN
2) Ask a question targeting that content_id
   - Method: POST
   - URL: http://localhost:5002/query
   - Headers: Authorization: Bearer YOUR_TOKEN
   - Body (form-data):
     - question: What conclusions were drawn about treatment A?
     - context: {"resource":"content","id":"<CONTENT_ID>"}

### 6) URL → CodeExec
Load tables from a web page and analyze.
Postman setup:
- Method: POST
- URL: http://localhost:5002/query
- Headers: Authorization: Bearer YOUR_TOKEN
- Body (form-data):
  - question: Extract all tables from this webpage and compute mean of numeric columns. Plot a bar chart of the top 10 countries by population.
  - context: {"resource":"code_exec"}
  - urls: https://en.wikipedia.org/wiki/List_of_countries_by_population

---

## Endpoints at a glance

- **POST `/upload_content`** (files: PDF, CSV, HTML, XML)
- **POST `/query`** (URLs or general questions)
- **Auth header:** `Authorization: Bearer <token>`

## What the agent does (behavior summary)

1. **Normalizes inputs** (CSV/HTML/XML/PDF/URL)
   - Detects file type and loads using appropriate loader
   - Extracts tables where possible
   - Handles encoding and delimiters automatically

2. **Extracts tables where possible; saves as CSV under `tmp/artifacts/<run_id>/figures/`**
   - Tables from HTML/XML are converted to pandas DataFrames
   - PDF tables extracted using `pdfplumber` or `camelot`
   - Each table saved as `data_0.csv`, `data_1.csv`, etc.

3. **Builds an analysis prompt with:**
   - **Extracted table paths** (CSV files ready to load)
   - **Original file/URL paths** (for PDF/URL if tables not extracted)
   - **Data profiles** (shape, missingness, dtypes, basic stats)

4. **Executes analysis using PythonREPLTool; returns text + artifacts (charts/base64)**
   - LLM generates Python code based on user request
   - Code executes safely with timeout and memory limits
   - Generated plots saved and returned as base64-encoded images

## Supported libraries and constraints

### Available Libraries
- **pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn**
- **pdfplumber, camelot, xmltodict, lxml, chardet**

### Strict Constraints
- **No runtime package installation** (pip/subprocess not allowed)
- **Use loaders:** `load_pdf`, `load_html`, `load_xml`, `load_url` from `app.code_exec.loaders`
- Agent will receive instructions to use only pre-installed libraries

## Helper functions (reference)

### Plotting Helpers (`app.code_exec.plotting`)
- `save_histogram(df, column, path, bins=30)` - Create histogram plots
- `save_scatter(df, x, y, path)` - Create scatter plots
- `save_correlation_heatmap(df, path)` - Generate correlation matrix heatmaps
- `save_boxplot(df, column, by, path)` - Create boxplots (grouped or single)
- `save_timeseries(df, x, y, path)` - Plot time-series line charts
- `save_pca_preview(df, path, n_components=2)` - Generate PCA visualizations

### Statistical Helpers (`app.code_exec.stats`)
- `t_test_independent(df, group_col, value_col, group_a, group_b)` - Independent samples t-test
- `mann_whitney(df, group_col, value_col, group_a, group_b)` - Mann-Whitney U test
- `anova_oneway(df, group_col, value_col)` - One-way ANOVA
- `kruskal_wallis(df, group_col, value_col)` - Kruskal-Wallis test
- `fdr_bh(p_values, alpha=0.05)` - Benjamini-Hochberg FDR correction
- `correlations(df, cols=None, method="pearson")` - Compute correlation matrix
- `linear_regression(df, y, X)` - Fit linear regression model

### Bioinformatics Helpers (`app.code_exec.bio_ops`)
- `differential_like(df, group_col, value_cols, group_a, group_b, method="ttest", fdr_alpha=0.05)` - Differential expression analysis
- `ora_enrichment(gene_set, universe, term_map)` - Over-Representation Analysis
- `gene_id_map(symbols, mapping)` - Map gene symbols to IDs
- `dose_response_ic50(df, concentration_col, response_col)` - Fit 4-parameter logistic curve and estimate IC50

### Transformation Helpers (`app.code_exec.transform`)
- `missingness_report(df)` - Count missing values per column
- `drop_duplicates(df, subset=None)` - Remove duplicate rows
- `join_tables(left, right, on, how="inner")` - Merge two DataFrames
- `scale_columns(df, cols, method="standard")` - Normalize columns (z-score or minmax)
- `unit_convert(df, col, factor)` - Convert units by multiplying column by factor

## Output format explained

The agent returns a structured response with the following fields:

- **`text`**: Human-readable summary of the analysis results
- **`outputs`**: List with stdout/stderr per step
  ```json
  [{"step": 1, "description": "Code execution", "stdout": "...", "stderr": ""}]
  ```
- **`artifacts`**: Figures with `name`, `format`, `path`, base64 preview when small (<10MB)
  ```json
  [{
    "name": "bar_chart",
    "format": "png",
    "path": "tmp/artifacts/<run_id>/figures/bar_chart.png",
    "data": "data:image/png;base64,...",
    "size_bytes": 12345
  }]
  ```
- **`manifest`**: Run metadata and environment limits
  ```json
  {
    "run_id": "...",
    "inputs": {"files": [], "urls": [], "instructions": "..."},
    "params": {"timeout": 120, "max_memory_mb": 2048},
    "artifacts": [...],
    "environment": {"limits": {...}, "llm": "..."}
  }
  ```
- **`resource`**: `{ "type": "code_exec", "id": "<run_id>" }`

## Timeouts and performance tips

### Default Limits
- `timeout_seconds=120` (2 minutes)
- `max_memory_mb=2048` (2 GB)
- `max_iterations=20` (agent iteration limit)

### Performance Optimization Tips
- **Use smaller samples for large tables** - Agent can sample data if needed
- **Avoid expensive global operations** - Prefer vectorized pandas operations
- **Generate fewer/lighter plots** - Limit number of visualizations per request
- **Prefer PNG format** - More efficient than SVG/PDF for web display

### Adjusting Limits
Pass custom limits via `options` parameter:
```json
{
  "timeout_seconds": 180,
  "max_memory_mb": 4096,
  "max_iterations": 30
}
```

## Security and safety

### Network Access
- **No internet for user code** unless explicitly allowed (policy dependent)
- Agent runs in isolated environment with restricted network access

### File System Access
- **File writes restricted to `tmp/artifacts/<run_id>/`**
- Cannot write outside designated output directory
- Original files are read-only

### Code Execution Safety
- **No dynamic imports/installs** - Rely on preinstalled libs and helpers
- Agent cannot use `subprocess`, `pip`, or package installation commands
- Timeout and memory limits prevent resource exhaustion
- All code execution tracked and logged

## Directory layout (where to look)

### Core Agent Code
- **Agent:** `app/code_exec/handler.py` - Main execution handler and prompt builder
- **Loaders:** `app/code_exec/loaders.py` - File/URL loading and normalization
- **Helpers:** `app/code_exec/{plotting,stats,transform,bio_ops}.py` - Helper function modules

### Output and Storage
- **Artifacts:** `tmp/artifacts/<run_id>/figures/` - Generated plots and analysis outputs
- **Manifests:** `tmp/artifacts/<run_id>/manifest.json` - Run metadata and provenance

### API Integration
- **API routes:** `app/routes.py` - Flask routes for `/upload_content` and `/query` endpoints

---

## How it works (short)
1) Request routing
   - `/upload_content` for files, `/query` for general/URL.
   - `resource="code_exec"` triggers the Code Execution Agent; otherwise PDFs go to RAG by default.

2) Document loading & profiling
   - The handler loads files/URLs, infers structure, and builds a concise prompt with file paths and quick data profiles.
   - **Lazy Data Inspection**: For datasets with >50 columns, the schema is truncated in the prompt (showing only first 20) to save tokens. The LLM is instructed to inspect columns dynamically using `df.columns` if needed.

3) Agent execution
   - Uses LangChain `PythonREPLTool` with a converted LLM (OpenAI/Gemini) to generate and execute Python.
   - Limits: `max_iterations`, `timeout_seconds`, `max_memory_mb`.
   - Retries on rate-limit (e.g., 429) with exponential backoff.

4) Artifacts & response
   - Images/tables saved under `tmp/artifacts/<run_id>/...` and returned as base64 + file paths.
   - Response structure: `text`, `outputs` (stdout/stderr), `artifacts`, `manifest`, `resource` (run_id).

### Common errors and remedies
- 429 Resource exhausted (Gemini): automatic retry; try again or switch model/keys.
- Iteration/time limit hit: simplify the request or increase `options` limits.
- Missing plotting libs: ensure `matplotlib`, `seaborn`, `numpy` are installed (already added in `pyproject.toml`).

### Viewing generated images
- Use the `artifacts[].path` value, e.g., `tmp/artifacts/<run_id>/figures/plot.png` on the host filesystem.



# Auto-Routing to Code Execution Agent Scenario

Here’s a simple end-to-end scenario demonstrating how a request **auto-routes** to the code execution agent, even without explicitly setting `context={"resource":"code_exec"}`.

## User Goal
Upload a **CSV** file and ask for data analysis.

---

## Request Details (Example via Postman)

### Method
`POST`

### URL
`http://localhost:5002/upload_content`

### Headers
* **Authorization:** `Bearer YOUR_TOKEN`

### Body (form-data)
| Key | Value | Note |
| :--- | :--- | :--- |
| **question** | `Compute mean of numeric columns and plot a bar chart of the means.` | The analysis keywords trigger the auto-route. |
| **files** | `[choose file] data.csv` | The file to be analyzed. |
| **context** | *[Do NOT include context]* | Omitted to show auto-routing functionality. |

---

## Why This Auto-Routes
* The **question** starts with “**Compute**” and includes “**plot/bar chart**.”
* This phrasing **matches analysis keywords**, triggering the code execution agent automatically for uploads of **CSV**, **HTML**, or **XML** file types.

---

## Example Successful Response (Trimmed)

```json
{
  "results": [
    {
      "filename": "data.csv",
      "response": {
        "resource": {
          "content_id": "f3c2a0c8-...-1a2b",
          "file_path": "storage/data_files/csv/f3c2a0c8-...-1a2b.csv"
        },
        "text": "CSV file uploaded successfully."
      },
      "answer": {
        "text": "Computed column-wise means and generated a bar chart.",
        "outputs": [
          {
            "step": 1,
            "description": "Code execution",
            "stdout": "Means computed for 7 numeric columns.",
            "stderr": ""
          }
        ],
        "artifacts": [
          {
            "name": "mean_values_bar",
            "format": "png",
            "path": "tmp/artifacts/<run_id>/figures/mean_values_bar.png",
            "data": "data:image/png;base64,..."
          }
        ],
        "resource": { "type": "code_exec", "id": "<run_id>" }
      }
    }
  ]
}
