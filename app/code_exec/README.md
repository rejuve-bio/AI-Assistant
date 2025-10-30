# Code Execution Agent (ReAct + LangChain)

This module adds a secure, document-aware code execution agent to the AI Assistant. It ingests CSV/HTML/XML/PDF/URL, plans with ReAct, executes Python in a sandbox, and returns artifacts (tables, plots, manifests) like other agents (Galaxy, Annotation, RAG, Hypothesis) via the unified `/query` API.

## What it does
- Data ingestion and normalization across CSV/HTML/XML/PDF/URL
- Lightweight profiling (shape, dtypes, missingness, numeric stats)
- ReAct planning prompt to produce minimal, safe steps/code
- Secure Python execution (timeout, memory caps, blocked imports)
- Visualization utilities (hist, box/violin, scatter, time-series, correlation heatmaps, PCA preview)
- Statistics (t-test, Mann–Whitney, ANOVA/Kruskal) with FDR; correlations; linear regression
- Bio helpers (differential-like analysis + volcano, ORA enrichment, pathway/ID mapping, IC50 fitting)
- Cleaning transforms (missingness report, dedup, joins, scaling, unit conversions)
- Artifact management (figures/tables/logs manifest, TTL cleanup)

## Module layout
- `handler.py`
  - High-level coordinator (`CodeExecutionHandler`) with `execute()` and tool-style wrappers (`tool_load_document`, `tool_profile_data`, `tool_run_python_sandboxed`, `tool_render_plot`, `tool_export_table`).
  - Writes a per-run manifest and returns a `resource: { type: "code_exec", id: <run_id> }` object.
- `loaders.py`
  - Loaders for `CSV`, `HTML`, `XML`, `PDF`, and `URL`.
  - `normalize_inputs(files, urls)` routes by extension/content-type to produce canonical structures (`tables`, `texts`, `metadata`).
- `sandbox.py`
  - Subprocess execution with `SandboxLimits` (timeout, memory hint, network allowlist flag).
  - Blocks dangerous builtins/imports and captures stdout/stderr.
- `plotting.py`
  - Publication-oriented plots: histogram, box/violin, scatter, time-series, correlation heatmap; PCA preview.
  - Uses Matplotlib Agg backend + seaborn (no display).
- `stats.py`
  - Statistical tests (t-test, Mann–Whitney, ANOVA/Kruskal) with FDR (statsmodels fallback included), correlations, and linear regression.
- `bio_ops.py`
  - Differential-like analysis with volcano components, ORA enrichment (Fisher), simple ID mapping, dose–response (4PL) IC50 fitting.
- `transform.py`
  - Missingness report, deduplication, joins, scaling, and unit conversions.
- `artifacts.py`
  - `ensure_run_dirs()` sets stable run folders; `build_manifest()` creates a run manifest; `cleanup_expired()` TTL cleanup.
- `README.md`
  - This document.

## How it integrates with other services
- Entry point remains `/query` (like Galaxy/Annotation/RAG/Hypothesis).
- In `app/main.py`:
  - `code_exec_agent` node is added to the LangGraph alongside other agents.
  - Classifier prompt extended with a `code_exec` category.
  - Routing sends code-exec intents to `_code_exec_agent`, which calls `CodeExecutionHandler.execute()`.
- In `app/routes.py`:
  - `/query` accepts `context.resource=code_exec`, optional `urls`, and `options` (JSON) for execution constraints.

## Execution workflow
1) Ingestion: `normalize_inputs(files, urls)` loads and normalizes documents.
2) Profiling: `profile_data()` summarizes schema and numeric stats.
3) Planning: `REACT_CODE_EXEC_PROMPT` proposes steps and minimal code.
4) Execution: `run_in_sandbox()` runs Python with blocked imports and timeout.
5) Rendering: plots/tables saved; a manifest is written per run.
6) Response: returns a concise text plus `resource: { type: "code_exec", id: <run_id> }` and artifact pointers.

## Safety and constraints
- Subprocess wall-time timeout; restricted imports/builtins; stdout/stderr captured.
- No network during execution by default (configurable allowlist flag).
- File access limited to a temp run directory.
- Size caps recommended at call sites; sample for previews on large data.

## Configuration (env with defaults)
- `CODE_EXEC_TIMEOUT` (default 60)
- `CODE_EXEC_MAX_MEM_MB` (default 1024)
- `CODE_EXEC_ALLOW_NETWORK` (default false)
- `ARTIFACT_TTL_MINUTES` (default 120)

## How to call it via API
Use the existing `/query` endpoint with `resource=code_exec`:

```bash
curl -X POST http://localhost:5002/query \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=Compute correlation heatmap and a PCA preview" \
  -d "context={\"resource\":\"code_exec\"}" \
  -d "urls=https://example.com/data.csv" \
  -d "options={\"timeout_seconds\":60,\"max_memory_mb\":1024}"
```

The agent streams progress via SocketIO (Parsing → Profiling → Planning → Executing → Rendering → Done).

## Typical scenarios
- CSV EDA: "Load this CSV, compute summary stats, correlation matrix, and plot a heatmap."
- PDF table reuse: "Extract the main results table from this PDF URL and reproduce a bar chart with CIs."
- HTML supplemental data: "Scrape tables from this HTML, harmonize gene symbols, run ORA enrichment, and export top terms."
- XML assay: "Normalize this XML assay file, fit dose–response, estimate IC50, and export a summary table."
- Multi-source joins: "Join this CSV with a CSV at a URL on `sample_id`, compute group comparisons, and make a volcano plot."

## Development notes
- Stats/bio functions intentionally lightweight; escalate to specialized tools only if needed.
- Many functions return `{ "type": "error", "error": "..." }` on failure—handle gracefully.
- Plotting functions use Agg backend; no GUI dependencies required.
- Artifacts live under `tmp/artifacts/<run_id>/` by default; use TTL cleanup to control disk usage.
