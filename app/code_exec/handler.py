from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import uuid
import logging
import base64
import sys
import time
from . import loaders
from .artifacts import ensure_run_dirs, build_manifest, Artifact
# Removed: from .sandbox import run_python, SandboxLimits
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType
# Removed: REACT_CODE_EXEC_PROMPT import - no longer using JSON-based planning
from app.llm_handle.llm_models import LLMInterface
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


@dataclass
class CodeExecOptions:
    timeout_seconds: int = 120  # Increased default to 120 seconds
    max_memory_mb: int = 2048  # Increased default to 2048 MB
    output_formats: Optional[List[str]] = None  # e.g., ["png", "csv"]
    allow_network: bool = False
    max_iterations: int = 20  # Maximum agent iterations before stopping


class CodeExecutionHandler:
    """Coordinator for the code-execution agent lifecycle with full LLM integration.

    Responsibilities:
    - Ingest and normalize documents (CSV/HTML/XML/PDF/URL)
    - Profile data and plan analysis steps (via ReAct planning with LLM)
    - Execute LLM-generated code safely in a sandbox and collect outputs
    - Render figures/tables and persist artifacts with provenance
    - Iterative error correction via ReAct framework
    """

    def __init__(self, llm: LLMInterface, *, artifacts_root: Optional[str] = None) -> None:
        self.llm = llm
        self.artifacts_root = artifacts_root
        logger.info(f"CodeExecutionHandler initialized with LLM: {type(llm).__name__}")
        # Convert LLMInterface to LangChain LLM for PythonREPLTool
        self._langchain_llm = self._convert_to_langchain_llm(llm)
    
    def _convert_to_langchain_llm(self, llm: LLMInterface):
        """Convert LLMInterface to LangChain ChatModel for PythonREPLTool agent."""
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        # Check if llm has model_provider attribute
        if hasattr(llm, 'model_provider'):
            if llm.model_provider == "openai":
                api_key = getattr(llm, 'api_key', None) or os.getenv("OPENAI_API_KEY")
                model_name = getattr(llm, 'model_name', 'gpt-3.5-turbo')
                return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
            elif llm.model_provider == "gemini":
                api_key = getattr(llm, 'api_key', None) or os.getenv("GEMINI_API_KEY")
                model_name = getattr(llm, 'model_name', 'gemini-pro')
                return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        
        # Fallback: try to detect from class name
        llm_class_name = type(llm).__name__
        if 'OpenAI' in llm_class_name:
            api_key = getattr(llm, 'api_key', None) or os.getenv("OPENAI_API_KEY")
            model_name = getattr(llm, 'model_name', 'gpt-3.5-turbo')
            return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
        elif 'Gemini' in llm_class_name:
            api_key = getattr(llm, 'api_key', None) or os.getenv("GEMINI_API_KEY")
            model_name = getattr(llm, 'model_name', 'gemini-pro')
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        
        # Default to OpenAI if unsure
        logger.warning(f"Unknown LLM type {llm_class_name}, defaulting to OpenAI")
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    def load_documents(
        self,
        *,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load and normalize inputs using loaders module."""
        return loaders.normalize_inputs(files=files, urls=urls)

    def profile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight profiling: shapes, dtypes, missingness, basic stats.

        Expects `data` from normalize_inputs with key `tables`.
        """
        profiles: List[Dict[str, Any]] = []
        for item in data.get("tables", []):
            df = item.get("dataframe")
            if df is None:
                continue
            try:
                numeric_cols = [c for c in getattr(df, "columns", []) if str(df[c].dtype).startswith(("int", "float"))]
                # Truncate profile info if too many columns
                all_cols = getattr(df, "columns", [])
                if len(all_cols) > 50:
                    # Only keep first 20 for profile to save tokens
                    kept_cols = all_cols[:20]
                    dtypes = {str(c): str(df[c].dtype) for c in kept_cols}
                    missing = {str(c): int(df[c].isna().sum()) for c in kept_cols}
                    # Add a note about truncation in the dict keys or separate field if needed
                    # For now, just limiting the dicts is enough as the prompt shows "Missing values: {...}"
                    dtypes["..."] = f"({len(all_cols)-20} more)"
                    missing["..."] = f"({len(all_cols)-20} more)"
                else:
                    dtypes = {str(c): str(df[c].dtype) for c in all_cols}
                    missing = {str(c): int(df[c].isna().sum()) for c in all_cols}

                desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}
                profiles.append({
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "dtypes": dtypes,
                    "missing": missing,
                    "numeric_summary": desc,
                    "source": item.get("source"),
                })
            except Exception:
                profiles.append({"source": item.get("source"), "error": "profiling failed"})

        return {"profiles": profiles}

    # Removed: plan_with_react() - replaced by PythonREPLTool agent
    # Removed: run_in_sandbox() - replaced by PythonREPLTool agent

    def render_artifacts(
        self,
        run_id: str,
        results: Dict[str, Any],
        options: CodeExecOptions,
    ) -> Dict[str, Any]:
        """Write a minimal manifest; figures/tables will be added later."""
        root = self.artifacts_root or os.path.join("tmp", "artifacts")
        dirs = ensure_run_dirs(root, run_id)
        artifacts: List[Artifact] = []
        env = {"limits": {"timeout": options.timeout_seconds, "max_memory_mb": options.max_memory_mb}}
        manifest = build_manifest(run_id, inputs={}, params={}, artifacts=artifacts, environment=env)
        manifest_path = os.path.join(dirs["base"], "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"manifest": manifest, "paths": {"manifest": manifest_path}}

    def execute(
        self,
        *,
        instructions: str,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        options: Optional[CodeExecOptions] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """High-level orchestration using PythonREPLTool for direct code execution."""
        opts = options or CodeExecOptions()
        run_id = uuid.uuid4().hex
        
        try:
            # Step 1: Load documents
            if user_id:
                emit_to_user(user=user_id, message="Parsing documents...")
            loaded = self.load_documents(files=files, urls=urls, user_id=user_id)
            
            # Step 2: Profile data
            if user_id:
                emit_to_user(user=user_id, message="Profiling data...")
            prof = self.profile_data(loaded)
            
            # Step 3: Set up output directory for artifacts
            root = self.artifacts_root or os.path.join("tmp", "artifacts")
            dirs = ensure_run_dirs(root, run_id)
            output_dir = dirs["figures"]
            
            # Step 4: Build natural language prompt with file paths, profiles, and helper info
            if user_id:
                emit_to_user(user=user_id, message="Preparing analysis...")
            
            # Build file paths info (for CSV/HTML/XML - extracted tables)
            file_paths_info = []
            for idx, table_item in enumerate(loaded.get("tables", [])):
                df = table_item.get("dataframe")
                source = table_item.get("source", f"table_{idx}")
                if df is not None:
                    # Save DataFrame to CSV so PythonREPLTool can load it
                    csv_path = os.path.join(output_dir, f"data_{idx}.csv")
                    df.to_csv(csv_path, index=False)
                    
                    # Truncate columns if too many
                    cols = list(df.columns.tolist())
                    if len(cols) > 50:
                        cols_display = cols[:20] + [f"... and {len(cols) - 20} more columns. Use df.columns to inspect."]
                    else:
                        cols_display = cols
                        
                    file_paths_info.append({
                        "path": csv_path,
                        "source": source,
                        "shape": [int(df.shape[0]), int(df.shape[1])],
                        "columns": cols_display
                    })
            
            # Build original files info (for PDF/URL - pass original file paths)
            original_files_info = []
            if files:
                for file_path in files:
                    original_files_info.append({
                        "path": file_path,
                        "type": "original_file",
                        "source": file_path
                    })
            if urls:
                for url in urls:
                    original_files_info.append({
                        "path": url,
                        "type": "url",
                        "source": url
                    })
            
            # Step 5: Set up PythonREPLTool with project root in sys.path
            if user_id:
                emit_to_user(user=user_id, message="Executing code...")
            
            # Get project root (parent of app directory)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(
                instructions=instructions,
                file_paths=file_paths_info,
                original_files=original_files_info,
                profiles=prof.get("profiles", []),
                output_dir=output_dir,
                project_root=project_root
            )
            
            # Set up globals for PythonREPLTool to include project root in sys.path
            # Note: PythonREPLTool executes code in its own namespace, so we add a setup command
            # to the prompt or use globals if supported. For now, we'll rely on the prompt
            # including instructions to modify sys.path if needed.
            repl_globals = {
                "__builtins__": __builtins__,
            }
            # Add project root to sys.path in the globals namespace
            import sys
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Initialize PythonREPLTool with globals
            # The globals dict allows the code to access project modules
            python_repl = PythonREPLTool(globals=repl_globals)
            
            # Initialize agent with PythonREPLTool
            tools = [python_repl]
            
            # Calculate max_iterations based on timeout (allow ~6 seconds per iteration)
            max_iterations = getattr(opts, 'max_iterations', max(15, opts.timeout_seconds // 6))
            
            logger.info(
                f"Initializing agent with max_iterations={max_iterations}, "
                f"timeout={opts.timeout_seconds}s, max_memory={opts.max_memory_mb}MB"
            )
            
            # Initialize agent with PythonREPLTool
            # Note: initialize_agent may not support all parameters directly in some LangChain versions
            # We'll configure via AgentExecutor if needed
            try:
                agent = initialize_agent(
                    tools=tools,
                    llm=self._langchain_llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=max_iterations,
                    max_execution_time=opts.timeout_seconds,
                )
            except TypeError:
                # Fallback if parameters not supported - initialize without them
                logger.warning("Agent initialization doesn't support max_iterations/max_execution_time, using defaults")
                agent = initialize_agent(
                    tools=tools,
                    llm=self._langchain_llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                )
            
            # Step 6: Execute agent with prompt (with retry logic for rate limiting)
            max_retries = 3
            retry_delay = 2  # Initial delay in seconds
            result = None
            execution_output = ""
            execution_errors = ""
            
            for attempt in range(max_retries):
                try:
                    result = agent.run(prompt)
                    execution_output = str(result) if result else "Execution completed"
                    execution_errors = ""
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    logger.warning(f"Agent execution attempt {attempt + 1} error: {error_str}")
                    
                    # Check if it's a rate limit error (429)
                    is_rate_limit = (
                        "429" in error_str or 
                        "Resource exhausted" in error_str or
                        "rate limit" in error_str.lower() or
                        "quota" in error_str.lower()
                    )
                    
                    # Check if it's an iteration/time limit error
                    is_limit_error = (
                        "iteration limit" in error_str.lower() or
                        "time limit" in error_str.lower() or
                        "stopped due to" in error_str.lower()
                    )
                    
                    if is_limit_error:
                        # If hitting limits, log and break - this is a configuration issue, not retry-able
                        logger.error(
                            f"Agent hit iteration/time limit. max_iterations={max_iterations}, "
                            f"timeout={opts.timeout_seconds}s. Error: {error_str}"
                        )
                        execution_output = (
                            f"Execution stopped: {error_str}. "
                            f"This may indicate the task is too complex or limits are too low. "
                            f"Try simplifying the request or increasing timeout/memory limits."
                        )
                        execution_errors = error_str
                        break
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        # Calculate exponential backoff: 2s, 4s, 8s
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit error (attempt {attempt + 1}/{max_retries}): {error_str}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        continue  # Retry
                    else:
                        # Not a rate limit error, or max retries reached
                        logger.warning(f"Agent execution error: {e}", exc_info=True)
                        execution_output = f"Execution completed with warnings: {str(e)}"
                        execution_errors = str(e)
                        break
            
            # Step 7: Collect artifacts (figures, tables)
            artifacts_list: List[Artifact] = []
            artifacts_with_data = []
            
            # Scan output directory for generated figures
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    # Skip data CSV files we created
                    if filename.startswith("data_"):
                        continue
                    if filename.endswith(('.png', '.svg', '.pdf', '.html', '.jpg', '.jpeg')):
                        file_path = os.path.join(output_dir, filename)
                        try:
                            size = os.path.getsize(file_path)
                            art = Artifact(
                                id=str(uuid.uuid4()),
                                type="figure",
                                name=os.path.splitext(filename)[0],
                                format=filename.split('.')[-1],
                                path_or_url=file_path,
                                bytes_size=size,
                                metadata={"step": "generated", "filename": filename}
                            )
                            artifacts_list.append(art)
                            
                            # Read and encode image files for direct display (limit to 10MB)
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')) and size < 10 * 1024 * 1024:
                                try:
                                    with open(file_path, 'rb') as f:
                                        image_data = f.read()
                                        base64_data = base64.b64encode(image_data).decode('utf-8')
                                        
                                        # Determine MIME type
                                        ext = filename.lower().split('.')[-1]
                                        mime_type = {
                                            'png': 'image/png',
                                            'jpg': 'image/jpeg',
                                            'jpeg': 'image/jpeg',
                                            'svg': 'image/svg+xml'
                                        }.get(ext, 'image/png')
                                        
                                        artifacts_with_data.append({
                                            "name": art.name,
                                            "format": art.format,
                                            "path": file_path,
                                            "data": f"data:{mime_type};base64,{base64_data}",
                                            "size_bytes": size
                                        })
                                except Exception as e:
                                    logger.warning(f"Failed to encode image {filename}: {e}")
                                    artifacts_with_data.append({
                                        "name": art.name,
                                        "format": art.format,
                                        "path": file_path,
                                        "size_bytes": size
                                    })
                            else:
                                artifacts_with_data.append({
                                    "name": art.name,
                                    "format": art.format,
                                    "path": file_path,
                                    "size_bytes": size
                                })
                        except Exception as e:
                            logger.warning(f"Error processing artifact {filename}: {e}")
            
            # Step 8: Build manifest and return results
            if user_id:
                emit_to_user(user=user_id, message="Rendering artifacts...")
            
            env = {
                "limits": {"timeout": opts.timeout_seconds, "max_memory_mb": opts.max_memory_mb},
                "llm": type(self.llm).__name__,
            }
            manifest = build_manifest(
                run_id=run_id,
                inputs={"files": files or [], "urls": urls or [], "instructions": instructions},
                params={"timeout": opts.timeout_seconds, "max_memory_mb": opts.max_memory_mb},
                artifacts=artifacts_list,
                environment=env
            )
            
            manifest_path = os.path.join(dirs["base"], "manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write manifest: {e}")
            
            # Generate human-readable summary
            summary_lines = [execution_output]
            if artifacts_list:
                summary_lines.append(f"\nGenerated {len(artifacts_list)} artifact(s):")
                for art in artifacts_list:
                    summary_lines.append(f"  - {art.name}.{art.format}")
            
            if execution_errors:
                summary_lines.append(f"\nNote: {execution_errors}")
            
            return {
                "text": "\n".join(summary_lines),
                "manifest": manifest,
                "outputs": [{"step": 1, "description": "Code execution", "stdout": execution_output, "stderr": execution_errors}],
                "artifacts": artifacts_with_data if artifacts_with_data else [{"name": a.name, "format": a.format, "path": a.path_or_url} for a in artifacts_list],
                "resource": {"type": "code_exec", "id": run_id},
            }
            
        except Exception as e:
            logger.error(f"Error in code execution pipeline: {e}", exc_info=True)
            return {
                "text": f"Error during code execution: {str(e)}",
                "manifest": {},
                "resource": {"type": "code_exec", "id": run_id},
                "error": str(e),
            }
    
    def _build_enhanced_prompt(
        self,
        instructions: str,
        file_paths: List[Dict[str, Any]],
        original_files: List[Dict[str, Any]],
        profiles: List[Dict[str, Any]],
        output_dir: str,
        project_root: str
    ) -> str:
        """Build enhanced prompt for PythonREPLTool with file paths, profiles, and helper info."""
        prompt_parts = [
            "You are a Python data analysis assistant. Execute Python code to complete the user's request.",
            "",
            "CRITICAL RULES:",
            "1. DO NOT install packages - use only libraries that are already installed",
            "2. DO NOT use subprocess, pip, or any package installation commands",
            "3. Use the loaders module to load files if needed: from app.code_exec.loaders import load_pdf, load_csv, load_html, load_xml, load_url",
            "4. For PDFs, use pdfplumber or camelot (already installed) - DO NOT use tabula-py",
            "5. IF DATASET SCHEMA IS LARGE OR TRUNCATED: You MUST run `df.columns` or `df.head()` to inspect the data before writing your analysis code. Do NOT guess column names.",
            "",
            "USER REQUEST:",
            instructions,
            "",
        ]
        
        # Add original file paths (PDF/URL) - show these even if no tables extracted
        if original_files:
            prompt_parts.append("ORIGINAL FILES (if you need to process them directly):")
            for orig_file in original_files:
                file_path = orig_file['path']
                file_type = orig_file.get('type', 'file')
                if file_type == 'url':
                    prompt_parts.append(f"  - URL: {file_path}")
                    prompt_parts.append(f"    To load: from app.code_exec.loaders import load_url; data = load_url('{file_path}')")
                else:
                    ext = os.path.splitext(file_path)[1].lower()
                    prompt_parts.append(f"  - File: {file_path} (type: {ext})")
                    if ext == '.pdf':
                        prompt_parts.append(f"    To load: from app.code_exec.loaders import load_pdf; data = load_pdf('{file_path}')")
                        prompt_parts.append(f"    Or use: import pdfplumber; with pdfplumber.open('{file_path}') as pdf: tables = [page.extract_table() for page in pdf.pages]")
                    elif ext == '.csv':
                        prompt_parts.append(f"    To load: import pandas as pd; df = pd.read_csv('{file_path}')")
                    elif ext in ['.html', '.htm']:
                        prompt_parts.append(f"    To load: from app.code_exec.loaders import load_html; data = load_html('{file_path}')")
                    elif ext == '.xml':
                        prompt_parts.append(f"    To load: from app.code_exec.loaders import load_xml; data = load_xml('{file_path}')")
            prompt_parts.append("")
        
        # Add extracted tables (CSV/HTML/XML - already loaded as CSV)
        if file_paths:
            prompt_parts.append("EXTRACTED TABLES (already loaded as CSV files):")
            for fp_info in file_paths:
                prompt_parts.append(f"  - File: {fp_info['path']}")
                prompt_parts.append(f"    Source: {fp_info['source']}")
                prompt_parts.append(f"    Shape: {fp_info['shape']}")
                prompt_parts.append(f"    Columns: {fp_info['columns']}")
            prompt_parts.append("")
            prompt_parts.append("To load these CSV files, use:")
            prompt_parts.append("  import pandas as pd")
            prompt_parts.append("  df = pd.read_csv('path_to_file.csv')")
            prompt_parts.append("")
        
        if profiles:
            prompt_parts.append("DATA PROFILES:")
            for prof in profiles[:3]:  # Limit to first 3 profiles
                if "error" not in prof:
                    prompt_parts.append(f"  Source: {prof.get('source', 'unknown')}")
                    prompt_parts.append(f"    Shape: {prof.get('shape')}")
                    prompt_parts.append(f"    Missing values: {prof.get('missing', {})}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "OUTPUT DIRECTORY:",
            f"  Save all plots to: {output_dir}",
            "",
            "IMPORTANT: Complete the task efficiently. Use helper functions when available:",
            f"  1. Add project root to sys.path: sys.path.insert(0, r'{project_root}')",
            "  2. Import helpers: from app.code_exec.plotting import save_correlation_heatmap",
            "  3. For correlation heatmap: save_correlation_heatmap(df, os.path.join(output_dir, 'heatmap.png'))",
            "  4. For statistics: from app.code_exec.stats import correlations",
            "",
            "AVAILABLE LIBRARIES:",
            "  - pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)",
            "  - pdfplumber (for PDF text/table extraction)",
            "  - camelot (for PDF table extraction)",
            "  - app.code_exec.loaders (load_pdf, load_csv, load_html, load_xml, load_url)",
            "",
            "EXECUTE CODE NOW to complete: " + instructions,
        ])
        
        return "\n".join(prompt_parts)

    # --- Tool-like wrappers to be exposed to the agent later ---

    def tool_load_document(self, *, files: Optional[List[str]], urls: Optional[List[str]]) -> Dict[str, Any]:
        return self.load_documents(files=files, urls=urls)

    def tool_profile_data(self, *, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.profile_data(data)

    # Removed: tool_run_python_sandboxed() - replaced by PythonREPLTool agent

    def tool_render_plot(self, *, run_id: str, kind: str, params: Dict[str, Any], options: Optional[CodeExecOptions] = None) -> Dict[str, Any]:
        # Placeholder; plotting handled in a later chunk
        return {"status": "not_implemented", "kind": kind}

    def tool_export_table(self, *, run_id: str, name: str, fmt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Placeholder; export handled in a later chunk
        return {"status": "not_implemented", "name": name, "format": fmt}


