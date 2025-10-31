from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import uuid
import logging
from . import loaders
from .artifacts import ensure_run_dirs, build_manifest, Artifact
from .sandbox import run_python, SandboxLimits
from .plotting import (
    save_histogram,
    save_scatter,
    save_boxplot,
    save_correlation_heatmap,
    save_timeseries,
    save_pca_preview,
)
from app.prompts.code_exec_prompt import REACT_CODE_EXEC_PROMPT
from app.llm_handle.llm_models import LLMInterface
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


@dataclass
class CodeExecOptions:
    timeout_seconds: int = 60
    max_memory_mb: int = 1024
    output_formats: Optional[List[str]] = None  # e.g., ["png", "csv"]
    allow_network: bool = False


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
                dtypes = {str(c): str(df[c].dtype) for c in getattr(df, "columns", [])}
                missing = {str(c): int(df[c].isna().sum()) for c in getattr(df, "columns", [])}
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

    def plan_with_react(self, instructions: str, profile: Dict[str, Any], loaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code plan and snippets via ReAct using LLM.
        
        Returns a plan with steps and executable Python code snippets.
        """
        try:
            # Build context for LLM
            profile_summary = json.dumps(profile, indent=2)
            data_summary = f"Loaded {len(loaded_data.get('tables', []))} table(s), {len(loaded_data.get('texts', []))} text block(s)"
            
            # Construct full prompt
            full_prompt = f"""{REACT_CODE_EXEC_PROMPT}

USER INSTRUCTIONS: {instructions}

DATA PROFILE:
{profile_summary}

DATA SUMMARY: {data_summary}

Based on the user instructions and data profile above, generate a plan with executable Python code snippets.
The code should:
1. Work with pandas DataFrames from context['tables']
2. Use available libraries: pandas, numpy, matplotlib, seaborn, scipy, statsmodels (if needed)
3. Generate plots and save them
4. Return results in a clear format

IMPORTANT: Return ONLY valid JSON in this format:
{{
    "steps": ["step1_description", "step2_description", ...],
    "code_snippets": ["python code for step1", "python code for step2", ...],
    "reasoning": "Brief explanation of the approach"
}}

Make sure the code snippets are complete, executable Python code. They will have access to:
- context['tables']: List of dicts with 'dataframe' (pandas DataFrame) and 'source' keys
- Standard libraries: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
- Output directory for saving plots will be provided in context['output_dir']
"""
            
            logger.info("Calling LLM for ReAct code generation")
            response = self.llm.generate(full_prompt)
            
            # Parse response
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                # Try to extract JSON from string
                try:
                    # Remove markdown code blocks if present
                    cleaned = response.strip()
                    if "```json" in cleaned:
                        start = cleaned.find("```json") + 7
                        end = cleaned.find("```", start)
                        cleaned = cleaned[start:end].strip()
                    elif "```" in cleaned:
                        start = cleaned.find("```") + 3
                        end = cleaned.find("```", start)
                        cleaned = cleaned[start:end].strip()
                    
                    parsed = json.loads(cleaned)
                    return parsed
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}")
                    # Fallback: try to extract code from response
                    return {"steps": ["Parse response"], "code_snippets": [f"# LLM Response:\n{response}"]}
            else:
                return {"steps": ["Unknown response format"], "code_snippets": [f"# Response: {response}"]}
                
        except Exception as e:
            logger.error(f"Error in ReAct planning: {e}", exc_info=True)
            return {"steps": ["Error in planning"], "code_snippets": [f"# Error: {str(e)}"]}

    def run_in_sandbox(
        self,
        code: str,
        data_context: Dict[str, Any],
        options: CodeExecOptions,
    ) -> Dict[str, Any]:
        limits = SandboxLimits(
            timeout_seconds=options.timeout_seconds,
            max_memory_mb=options.max_memory_mb,
            allow_network=options.allow_network,
        )
        return run_python(code=code, context=data_context, limits=limits)

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
        """High-level orchestration with full LLM → code generation → execution pipeline."""
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
            
            # Step 3: Generate plan and code via LLM (ReAct)
            if user_id:
                emit_to_user(user=user_id, message="Planning analysis with AI...")
            plan = self.plan_with_react(instructions, prof, loaded)
            
            steps = plan.get("steps", [])
            code_snippets = plan.get("code_snippets", [])
            
            if not code_snippets:
                return {
                    "text": "No code was generated. Please rephrase your request.",
                    "manifest": {},
                    "resource": {"type": "code_exec", "id": run_id},
                }
            
            # Step 4: Set up output directory for artifacts
            root = self.artifacts_root or os.path.join("tmp", "artifacts")
            dirs = ensure_run_dirs(root, run_id)
            output_dir = dirs["figures"]
            
            # Prepare data context for sandbox
            data_context = {
                "tables": loaded.get("tables", []),
                "texts": loaded.get("texts", []),
                "metadata": loaded.get("metadata", {}),
                "profiles": prof.get("profiles", []),
                "output_dir": output_dir,
                "run_id": run_id,
            }
            
            # Step 5: Execute code snippets iteratively with ReAct error correction
            all_outputs = []
            all_artifacts = []
            execution_summary = []
            max_retries = 3
            has_errors = False
            
            for idx, (step_desc, code) in enumerate(zip(steps, code_snippets)):
                if user_id:
                    emit_to_user(user=user_id, message=f"Executing step {idx+1}/{len(steps)}: {step_desc[:50]}...")
                
                attempt = 0
                success = False
                last_error = None
                
                while attempt < max_retries and not success:
                    try:
                        # Execute code in sandbox
                        exec_result = self.run_in_sandbox(code=code, data_context=data_context, options=opts)
                        
                        stdout = exec_result.get("stdout", "")
                        stderr = exec_result.get("stderr", "")
                        
                        # Check if execution failed (look for common error indicators)
                        execution_failed = bool(
                            stderr and (
                                "Error" in stderr or 
                                "Traceback" in stderr or 
                                "Exception" in stderr or
                                "NameError" in stderr or
                                "AttributeError" in stderr or
                                "KeyError" in stderr or
                                "ValueError" in stderr or
                                "TypeError" in stderr
                            )
                        )
                        
                        if execution_failed:
                            if attempt < max_retries - 1:
                                # ReAct: Ask LLM to fix the error
                                if user_id:
                                    emit_to_user(user=user_id, message=f"Step {idx+1} had an error, asking AI to fix it...")
                                
                                fix_prompt = f"""The following Python code failed with an error. Fix it and return only the corrected code.

ORIGINAL CODE:
```python
{code}
```

ERROR MESSAGE:
{stderr}

CONTEXT: The code is working with pandas DataFrames from context['tables']. Data profile: {json.dumps(prof.get('profiles', [])[:1], indent=2)}

Return ONLY the corrected Python code, without explanations or markdown."""
                                
                                fixed_code_response = self.llm.generate(fix_prompt)
                                if isinstance(fixed_code_response, str):
                                    # Extract code from response
                                    if "```python" in fixed_code_response:
                                        start = fixed_code_response.find("```python") + 9
                                        end = fixed_code_response.find("```", start)
                                        code = fixed_code_response[start:end].strip()
                                    elif "```" in fixed_code_response:
                                        start = fixed_code_response.find("```") + 3
                                        end = fixed_code_response.find("```", start)
                                        code = fixed_code_response[start:end].strip()
                                    else:
                                        code = fixed_code_response.strip()
                                elif isinstance(fixed_code_response, dict) and "code" in fixed_code_response:
                                    code = fixed_code_response["code"]
                                
                                attempt += 1
                                continue
                            else:
                                last_error = stderr
                                has_errors = True
                        
                        # Success or final attempt
                        all_outputs.append({"step": idx+1, "description": step_desc, "stdout": stdout, "stderr": stderr})
                        execution_summary.append(f"Step {idx+1}: {step_desc} - {'Success' if not stderr else 'Completed with warnings'}")
                        success = True
                        
                    except Exception as e:
                        logger.error(f"Error executing step {idx+1}: {e}", exc_info=True)
                        last_error = str(e)
                        has_errors = True
                        attempt += 1
                        if attempt >= max_retries:
                            all_outputs.append({"step": idx+1, "description": step_desc, "error": last_error})
                            execution_summary.append(f"Step {idx+1}: {step_desc} - Failed after {max_retries} attempts")
            
            # Step 6: Collect artifacts (figures, tables)
            artifacts_list: List[Artifact] = []
            
            # Scan output directory for generated figures
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if filename.endswith(('.png', '.svg', '.pdf', '.html')):
                        file_path = os.path.join(output_dir, filename)
                        try:
                            size = os.path.getsize(file_path)
                            artifacts_list.append(Artifact(
                                id=str(uuid.uuid4()),
                                type="figure",
                                name=os.path.splitext(filename)[0],
                                format=filename.split('.')[-1],
                                path_or_url=file_path,
                                bytes_size=size,
                                metadata={"step": "generated", "filename": filename}
                            ))
                        except Exception as e:
                            logger.warning(f"Error processing artifact {filename}: {e}")
            
            # Step 7: Build manifest and return results
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
            summary_lines = [
                f"Completed {len(execution_summary)} step(s):",
                *execution_summary,
            ]
            if artifacts_list:
                summary_lines.append(f"\nGenerated {len(artifacts_list)} artifact(s):")
                for art in artifacts_list:
                    summary_lines.append(f"  - {art.name}.{art.format}")
            
            if has_errors:
                summary_lines.append(f"\nWarning: Some steps encountered errors. Check outputs for details.")
            
            return {
                "text": "\n".join(summary_lines),
                "manifest": manifest,
                "outputs": all_outputs,
                "artifacts": [{"name": a.name, "format": a.format, "path": a.path_or_url} for a in artifacts_list],
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

    # --- Tool-like wrappers to be exposed to the agent later ---

    def tool_load_document(self, *, files: Optional[List[str]], urls: Optional[List[str]]) -> Dict[str, Any]:
        return self.load_documents(files=files, urls=urls)

    def tool_profile_data(self, *, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.profile_data(data)

    def tool_run_python_sandboxed(self, *, code: str, context: Dict[str, Any], options: Optional[CodeExecOptions] = None) -> Dict[str, Any]:
        return self.run_in_sandbox(code=code, data_context=context, options=options or CodeExecOptions())

    def tool_render_plot(self, *, run_id: str, kind: str, params: Dict[str, Any], options: Optional[CodeExecOptions] = None) -> Dict[str, Any]:
        # Placeholder; plotting handled in a later chunk
        return {"status": "not_implemented", "kind": kind}

    def tool_export_table(self, *, run_id: str, name: str, fmt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Placeholder; export handled in a later chunk
        return {"status": "not_implemented", "name": name, "format": fmt}


