from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import uuid
from . import loaders
from .artifacts import ensure_run_dirs, build_manifest, Artifact
from .sandbox import run_python, SandboxLimits

# Placeholders to be implemented in subsequent chunks
# - Loaders will normalize various document types to canonical structures
# - Sandbox will execute python code safely with resource limits
# - Plotting will create publication-ready figures
# - Artifacts will manage manifests and paths


@dataclass
class CodeExecOptions:
    timeout_seconds: int = 60
    max_memory_mb: int = 1024
    output_formats: Optional[List[str]] = None  # e.g., ["png", "csv"]
    allow_network: bool = False


class CodeExecutionHandler:
    """Coordinator for the code-execution agent lifecycle.

    Responsibilities:
    - Ingest and normalize documents (CSV/HTML/XML/PDF/URL)
    - Profile data and plan analysis steps (via ReAct planning)
    - Execute code safely in a sandbox and collect outputs
    - Render figures/tables and persist artifacts with provenance
    """

    def __init__(self, *, artifacts_root: Optional[str] = None) -> None:
        self.artifacts_root = artifacts_root

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

    def plan_with_react(self, instructions: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Stub: produce a step plan and code snippets via ReAct. Implemented later."""
        return {"steps": [], "code_snippets": []}

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
        """High-level orchestration entrypoint used by the agent node."""
        opts = options or CodeExecOptions()
        loaded = self.load_documents(files=files, urls=urls, user_id=user_id)
        prof = self.profile_data(loaded)
        plan = self.plan_with_react(instructions, prof)

        # In a future chunk we will iterate over steps/code_snippets
        exec_result = self.run_in_sandbox(code="", data_context=loaded, options=opts)

        # Generate run id and write minimal manifest
        run_id = uuid.uuid4().hex
        manifest = self.render_artifacts(run_id, exec_result, opts)
        return {
            "text": "Code execution completed (scaffold)",
            "manifest": manifest,
            "resource": {"type": "code_exec", "id": run_id},
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


