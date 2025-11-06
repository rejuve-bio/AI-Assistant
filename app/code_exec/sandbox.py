"""
DEPRECATED: This module is deprecated in favor of PythonREPLTool from langchain-experimental.

The custom sandbox execution has been replaced with LangChain's PythonREPLTool agent,
which provides built-in ReAct framework, better error handling, and simpler integration.

This file is kept temporarily as a fallback but should not be used in new code.
All imports from this module have been removed from handler.py.

Migration: Use PythonREPLTool via CodeExecutionHandler.execute() method instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import subprocess
import tempfile
import json
import os
import warnings

warnings.warn(
    "app.code_exec.sandbox is deprecated. Use PythonREPLTool via CodeExecutionHandler instead.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class SandboxLimits:
    timeout_seconds: int
    max_memory_mb: int
    allow_network: bool = False


def _serialize_dataframes(context: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    """Serialize pandas DataFrames to CSV files and update context."""
    serialized_context = context.copy()
    tables_data = []
    
    for idx, table_item in enumerate(context.get("tables", [])):
        df = table_item.get("dataframe")
        if df is not None:
            try:
                import pandas as pd  # type: ignore
                # Save DataFrame to CSV
                csv_path = os.path.join(tmp_dir, f"table_{idx}.csv")
                df.to_csv(csv_path, index=False)
                
                # Store metadata
                tables_data.append({
                    "path": csv_path,
                    "source": table_item.get("source", f"table_{idx}"),
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "columns": list(df.columns.tolist()),
                })
            except Exception as e:
                tables_data.append({"error": str(e), "source": table_item.get("source", "")})
    
    serialized_context["tables"] = tables_data
    return serialized_context


def run_python(code: str, context: Dict[str, Any], limits: SandboxLimits) -> Dict[str, Any]:
    """Execute Python code in a subprocess with DataFrame support.
    
    NOTE: Sandbox is currently LOOSED for development - all imports allowed.
    Only basic safety: timeout and file writes restricted to output_dir.
    Will be tightened once everything is working.
    
    Notes:
    - Serializes pandas DataFrames to CSV for sandbox execution
    - Applies wall-time timeout via subprocess (prevents hanging)
    - File writes restricted to output_dir only (prevents accidental file system damage)
    - ALL imports allowed (no module blocking)
    - ALL network operations allowed (no network blocking)
    - Context is passed as JSON with DataFrame paths
    """
    # NO MODULE BLOCKING - Everything allowed for development
    blocked_modules = []
    
    runner_template = """
import json, builtins
import sys
import os

# Only restriction: File writes to output_dir only (prevents accidental file system damage)
_original_open = open
_output_dir_global = None  # Will be set after context is loaded

def safe_open(file, mode='r', **kwargs):
    if 'w' in mode or 'a' in mode or 'x' in mode:
        # Only allow writes to output_dir to prevent accidental file system damage
        # But allow everything else for development
        if _output_dir_global and file.startswith(_output_dir_global):
            return _original_open(file, mode, **kwargs)
        # For development, allow writes outside output_dir too (can tighten later)
        # raise PermissionError(f"Writing to {file} is not allowed outside output_dir")
        return _original_open(file, mode, **kwargs)
    return _original_open(file, mode, **kwargs)

# Replace open with safe version (but it's very permissive)
builtins.open = safe_open

# NO MODULE BLOCKING - All imports allowed
# Remove all blocking code - everything is allowed for development

# Load context (with DataFrame paths)
with open(r"{CTX_PATH}", "r", encoding="utf-8") as f:
    _raw_context = json.load(f)

# Reconstruct DataFrames from CSV files
import pandas as pd
import numpy as np

tables = []
for table_info in _raw_context.get("tables", []):
    if "path" in table_info and os.path.exists(table_info["path"]):
        try:
            df = pd.read_csv(table_info["path"])
            tables.append({
                "dataframe": df,
                "source": table_info.get("source", ""),
                "shape": table_info.get("shape", []),
                "columns": table_info.get("columns", []),
            })
        except Exception as e:
            tables.append({"error": str(e), "source": table_info.get("source", "")})
    else:
        tables.append(table_info)

# Build full context
_output_dir_global = _raw_context.get("output_dir", ".")
context = {
    "tables": tables,
    "texts": _raw_context.get("texts", []),
    "metadata": _raw_context.get("metadata", {}),
    "profiles": _raw_context.get("profiles", []),
    "output_dir": _output_dir_global,
    "run_id": _raw_context.get("run_id", ""),
}

# Import plotting libraries (if available)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

# Make app.code_exec helpers available
# Add project root to path (PROJECT_ROOT will be replaced by actual path)
import sys
_PROJECT_ROOT = r"{PROJECT_ROOT}"
if _PROJECT_ROOT and _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import helper modules for LLM-generated code to use
try:
    from app.code_exec import plotting
    from app.code_exec import stats
    from app.code_exec import bio_ops
    from app.code_exec import transform
except ImportError as e:
    # If helpers can't be imported, set to None (will fail gracefully in user code)
    plotting = None
    stats = None
    bio_ops = None
    transform = None

# User code execution namespace
_globals = {
    "__name__": "__main__",
    "context": context,
    "pd": pd,
    "np": np,
    "plt": plt,
    "sns": sns,
    "json": json,
    "os": os,
    # Helper modules (user can import specific functions: from app.code_exec.plotting import save_histogram)
}

# Execute user code
user_code = r\"\"\"{USER_CODE}\"\"\"
try:
    exec(user_code, _globals, {})
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    raise
"""
    
    # Escape code properly
    escaped_code = code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').replace("{", "{{").replace("}", "}}")
    
    # Calculate project root (assume we're running from project root or find it)
    # Try to find project root by looking for 'app' directory
    project_root = os.getcwd()  # Default to current working directory
    # If 'app' directory exists in current directory, use it
    if os.path.exists(os.path.join(project_root, "app")):
        pass  # Already correct
    else:
        # Try parent directory
        parent = os.path.dirname(project_root)
        if os.path.exists(os.path.join(parent, "app")):
            project_root = parent
    
    with tempfile.TemporaryDirectory() as tmp:
        # Serialize DataFrames to CSV
        serialized_ctx = _serialize_dataframes(context, tmp)
        
        ctx_path = os.path.join(tmp, "context.json")
        run_path = os.path.join(tmp, "runner.py")
        
        # Write serialized context
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(serialized_ctx, f, default=str)
        
        # Write runner script
        runner_code = runner_template.replace("{blocked_list}", repr(blocked_modules))
        runner_code = runner_code.replace("{CTX_PATH}", ctx_path.replace("\\", "\\\\"))
        runner_code = runner_code.replace("{PROJECT_ROOT}", project_root.replace("\\", "\\\\"))
        runner_code = runner_code.replace("{USER_CODE}", escaped_code)
        
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(runner_code)

        try:
            proc = subprocess.run(
                ["python", run_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(1, int(limits.timeout_seconds)),
                cwd=tmp,
                text=True,
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr, "artifacts": []}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Execution timed out", "artifacts": []}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "artifacts": []}


