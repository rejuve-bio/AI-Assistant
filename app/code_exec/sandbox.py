from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import subprocess
import tempfile
import json
import os


@dataclass
class SandboxLimits:
    timeout_seconds: int
    max_memory_mb: int
    allow_network: bool = False


def run_python(code: str, context: Dict[str, Any], limits: SandboxLimits) -> Dict[str, Any]:
    """Execute Python code in a subprocess with basic restrictions and timeout.

    Notes:
    - Applies a wall-time timeout via subprocess.
    - Blocks selected imports by injecting a guard into the runner.
    - Context is passed as JSON and reconstructed as Python objects.
    - For portability across OS (including Windows), we avoid resource module.
    """
    blocked_modules = [
        "socket", "requests", "urllib", "ftplib", "http", "subprocess", "os", "pathlib", "shutil", "sys",
    ]

    runner = f"""
import json, builtins

# Block dangerous builtins
for name in ["open", "exec", "eval", "compile", "__import__"]:
    setattr(builtins, name, None)

# Block selected modules by preloading dummies
class _Blocked:
    def __getattr__(self, name):
        raise ImportError("Module blocked")

import sys
blocked = _Blocked()
for m in {blocked_list}:
    sys.modules[m] = blocked

# Reconstruct context
with open(r"{{CTX}}", "r", encoding="utf-8") as f:
    _context = json.load(f)

# User code execution namespace
_globals = {{"__name__": "__main__", "context": _context}}
_locals = None

# Execute user code
user_code = r"""{user_code}"""
exec(user_code, _globals, _locals)
""".replace("{blocked_list}", repr(blocked_modules)).replace("{user_code}", code.replace("\\", "\\\\").replace('"""', '"\"\"')).replace("{", "{")

    with tempfile.TemporaryDirectory() as tmp:
        ctx_path = os.path.join(tmp, "context.json")
        run_path = os.path.join(tmp, "runner.py")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(context, f)
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(runner.replace("{{CTX}}", ctx_path))

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


