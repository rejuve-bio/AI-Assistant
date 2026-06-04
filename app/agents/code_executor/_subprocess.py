import subprocess
import tempfile
import os
import shutil
import resource
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 300
OUTPUTS_DIR = "uploads/outputs"  # persistent output storage

TOOL_CMD = {
    "python": ("python3", "script.py"),
    "R":      ("Rscript", "script.R"),
    "plink":  ("bash",    "script.sh"),
    "bash":   ("bash",    "script.sh"),
}

TOOL_EXT = {
    "python": "script.py",
    "R":      "script.R",
    "plink":  "script.sh",
    "bash":   "script.sh",
}


def _set_resource_limits():
    """Apply memory + process limits to the child process."""
    try:
        resource.setrlimit(resource.RLIMIT_AS,    (2 * 1024**3, 2 * 1024**3))  # 2 GB virtual memory
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))                   # max subprocesses
    except Exception:
        pass


class Subprocess:
    """Runs generated scripts directly inside the app container."""

    def run(self, code: str, tool: str, image: str = None,
            file_paths: Optional[List[str]] = None,
            user_id: str = None, step_id: int = None, session_id: str = None) -> dict:

        workdir = tempfile.mkdtemp(prefix="ai_sandbox_")
        input_dir  = os.path.join(workdir, "input")
        output_dir = os.path.join(workdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Copy user files into input/
        if file_paths:
            for src in file_paths:
                if src and os.path.exists(src):
                    dst = os.path.join(input_dir, os.path.basename(src))
                    try:
                        shutil.copy2(src, dst)
                        logger.info(f"Copied {src} → input/")
                    except Exception as e:
                        logger.warning(f"Could not copy {src}: {e}")
                else:
                    logger.warning(f"File not found, skipping: {src}")

        script_name = TOOL_EXT.get(tool, "script.py")
        script_path = os.path.join(workdir, script_name)
        with open(script_path, "w") as f:
            f.write(code)

        interpreter, script_arg = TOOL_CMD.get(tool, ("python3", "script.py"))

        # Build subprocess environment:
        # 1. PYTHONPATH → project root so scripts can import app.tools.biomni.*
        # 2. BIOMNI_DATA_LAKE → parquet data directory (inherited from app env)
        # 3. R_LIBS_USER → ensure R finds Bioconductor packages installed in Docker
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        env = os.environ.copy()

        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{existing_py}" if existing_py else project_root

        # Ensure BIOMNI_DATA_LAKE is forwarded (may be set in .env or docker-compose)
        if "BIOMNI_DATA_LAKE" not in env:
            env["BIOMNI_DATA_LAKE"] = "/data/biomni"

        # Help R find system-installed Bioconductor packages
        if "R_LIBS_SITE" not in env:
            env["R_LIBS_SITE"] = "/usr/local/lib/R/library:/usr/lib/R/library"

        logger.info(f"Running {interpreter} {script_arg} in {workdir} files={len(file_paths or [])}")

        try:
            proc = subprocess.run(
                [interpreter, script_arg],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=workdir,
                env=env,
                preexec_fn=_set_resource_limits,
            )
            output_files = self._collect_outputs(output_dir, user_id, session_id, step_id)
            return {
                "success":      proc.returncode == 0,
                "exit_code":    proc.returncode,
                "stdout":       proc.stdout[:8000],
                "stderr":       proc.stderr[:4000],
                "error":        "",
                "output_files": output_files,
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Execution timed out after {TIMEOUT_SECONDS}s")
            return {
                "success": False, "exit_code": -1,
                "stdout": "", "stderr": "",
                "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds.",
                "output_files": [],
            }
        except FileNotFoundError as e:
            logger.error(f"Interpreter not found: {e}")
            return {
                "success": False, "exit_code": -1,
                "stdout": "", "stderr": "",
                "error": f"Interpreter '{interpreter}' not found: {e}",
                "output_files": [],
            }
        except Exception as e:
            logger.error(f"Sandbox error: {e}", exc_info=True)
            return {
                "success": False, "exit_code": -1,
                "stdout": "", "stderr": "",
                "error": str(e),
                "output_files": [],
            }
        finally:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    def _collect_outputs(self, output_dir: str, user_id: str, session_id: str, step_id) -> List[str]:
        """Copy files from output/ to a persistent location and return their accessible paths."""
        saved = []
        if not os.path.isdir(output_dir):
            return saved
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        if not files:
            return saved

        dest_dir = os.path.join(OUTPUTS_DIR, str(user_id or "unknown"), str(session_id or "0"), str(step_id or "0"))
        os.makedirs(dest_dir, exist_ok=True)

        for fname in files:
            src = os.path.join(output_dir, fname)
            dst = os.path.join(dest_dir, fname)
            try:
                shutil.copy2(src, dst)
                saved.append(dst)
                logger.info(f"Saved output: {dst}")
            except Exception as e:
                logger.warning(f"Could not save output {fname}: {e}")
        return saved
