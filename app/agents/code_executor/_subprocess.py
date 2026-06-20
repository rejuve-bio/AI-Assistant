import base64
import io
import json
import logging
import os
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import List, Optional


logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = int(os.getenv("E2B_EXECUTION_TIMEOUT", "300"))
TEMPLATE_ID = os.getenv("E2B_TEMPLATE_ID", "code-interpreter-v1")
OUTPUTS_DIR = Path("uploads/outputs")
REMOTE_ROOT = PurePosixPath("/home/user")
REMOTE_INPUT_DIR = REMOTE_ROOT / "input"
REMOTE_OUTPUT_DIR = REMOTE_ROOT / "output"

TOOL_COMMANDS = {
    "R": ["Rscript", "script.R"],
    "plink": ["bash", "script.sh"],
    "bash": ["bash", "script.sh"],
}

TOOL_EXT = {
    "python": "script.py",
    "R": "script.R",
    "plink": "script.sh",
    "bash": "script.sh",
}


def _get_api_key() -> str | None:
    return os.getenv("E2B_API_KEY") or os.getenv("E2B_Sandbox_KEY")


@lru_cache(maxsize=1)
def _biomni_runtime_bundle() -> bytes:
    source_dir = Path(__file__).resolve().parents[2] / "tools" / "biomni"
    if not source_dir.is_dir():
        raise RuntimeError(f"Biomni runtime not found at {source_dir}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("app/__init__.py", "")
        archive.writestr("app/tools/__init__.py", "")
        # Avoid importing the host-only retriever and SentenceTransformer.
        archive.writestr("app/tools/biomni/__init__.py", "")
        for source_path in source_dir.rglob("*"):
            if not source_path.is_file():
                continue
            if source_path.name == "__init__.py" or "__pycache__" in source_path.parts:
                continue
            relative_path = source_path.relative_to(source_dir)
            archive.write(
                source_path,
                str(PurePosixPath("app/tools/biomni") / relative_path),
            )
    return buffer.getvalue()


class E2BBackend:
    """
    Backward-compatible code runner powered by the E2B SDK.

    Generated code and uploaded files are executed in an isolated E2B
    microVM. Nothing is executed as a child process of the Flask application.
    """

    def __init__(self):
        self.api_key = _get_api_key()
        if not self.api_key:
            logger.warning(
                "E2B API key is not configured. Set E2B_API_KEY or E2B_Sandbox_KEY."
            )

    def run(
        self,
        code: str,
        tool: str,
        image: str = None,
        file_paths: Optional[List[str]] = None,
        user_id: str = None,
        step_id: int = None,
        session_id: str = None,
    ) -> dict:
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            return self._error_result(
                "e2b-code-interpreter is not installed.", backend_error=True
            )

        if not self.api_key:
            return self._error_result(
                "E2B API key is not configured.", backend_error=True
            )

        sandbox = None
        try:
            sandbox = Sandbox.create(
                template=TEMPLATE_ID,
                timeout=TIMEOUT_SECONDS,
                api_key=self.api_key,
            )
            self._prepare_workspace(sandbox)
            self._install_biomni_runtime(sandbox)
            upload_errors = self._upload_files(sandbox, file_paths or [])
            if upload_errors:
                return self._error_result("; ".join(upload_errors))

            execution = self._execute(sandbox, code, tool)
            stdout = "\n".join(execution.logs.stdout) if execution.logs else ""
            stderr = "\n".join(execution.logs.stderr) if execution.logs else ""
            error_text = str(execution.error) if execution.error else ""

            output_files = self._download_outputs(
                sandbox,
                user_id=user_id,
                session_id=session_id,
                step_id=step_id,
            )

            plots = []
            for result in execution.results or []:
                if getattr(result, "png", None):
                    plots.append({"type": "png", "data": result.png})
                elif getattr(result, "svg", None):
                    plots.append({"type": "svg", "data": result.svg})

            return {
                "success": not bool(execution.error),
                "exit_code": 1 if execution.error else 0,
                "stdout": stdout[:8000],
                "stderr": stderr[:4000],
                "error": error_text,
                "output_files": output_files,
                "plots": plots,
                "backend": "e2b",
                "backend_error": False,
            }
        except Exception as exc:
            logger.error("E2B execution failed: %s", exc, exc_info=True)
            return self._error_result(
                str(exc),
                backend_error=self._is_backend_error(str(exc)),
            )
        finally:
            if sandbox is not None:
                try:
                    if hasattr(sandbox, "kill"):
                        sandbox.kill()
                    elif hasattr(sandbox, "close"):
                        sandbox.close()
                except Exception:
                    pass

    @staticmethod
    def _prepare_workspace(sandbox) -> None:
        sandbox.run_code(
            (
                "from pathlib import Path\n"
                f"Path({str(REMOTE_INPUT_DIR)!r}).mkdir(parents=True, exist_ok=True)\n"
                f"Path({str(REMOTE_OUTPUT_DIR)!r}).mkdir(parents=True, exist_ok=True)\n"
            ),
            timeout=20,
        )

    @staticmethod
    def _install_biomni_runtime(sandbox) -> None:
        archive_path = REMOTE_ROOT / "biomni_runtime.zip"
        sandbox.files.write(str(archive_path), _biomni_runtime_bundle())
        result = sandbox.run_code(
            (
                "from zipfile import ZipFile\n"
                f"with ZipFile({str(archive_path)!r}) as archive:\n"
                f"    archive.extractall({str(REMOTE_ROOT)!r})\n"
            ),
            timeout=20,
        )
        if result.error:
            raise RuntimeError(f"Could not install Biomni runtime: {result.error}")

    @staticmethod
    def _upload_files(sandbox, file_paths: List[str]) -> list[str]:
        errors = []
        for local_path in file_paths:
            if not local_path or not os.path.isfile(local_path):
                errors.append(f"Input file not found: {local_path}")
                continue

            remote_path = REMOTE_INPUT_DIR / os.path.basename(local_path)
            try:
                sandbox.files.write(str(remote_path), Path(local_path).read_bytes())
            except Exception as exc:
                errors.append(f"Could not upload {local_path}: {exc}")
        return errors

    @staticmethod
    def _execute(sandbox, code: str, tool: str):
        if tool == "python":
            wrapped_code = (
                "import os\n"
                f"os.chdir({str(REMOTE_ROOT)!r})\n"
                + code
            )
            return sandbox.run_code(wrapped_code, timeout=TIMEOUT_SECONDS)

        script_name = TOOL_EXT.get(tool, "script.sh")
        command = TOOL_COMMANDS.get(tool, ["bash", script_name])
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")

        wrapper = (
            "import base64\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"os.chdir({str(REMOTE_ROOT)!r})\n"
            f"script = Path({script_name!r})\n"
            f"script.write_bytes(base64.b64decode({encoded_code!r}))\n"
            f"proc = subprocess.run({command!r}, capture_output=True, text=True)\n"
            "if proc.stdout:\n"
            "    print(proc.stdout, end='')\n"
            "if proc.stderr:\n"
            "    print(proc.stderr, end='', file=sys.stderr)\n"
            "if proc.returncode:\n"
            "    raise RuntimeError(f'Command exited with code {proc.returncode}')\n"
        )
        return sandbox.run_code(wrapper, timeout=TIMEOUT_SECONDS)

    def _download_outputs(
        self,
        sandbox,
        *,
        user_id: str,
        session_id: str,
        step_id: int,
    ) -> list[str]:
        listing = sandbox.run_code(
            (
                "import json\n"
                "from pathlib import Path\n"
                f"root = Path({str(REMOTE_OUTPUT_DIR)!r})\n"
                "print(json.dumps([str(p) for p in root.rglob('*') if p.is_file()]))\n"
            ),
            timeout=20,
        )
        lines = listing.logs.stdout if listing.logs else []
        if not lines:
            return []

        try:
            remote_files = json.loads(lines[-1])
        except (TypeError, json.JSONDecodeError):
            logger.warning("Could not parse E2B output file listing: %s", lines)
            return []

        destination = (
            OUTPUTS_DIR
            / str(user_id or "unknown")
            / str(session_id or "0")
            / str(step_id or "0")
        )
        destination.mkdir(parents=True, exist_ok=True)

        saved = []
        for remote_path in remote_files:
            try:
                content = sandbox.files.read(remote_path)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                local_path = destination / PurePosixPath(remote_path).name
                local_path.write_bytes(content)
                saved.append(str(local_path))
            except Exception as exc:
                logger.warning("Could not download %s: %s", remote_path, exc)
        return saved

    @staticmethod
    def _is_backend_error(error: str) -> bool:
        lowered = error.lower()
        return any(
            marker in lowered
            for marker in (
                "api key",
                "unauthorized",
                "connection",
                "bad gateway",
                "port is not open",
                "template",
                "timeout",
            )
        )

    @staticmethod
    def _error_result(message: str, backend_error: bool = False) -> dict:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": message,
            "error": message,
            "output_files": [],
            "plots": [],
            "backend": "e2b",
            "backend_error": backend_error,
        }


class DockerBackend:
    """Hardened local Docker fallback for E2B infrastructure failures."""

    def __init__(self):
        self.image = os.getenv(
            "ACTION_SANDBOX_DOCKER_IMAGE",
            "rejuv-coding-sandbox:latest",
        )
        self.timeout = int(
            os.getenv("ACTION_SANDBOX_DOCKER_TIMEOUT", str(TIMEOUT_SECONDS))
        )
        self.memory = os.getenv("ACTION_SANDBOX_DOCKER_MEMORY", "2g")
        self.cpus = float(os.getenv("ACTION_SANDBOX_DOCKER_CPUS", "1"))
        self.pids_limit = int(os.getenv("ACTION_SANDBOX_DOCKER_PIDS_LIMIT", "256"))
        allow_internet = (
            os.getenv("ACTION_SANDBOX_ALLOW_DOCKER_INTERNET", "false").lower()
            == "true"
        )
        self.network = os.getenv(
            "ACTION_SANDBOX_DOCKER_NETWORK",
            "bridge" if allow_internet else "none",
        )

    def run(
        self,
        code: str,
        tool: str,
        image: str = None,
        file_paths: Optional[List[str]] = None,
        user_id: str = None,
        step_id: int = None,
        session_id: str = None,
    ) -> dict:
        try:
            import docker
            from docker.errors import ImageNotFound
        except ImportError:
            return self._error_result(
                "Docker SDK for Python is not installed.", backend_error=True
            )

        container = None
        try:
            client = docker.from_env()
            selected_image = image or self.image
            try:
                client.images.get(selected_image)
            except ImageNotFound:
                return self._error_result(
                    f"Docker sandbox image {selected_image!r} is not available.",
                    backend_error=True,
                )

            container = client.containers.create(
                image=selected_image,
                command=["sleep", str(max(self.timeout + 30, 60))],
                detach=True,
                working_dir=str(REMOTE_ROOT),
                network_mode=self.network,
                mem_limit=self.memory,
                nano_cpus=int(self.cpus * 1_000_000_000),
                pids_limit=self.pids_limit,
                read_only=True,
                tmpfs={
                    "/tmp": "rw,nosuid,nodev,size=128m,uid=1000,gid=1000",
                    str(REMOTE_ROOT): "rw,nosuid,nodev,size=1g,uid=1000,gid=1000",
                },
                user="sandbox",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
            )
            container.start()
            self._install_biomni_runtime(container)

            script_name = TOOL_EXT.get(tool, "script.py")
            self._write_bytes(
                container,
                str(REMOTE_ROOT / script_name),
                code.encode("utf-8"),
            )
            for local_path in file_paths or []:
                if not local_path or not os.path.isfile(local_path):
                    return self._error_result(f"Input file not found: {local_path}")
                self._write_bytes(
                    container,
                    str(REMOTE_INPUT_DIR / os.path.basename(local_path)),
                    Path(local_path).read_bytes(),
                )

            command = self._command(tool, script_name)
            exec_result = container.exec_run(
                ["sh", "-lc", f"timeout {self.timeout} {command}"],
                demux=True,
            )
            stdout_raw, stderr_raw = exec_result.output or (b"", b"")
            stdout = self._decode(stdout_raw)
            stderr = self._decode(stderr_raw)
            exit_code = int(exec_result.exit_code or 0)
            if exit_code == 124:
                stderr = (
                    stderr + f"\nDocker sandbox timed out after {self.timeout}s."
                ).strip()

            output_files = self._download_outputs(
                container,
                user_id=user_id,
                session_id=session_id,
                step_id=step_id,
            )
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout[:8000],
                "stderr": stderr[:4000],
                "error": stderr if exit_code else "",
                "output_files": output_files,
                "plots": [],
                "backend": "docker",
                "backend_error": False,
            }
        except Exception as exc:
            logger.error("Docker sandbox execution failed: %s", exc, exc_info=True)
            return self._error_result(str(exc), backend_error=True)
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    @staticmethod
    def _command(tool: str, script_name: str) -> str:
        if tool == "python":
            return f"python {script_name}"
        if tool == "R":
            return f"Rscript {script_name}"
        return f"bash {script_name}"

    def _install_biomni_runtime(self, container) -> None:
        archive_path = str(REMOTE_ROOT / "biomni_runtime.zip")
        self._write_bytes(container, archive_path, _biomni_runtime_bundle())
        extract_code = (
            "from zipfile import ZipFile\n"
            f"with ZipFile({archive_path!r}) as archive:\n"
            f"    archive.extractall({str(REMOTE_ROOT)!r})\n"
        )
        result = container.exec_run(["python", "-c", extract_code], demux=True)
        if result.exit_code != 0:
            raise RuntimeError(f"Could not install Biomni runtime: {result.output}")

    @staticmethod
    def _write_bytes(container, remote_path: str, content: bytes) -> None:
        init_code = (
            "from pathlib import Path\n"
            f"path = Path({remote_path!r})\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "path.write_bytes(b'')\n"
        )
        result = container.exec_run(["python", "-c", init_code], demux=True)
        if result.exit_code != 0:
            raise RuntimeError(f"Could not prepare {remote_path}: {result.output}")

        for index in range(0, len(content), 24_000):
            encoded = base64.b64encode(content[index : index + 24_000]).decode(
                "ascii"
            )
            append_code = (
                "import base64\n"
                "from pathlib import Path\n"
                f"path = Path({remote_path!r})\n"
                f"chunk = base64.b64decode({encoded!r})\n"
                "with path.open('ab') as handle:\n"
                "    handle.write(chunk)\n"
            )
            result = container.exec_run(["python", "-c", append_code], demux=True)
            if result.exit_code != 0:
                raise RuntimeError(f"Could not write {remote_path}: {result.output}")

    def _download_outputs(
        self,
        container,
        *,
        user_id: str,
        session_id: str,
        step_id: int,
    ) -> list[str]:
        list_code = (
            "from pathlib import Path\n"
            f"root = Path({str(REMOTE_OUTPUT_DIR)!r})\n"
            "for path in root.rglob('*'):\n"
            "    if path.is_file(): print(str(path))\n"
        )
        result = container.exec_run(["python", "-c", list_code], demux=True)
        stdout_raw, _ = result.output or (b"", b"")
        remote_files = [
            line.strip()
            for line in self._decode(stdout_raw).splitlines()
            if line.strip()
        ]

        destination = (
            OUTPUTS_DIR
            / str(user_id or "unknown")
            / str(session_id or "0")
            / str(step_id or "0")
        )
        destination.mkdir(parents=True, exist_ok=True)

        saved = []
        for remote_path in remote_files:
            read_code = (
                "import base64\n"
                "from pathlib import Path\n"
                f"print(base64.b64encode(Path({remote_path!r}).read_bytes()).decode('ascii'))\n"
            )
            read_result = container.exec_run(
                ["python", "-c", read_code],
                demux=True,
            )
            content_raw, _ = read_result.output or (b"", b"")
            encoded = self._decode(content_raw).strip()
            if not encoded:
                continue
            local_path = destination / PurePosixPath(remote_path).name
            local_path.write_bytes(base64.b64decode(encoded))
            saved.append(str(local_path))
        return saved

    @staticmethod
    def _decode(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return value.decode("utf-8", errors="replace")

    @staticmethod
    def _error_result(message: str, backend_error: bool = False) -> dict:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": message,
            "error": message,
            "output_files": [],
            "plots": [],
            "backend": "docker",
            "backend_error": backend_error,
        }


class Subprocess:
    """Select E2B as primary and Docker as an optional hardened fallback."""

    def __init__(self):
        self.primary = os.getenv("ACTION_SANDBOX_BACKEND", "e2b").lower()
        self.fallback = os.getenv("ACTION_SANDBOX_FALLBACK", "docker").lower()
        self.e2b = E2BBackend()
        self.docker = DockerBackend()

    def run(self, **kwargs) -> dict:
        primary_result = self._backend(self.primary).run(**kwargs)
        if not self._should_fallback(primary_result):
            return primary_result

        if not self.fallback or self.fallback == self.primary:
            return primary_result

        logger.warning(
            "Sandbox backend %s failed; falling back to %s",
            primary_result.get("backend"),
            self.fallback,
        )
        fallback_result = self._backend(self.fallback).run(**kwargs)
        fallback_result["fallback_from"] = primary_result.get("backend")
        fallback_result["primary_error"] = primary_result.get("error")
        return fallback_result

    def _backend(self, name: str):
        return self.docker if name == "docker" else self.e2b

    @staticmethod
    def _should_fallback(result: dict) -> bool:
        return bool(result.get("backend_error"))
