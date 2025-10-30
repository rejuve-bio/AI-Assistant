import pytest


def test_normalize_inputs_smoke():
    from app.code_exec import loaders

    out = loaders.normalize_inputs(files=[], urls=[])
    assert isinstance(out, dict)
    assert "tables" in out and "texts" in out and "metadata" in out


def test_sandbox_runs_empty_code():
    from app.code_exec.sandbox import run_python, SandboxLimits

    res = run_python(code="print('ok')", context={}, limits=SandboxLimits(timeout_seconds=2, max_memory_mb=128))
    assert "stdout" in res and "stderr" in res

