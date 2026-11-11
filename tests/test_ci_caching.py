from pathlib import Path

import yaml


def test_docker_cache_export_is_best_effort_only_for_github():
    workflow_path = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci-caching.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["warm-docker-cache"]["steps"]
    build = next(
        step for step in steps
        if step.get("uses", "").startswith("docker/build-push-action@")
    )
    exporters = {}
    for line in build["with"]["cache-to"].splitlines():
        options = dict(option.split("=", 1) for option in line.split(","))
        exporters[options["type"]] = options

    assert exporters["gha"]["ignore-error"] == "true"
    assert exporters["gha"]["mode"] == "max"
    assert exporters["local"]["dest"] == "/tmp/.buildx-cache-new"
    assert exporters["local"].get("ignore-error", "false") == "false"
    assert not build.get("continue-on-error", False)
