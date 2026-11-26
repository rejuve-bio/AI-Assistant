"""Export the FastAPI OpenAPI specification to docs/openapi.json and docs/openapi.yaml.

Usage:
    python scripts/export_openapi.py
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_lightweight_app_package() -> None:
    """Register `app` as a namespace package pointing at ./app so importing
    `app.routes` and `app.openapi` does not execute `app/__init__.py`."""
    if "app" not in sys.modules:
        pkg = types.ModuleType("app")
        pkg.__path__ = [str(REPO_ROOT / "app")]
        sys.modules["app"] = pkg


def build_openapi_spec() -> dict:
    """Build the OpenAPI spec from main_router and app.openapi without external services."""
    _ensure_lightweight_app_package()

    with patch("pymongo.MongoClient"), patch("redis.from_url"):
        from fastapi import FastAPI
        from app.openapi import (
            API_DESCRIPTION,
            API_TITLE,
            API_VERSION,
            TAGS_METADATA,
            configure_openapi,
        )
        from app.routes import main_router

        app = FastAPI(
            title=API_TITLE,
            version=API_VERSION,
            description=API_DESCRIPTION,
            openapi_tags=TAGS_METADATA,
        )
        app.include_router(main_router)
        configure_openapi(app)
        return app.openapi()


def main() -> None:
    spec = build_openapi_spec()
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    json_path = docs_dir / "openapi.json"
    yaml_path = docs_dir / "openapi.yaml"

    json_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(
        f"Wrote {json_path} and {yaml_path} "
        f"({len(spec['paths'])} paths, {len(spec['components']['schemas'])} schemas)"
    )


if __name__ == "__main__":
    main()
