import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.routing import APIRoute

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def openapi_spec():
    saved_app_modules = {
        k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")
    }
    for k in list(saved_app_modules):
        sys.modules.pop(k, None)

    pkg = types.ModuleType("app")
    pkg.__path__ = [str(REPO_ROOT / "app")]
    sys.modules["app"] = pkg

    try:
        with patch("pymongo.MongoClient"), patch("redis.from_url"):
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
            yield app.openapi(), main_router
    finally:
        for k in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved_app_modules)


def test_all_router_endpoints_present_in_openapi(openapi_spec):
    spec, main_router = openapi_spec
    paths = spec["paths"]

    expected = {
        (route.path, method.lower())
        for route in main_router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    }
    actual = {
        (path, method.lower())
        for path, item in paths.items()
        for method in item
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }
    assert expected == actual
    assert len(actual) == 21


def test_bearer_security_applied_except_health_check(openapi_spec):
    spec, _ = openapi_spec
    assert "BearerAuth" in spec["components"]["securitySchemes"]

    for path, item in spec["paths"].items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            if (path, method.lower()) == ("/", "get"):
                assert op["security"] == []
            else:
                assert op["security"] == [{"BearerAuth": []}]
            for param in op.get("parameters", []):
                assert not (
                    param.get("in") == "header"
                    and param.get("name", "").lower() == "authorization"
                )


def test_get_operations_have_no_request_body(openapi_spec):
    spec, _ = openapi_spec
    for path, item in spec["paths"].items():
        if "get" in item:
            assert "requestBody" not in item["get"], f"GET {path} must not have requestBody"


def test_all_schema_refs_resolve(openapi_spec):
    spec, _ = openapi_spec
    defined = set(spec["components"]["schemas"].keys())
    referenced = set(re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(spec)))
    missing = referenced - defined
    assert not missing, f"Unresolved schema refs: {missing}"


def test_v1_request_bodies_and_legacy_deprecation(openapi_spec):
    spec, _ = openapi_spec
    paths = spec["paths"]

    v1_query_ct = paths["/v1/query"]["post"]["requestBody"]["content"]
    assert "application/json" in v1_query_ct
    assert "multipart/form-data" in v1_query_ct

    assert "application/json" in paths["/v1/threads/{thread_id}/context"]["put"]["requestBody"]["content"]
    assert "application/json" in paths["/v1/threads/{thread_id}/messages"]["post"]["requestBody"]["content"]
    assert "application/json" in paths["/v1/threads/{thread_id}"]["patch"]["requestBody"]["content"]

    legacy_paths = [p for p in paths if not p.startswith("/v1/") and p != "/"]
    for path in legacy_paths:
        for method, op in paths[path].items():
            if isinstance(op, dict):
                assert op.get("deprecated") is True, f"{method.upper()} {path} should be deprecated"


def test_socketio_documented_in_openapi(openapi_spec):
    spec, _ = openapi_spec
    assert "/socket.io/" in spec["info"]["description"]
    socketio_ext = spec["x-socketio"]
    assert socketio_ext["path"] == "/socket.io/"
    assert set(socketio_ext["client_events"].keys()) == {
        "connect",
        "disconnect",
        "join_room",
        "question",
    }
    assert set(socketio_ext["server_events"].keys()) == {"response", "update", "error"}


def test_exported_openapi_files_match_app(openapi_spec):
    spec, _ = openapi_spec
    json_path = REPO_ROOT / "docs" / "openapi.json"
    yaml_path = REPO_ROOT / "docs" / "openapi.yaml"

    assert json_path.exists(), "docs/openapi.json missing — run scripts/export_openapi.py"
    assert yaml_path.exists(), "docs/openapi.yaml missing — run scripts/export_openapi.py"

    assert json.loads(json_path.read_text(encoding="utf-8")) == spec
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == spec
