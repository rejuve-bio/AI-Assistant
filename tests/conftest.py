"""
tests/conftest.py

Provides helper utilities for importing app submodules WITHOUT triggering
the full app/__init__.py import chain (which loads transformers, torch,
LLMs, Redis, etc. and OOM-kills the container).

Strategy: Use importlib.util to load individual .py files directly.
"""
from __future__ import annotations

import importlib.util
import importlib.machinery
import logging
import os
import sys
import types
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prevent app/__init__.py from ever being executed during test collection.
# We replace the 'app' package with a thin shell module that lets
# `from app.x.y import Z` work by resolving submodules from the filesystem.
# ---------------------------------------------------------------------------

_APP_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "app")
)


class _AppFinder(importlib.abc.MetaPathFinder):
    """
    Custom import finder that intercepts `app.*` imports and loads individual
    Python files from the app/ directory WITHOUT executing app/__init__.py.
    This prevents the heavy Flask/ML startup from OOM-killing the container.
    """

    def find_module(self, fullname, path=None):
        if fullname == "app" or fullname.startswith("app."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        parts = fullname.split(".")
        # Build the filesystem path
        rel = os.path.join(*parts[1:]) if len(parts) > 1 else ""
        dir_path = os.path.join(_APP_ROOT, rel)
        file_path = os.path.join(_APP_ROOT, rel + ".py") if rel else None

        if fullname == "app":
            # Create a thin namespace package — no __init__.py execution
            mod = types.ModuleType("app")
            mod.__path__ = [_APP_ROOT]
            mod.__package__ = "app"
            mod.__spec__ = importlib.machinery.ModuleSpec(
                "app", None, is_package=True
            )
            sys.modules["app"] = mod
            return mod

        if os.path.isdir(dir_path):
            # It's a sub-package (e.g. app.rag, app.storage)
            init_file = os.path.join(dir_path, "__init__.py")
            mod = types.ModuleType(fullname)
            mod.__path__ = [dir_path]
            mod.__package__ = fullname
            mod.__spec__ = importlib.machinery.ModuleSpec(
                fullname, None, is_package=True
            )
            sys.modules[fullname] = mod

            # Execute the sub-package __init__.py ONLY if it exists and is NOT
            # the top-level app/__init__.py (which is the heavy one).
            if os.path.isfile(init_file):
                spec = importlib.util.spec_from_file_location(
                    fullname, init_file,
                    submodule_search_locations=[dir_path],
                )
                if spec and spec.loader:
                    spec.loader.exec_module(mod)
            return mod

        if file_path and os.path.isfile(file_path):
            # It's a module file (e.g. app.rag.utils.content_processor)
            spec = importlib.util.spec_from_file_location(fullname, file_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = ".".join(parts[:-1])
                sys.modules[fullname] = mod
                spec.loader.exec_module(mod)
                return mod

        raise ModuleNotFoundError(f"conftest _AppFinder: cannot find {fullname}")


# Install our custom finder BEFORE Python's default finders
sys.meta_path.insert(0, _AppFinder())

# ---------------------------------------------------------------------------
# Stub heavy third-party packages that individual app modules import
# ---------------------------------------------------------------------------


def _stub(name: str) -> MagicMock:
    """MagicMock with __spec__ so importlib.util.find_spec() passes."""
    m = MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__path__ = []
    return m


# sentence_transformers — prevent model loading
_mock_st = MagicMock()
_mock_st.encode.return_value = [[0.1] * 384]
_mock_ce = MagicMock()
_mock_ce.predict.return_value = [0.9, 0.5, 0.3]

_st = _stub("sentence_transformers")
_st.SentenceTransformer.return_value = _mock_st
_st.CrossEncoder.return_value = _mock_ce
sys.modules["sentence_transformers"] = _st
sys.modules["sentence_transformers.cross_encoder"] = _stub("sentence_transformers.cross_encoder")
sys.modules["sentence_transformers.cross_encoder"].CrossEncoder.return_value = _mock_ce

# apscheduler
sys.modules["apscheduler"] = _stub("apscheduler")
sys.modules["apscheduler.schedulers"] = _stub("apscheduler.schedulers")
sys.modules["apscheduler.schedulers.background"] = _stub("apscheduler.schedulers.background")

# pymongo
_mongo_client = MagicMock()
_mongo_db = MagicMock()
_mongo_col = MagicMock()
_mongo_col.find.return_value = []
_mongo_db.__getitem__.return_value = _mongo_col
_mongo_client.__getitem__.return_value = _mongo_db
_pymongo = _stub("pymongo")
_pymongo.MongoClient.return_value = _mongo_client
sys.modules["pymongo"] = _pymongo

# redis
sys.modules["redis"] = _stub("redis")

# qdrant_client
sys.modules["qdrant_client"] = _stub("qdrant_client")
_qdrant_http = _stub("qdrant_client.http")
sys.modules["qdrant_client.http"] = _qdrant_http

_qm = _stub("qdrant_client.http.models")
from collections import namedtuple as _nt
_qm.PointStruct = _nt("PointStruct", ["id", "vector", "payload"])
sys.modules["qdrant_client.http.models"] = _qm
_qdrant_http.models = _qm

# langchain-google-genai (pulls transformers → torch)
sys.modules["langchain_google_genai"] = _stub("langchain_google_genai")

# eventlet / gunicorn (sometimes imported by socketio)
sys.modules["eventlet"] = _stub("eventlet")

# flask-socketio
sys.modules["flask_socketio"] = _stub("flask_socketio")

# flask_limiter
sys.modules["flask_limiter"] = _stub("flask_limiter")
sys.modules["flask_limiter.util"] = _stub("flask_limiter.util")


# transformers + torch — pulled in by langchain_core, ~1 GB combined
for _t_name in [
    "torch", "torch.nn", "torch.nn.functional", "torch.utils",
    "torch.utils.data", "torch.cuda",
    "transformers", "transformers.utils", "transformers.utils.import_utils",
    "transformers.utils.versions", "transformers.utils.generic",
    "transformers.utils.auto_docstring", "transformers.dependency_versions_check",
]:
    sys.modules[_t_name] = _stub(_t_name)

# Logging file handler
import logging.handlers as _lh
_lh.TimedRotatingFileHandler = MagicMock()  # type: ignore[misc]

logger.info("conftest.py: app import bypass and stubs installed")
