"""Client for mcp-xp's "Galaxy Runs Through Chat" REST API
(/v1/galaxy/..., /v1/datasets/...) -- a separate, newer mechanism from
galaxy.py's MCP-based GalaxyHandler, which stays unchanged alongside this
(confirmed by mcp-xp: the MCP tools at GALAXY_MCP_SERVER are untouched).

Per the spec, the assistant's own job is read-only and propose-only: find a
tool/workflow, read its real input schema, see which thread files match,
and check a tentative input set is acceptable to Galaxy -- never upload,
download, start, track or cancel a run. Those are the platform server's
job, calling mcp-xp directly with the user's JWT, and are never reached
from this codebase.

Only the four read-only routes below are implemented, because they're the
only ones mcp-xp has confirmed live and the only ones the assistant is
spec'd to call at all:
    GET  /v1/galaxy/search
    POST /v1/galaxy/describe
    GET  /v1/galaxy/threads/{thread_id}/files
    POST /v1/galaxy/validate

Deliberately NOT implemented here -- space is left for them only if the
assistant's own scope ever changes, since today they belong to the
platform server, not us:
    POST /v1/galaxy/runs                       (start a run)
    GET  /v1/galaxy/runs[?thread_id=] / {id}    (list/track a run)
    POST /v1/galaxy/runs/{id}/cancel            (cancel a run)
    POST /v1/galaxy/links                       (Open in Galaxy)
    POST /v1/datasets/upload                    (upload a file)
    GET  /v1/datasets/{id}/download              (download a file)
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GALAXY_RUNS_API_BASE_URL = os.getenv("GALAXY_RUNS_API_BASE_URL")


class GalaxyRunsAPI:
    """Thin wrapper, same shape as hypothesis_generation.HypothesisGeneration's
    _make_api_request -- every method returns the real response dict, or
    {"error": "..."} with mcp-xp's or the network's own message, never a
    guessed/invented one."""

    def _make_api_request(
        self,
        method: str,
        url: str,
        token: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if not GALAXY_RUNS_API_BASE_URL:
            logger.error("GALAXY_RUNS_API_BASE_URL is not set")
            return {"error": "the Galaxy runs service isn't configured"}
        headers = {"Authorization": f"Bearer {token}"}
        try:
            logger.debug(f"Making {method} request to {url} with data {data} and params {params}")
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=15)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Galaxy runs API request to {url} timed out")
            return {"error": "Galaxy runs service timed out"}
        except requests.exceptions.HTTPError as e:
            detail = None
            try:
                body = e.response.json() if e.response is not None else None
                if isinstance(body, dict):
                    detail = body.get("detail")
            except ValueError:
                pass
            status = e.response.status_code if e.response is not None else "?"
            message = detail or f"HTTP {status}"
            logger.error(f"Galaxy runs API request failed: {message}")
            return {"error": message}
        except requests.exceptions.RequestException as e:
            logger.error(f"Galaxy runs API request failed: {e}")
            return {"error": "Request failed -- could not reach the Galaxy runs service"}

    def search(self, token: str, query: str, kind: Optional[str] = None) -> Dict[str, Any]:
        """GET /v1/galaxy/search?q=&kind= -- installed, runnable tools/workflows
        matching query. kind is 'tool' or 'workflow', or omitted for both."""
        params = {"q": query}
        if kind:
            params["kind"] = kind
        return self._make_api_request(
            "GET", f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/search", token, params=params
        )

    def describe(
        self, token: str, thread_id: str, target: Dict[str, Any], full: bool = False
    ) -> Dict[str, Any]:
        """POST /v1/galaxy/describe -- a target's real input schema: required
        inputs, types, conditionals, accepted formats, and which of the
        thread's own files already match a file input. full=True asks for
        every setting (defaults included), used to build all_values later."""
        url = f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/describe"
        if full:
            url += "?full=true"
        return self._make_api_request(
            "POST", url, token, data={"thread_id": thread_id, "target": target}
        )

    def thread_files(self, token: str, thread_id: str) -> Dict[str, Any]:
        """GET /v1/galaxy/threads/{thread_id}/files -- every dataset already in
        this thread's Galaxy history, Galaxy's own ground truth of what's
        available to use as a file input."""
        return self._make_api_request(
            "GET", f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/threads/{thread_id}/files", token
        )

    def validate(
        self,
        token: str,
        thread_id: str,
        target: Dict[str, Any],
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST /v1/galaxy/validate -- {valid, errors} for a tentative
        target/inputs/parameters, Galaxy's own check, called before a
        proposal card is ever shown as ready."""
        return self._make_api_request(
            "POST",
            f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/validate",
            token,
            data={
                "thread_id": thread_id,
                "target": target,
                "inputs": inputs,
                "parameters": parameters or {},
            },
        )
