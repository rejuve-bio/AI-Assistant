import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GALAXY_RUNS_API_BASE_URL = os.getenv("GALAXY_RUNS_API_BASE_URL")


class GalaxyRunsAPI:

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
        params = {"q": query}
        if kind:
            params["kind"] = kind
        return self._make_api_request(
            "GET", f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/search", token, params=params
        )

    def describe(
        self, token: str, thread_id: str, target: Dict[str, Any], full: bool = False
    ) -> Dict[str, Any]:
        url = f"{GALAXY_RUNS_API_BASE_URL}/v1/galaxy/describe"
        if full:
            url += "?full=true"
        return self._make_api_request(
            "POST", url, token, data={"thread_id": thread_id, "target": target}
        )

    def thread_files(self, token: str, thread_id: str) -> Dict[str, Any]:
        
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
