import logging
import os
import time

import requests
from typing import Any, Dict, List, Optional, Tuple

from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)

HYPOTHESIS_MAIN_ENDPOINT = os.getenv("HYPOTHESIS_MAIN_ENDPOINT")
HYPOTHESIS_DATA_API = os.getenv("HYPOTHESIS_DATA_API")


class HypothesisAPIMixin:
    """HTTP layer for the Hypothesis service — pipeline steps and ID lookups."""

    def _make_api_request(
        self,
        method: str,
        url: str,
        token: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            logger.debug("Making %s request to %s data=%s params=%s", method, url, data, params)
            if data and method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers)
            elif method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s", e)
            return {"error": "Request failed Please Try Again"}

    def _validate_response(self, response: Dict[str, Any], required_keys: List[str] = []) -> Tuple[bool, str]:
        if "error" in response:
            return False, response["error"]
        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        return True, ""

    def _step_1_enrich(self, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Step 1: Starting enrichment with params: %s", params)
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("POST", url, token, data=params)
        valid, error = self._validate_response(response, required_keys=["hypothesis_id"])
        if not valid:
            return {"error": f"Enrichment start failed: {error}"}
        return response

    def _step_2_poll(self, token: str, hypothesis_id: str) -> Dict[str, Any]:
        logger.info("Step 2: Polling status for hypothesis ID: %s", hypothesis_id)
        max_retries = 6
        retry_delay = 10
        url = HYPOTHESIS_MAIN_ENDPOINT
        for attempt in range(max_retries):
            response = self._make_api_request("GET", url, token, params={"id": hypothesis_id})
            valid, error = self._validate_response(response, required_keys=["status"])
            if not valid:
                return {"error": f"Status check failed: {error}"}
            status = response.get("status", "").lower()
            enrich_id = response.get("enrich_id")
            logger.info("Polling attempt %d/%d: status=%s enrich_id=%s", attempt + 1, max_retries, status, enrich_id)
            if enrich_id:
                return response
            if status in ("pending", "running"):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {"error": "Enrichment timed out after maximum retries"}
            elif status in ("failed", "error"):
                return {"error": f"Enrichment failed with status: {status}"}
            else:
                return {"error": f"Unknown status: {status}"}
        return {"error": "Enrichment timed out"}

    def _step_3_get_results(self, token: str, enrich_id: str) -> Dict[str, Any]:
        logger.info("Step 3: Fetching results for enrich ID: %s", enrich_id)
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("GET", url, token, params={"id": enrich_id})
        valid, error = self._validate_response(response, required_keys=["GO_terms", "causal_gene"])
        if not valid:
            return {"error": f"Result fetch failed: {error}"}
        return response

    def _step_4_generate(self, token: str, enrich_id: str, go_term_id: str) -> Dict[str, Any]:
        logger.info("Step 4: Generating hypothesis enrich_id=%s go=%s", enrich_id, go_term_id)
        url = HYPOTHESIS_MAIN_ENDPOINT
        response = self._make_api_request("POST", url, token, params={"id": enrich_id, "go": go_term_id})
        valid, error = self._validate_response(response, required_keys=["summary", "graph"])
        if not valid:
            return {"error": f"Final generation failed: {error}"}
        return response

    def get_by_hypothesis_id(self, token: str, hypothesis_id: str, user_id) -> Dict[str, Any]:
        logger.info("Retrieving hypothesis by ID: %s", hypothesis_id)
        emit_to_user(user=user_id, message=f"Retrieving hypothesis by ID: {hypothesis_id}")
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(HYPOTHESIS_MAIN_ENDPOINT, params={"id": hypothesis_id}, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info("Hypothesis GET response status: %s", data.get("status"))
            summary = data.get("summary", "")
            graph = data.get("graph", {})
            if not summary:
                logger.warning("Hypothesis %s returned no summary", hypothesis_id)
                return {"text": "NO summaries provided"}
            return {
                "text": f"Summary: {summary}",
                "resource": {"id": hypothesis_id, "type": "hypothesis", "graph": graph},
            }
        except Exception as e:
            logger.error("Failed to retrieve hypothesis by ID %s: %s", hypothesis_id, e)
            emit_to_user(user=user_id, message="Error retrieving hypothesis")
            return {"text": "NO summaries provided"}
