import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

FILE_SERVICE_BASE_URL = os.getenv("FILE_SERVICE_BASE_URL")


class FileServiceClient:

    def _request(self, method: str, path: str, token: str, **kwargs) -> Dict[str, Any]:
        if not FILE_SERVICE_BASE_URL:
            return {"error": "the file service isn't configured"}
        url = f"{FILE_SERVICE_BASE_URL}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.request(method, url, headers=headers, timeout=15, **kwargs)
            if response.status_code == 404:
                return {"error": "not_found", "status": 404}
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.Timeout:
            logger.error(f"File service request to {url} timed out")
            return {"error": "the file service timed out"}
        except requests.exceptions.HTTPError as e:
            detail, code = None, None
            try:
                body = e.response.json() if e.response is not None else None
                if isinstance(body, dict):
                    # Routes use either {text}/{detail} or {code, message, retryable}.
                    detail = body.get("text") or body.get("detail") or body.get("message")
                    code = body.get("code")
            except ValueError:
                pass
            status = e.response.status_code if e.response is not None else "?"
            result = {"error": detail or f"HTTP {status}", "status": status}
            if code:
                result["code"] = code
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"File service request failed: {e}")
            return {"error": "could not reach the file service"}

    def get_file(self, token: str, file_id: str) -> Optional[Dict[str, Any]]:
        result = self._request("GET", f"/v1/files/{file_id}", token)
        if result.get("error") == "not_found":
            return None
        if result.get("error"):
            logger.warning(f"get_file({file_id}) failed: {result['error']}")
            return None
        return result

    def get_summary(self, token: str, file_id: str) -> Optional[Dict[str, Any]]:
        result = self._request("GET", f"/v1/files/{file_id}/summary", token)
        if result.get("error"):
            return None
        return result

    def get_download_url(self, token: str, file_id: str, for_: Optional[str] = None) -> Optional[Dict[str, Any]]:
        payload = {"for": for_} if for_ else {}
        result = self._request("POST", f"/v1/files/{file_id}/download-url", token, json=payload)
        if result.get("error"):
            logger.warning(f"get_download_url({file_id}) failed: {result['error']}")
            return None
        return result

    def get_content(self, token: str, file_id: str, max_bytes: int = 60000) -> Optional[str]:
        if not FILE_SERVICE_BASE_URL:
            return None
        url = f"{FILE_SERVICE_BASE_URL}/v1/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {token}", "Range": f"bytes=0-{max_bytes - 1}"}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code in (404, 400):
                return None
            response.raise_for_status()
            return response.content.decode("utf-8", errors="replace")
        except requests.exceptions.RequestException as e:
            logger.error(f"get_content({file_id}) failed: {e}")
            return None

    def list_files(self, token: str, search: Optional[str] = None, cursor: Optional[str] = None,
                    limit: Optional[int] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        params = {k: v for k, v in {
            "search": search, "cursor": cursor, "limit": limit, "thread_id": thread_id,
        }.items() if v is not None}
        return self._request("GET", "/v1/files", token, params=params)

    def batch_get(self, token: str, file_ids: List[str]) -> Dict[str, Any]:
        return self._request("POST", "/v1/files/batch-get", token, json={"ids": file_ids})

    def deliver_to_galaxy(self, token: str, file_id: str, thread_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/v1/files/{file_id}/deliveries", token, json={"thread_id": thread_id})

    def import_url(self, token: str, url: str, name: Optional[str] = None,
                    thread_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": url}
        if name:
            payload["name"] = name
        if thread_id:
            payload["thread_id"] = thread_id
        return self._request("POST", "/v1/imports", token, json=payload)
