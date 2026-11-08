"""Archived -- the old "run it for real" generate flow, superseded by
HypothesisGeneration.propose_hypothesis in hypothesis.py. Not imported or
called anywhere; kept only for reference."""

from typing import Dict, Any, Tuple, Optional, List
import logging
import os
import time
import requests

from app.prompts.hypothesis_prompt import hypothesis_format_prompt
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)

HYPOTHESIS_MAIN_ENDPOINT = os.getenv('HYPOTHESIS_MAIN_ENDPOINT')
HYPOTHESIS_DATA_API = os.getenv('HYPOTHESIS_DATA_API')


class HypothesisGenerationL:

    def __init__(self, llm) -> None:
        self.llm = llm

    def _make_api_request(self, method: str, url: str, token: str,
                           params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
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
        except requests.exceptions.HTTPError as e:
            detail = None
            try:
                body = e.response.json() if e.response is not None else None
                if isinstance(body, dict):
                    detail = body.get("detail") or body.get("error")
            except ValueError:
                pass
            status = e.response.status_code if e.response is not None else "?"
            return {"error": detail or f"HTTP {status}"}
        except requests.exceptions.RequestException:
            return {"error": "Request failed -- could not reach the hypothesis service"}

    def _validate_response(self, response: Dict[str, Any], required_keys: List[str] = []) -> Tuple[bool, str]:
        if "error" in response:
            return False, response["error"]
        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        return True, ""

    def _step_1_enrich(self, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("POST", url, token, data=params)
        valid, error = self._validate_response(response, required_keys=["hypothesis_id"])
        if not valid:
            return {"error": f"Enrichment start failed: {error}"}
        return response

    def _step_2_poll(self, token: str, hypothesis_id: str) -> Dict[str, Any]:
        max_retries = 6
        retry_delay = 10
        url = f"{HYPOTHESIS_MAIN_ENDPOINT}/hypothesis"
        for attempt in range(max_retries):
            response = self._make_api_request("GET", url, token, params={"id": hypothesis_id})
            valid, error = self._validate_response(response, required_keys=["status"])
            if not valid:
                return {"error": f"Status check failed: {error}"}
            status = response.get("status")
            if status == "completed":
                if "enrich_id" not in response:
                    return {"error": "Completed status but missing enrich_id"}
                return response
            elif status == "pending":
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {"error": "Enrichment timed out after maximum retries"}
            else:
                return {"error": f"Unknown status: {status}"}
        return {"error": "Enrichment timed out"}

    def _step_3_get_results(self, token: str, enrich_id: str) -> Dict[str, Any]:
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("GET", url, token, params={"id": enrich_id})
        valid, error = self._validate_response(response, required_keys=["GO_terms", "causal_gene"])
        if not valid:
            return {"error": f"Result fetch failed: {error}"}
        return response

    def _step_4_generate(self, token: str, enrich_id: str, go_term_id: str) -> Dict[str, Any]:
        url = f"{HYPOTHESIS_MAIN_ENDPOINT}/hypothesis"
        response = self._make_api_request("POST", url, token, data={"id": enrich_id, "go": go_term_id})
        valid, error = self._validate_response(response, required_keys=["summary", "graph"])
        if not valid:
            return {"error": f"Final generation failed: {error}"}
        return response

    def format_user_query(self, query: str, user_id, thread_id=None) -> Dict[str, Any]:
        import re
        try:
            prompt = hypothesis_format_prompt.format(question=query)
            response = self.llm.generate(prompt)

            if not response:
                emit_to_user(user=user_id, message="Warning: Empty response from query formatting", thread_id=thread_id)
                return {}
            if isinstance(response, str):
                emit_to_user(user=user_id, message="Warning: Could not parse extraction response", thread_id=thread_id)
                return {}

            regex_variants = re.findall(r'\brs\d+\b', query, re.IGNORECASE)
            if regex_variants:
                user_variant = regex_variants[0]
                llm_variant = response.get("variant")
                if llm_variant and llm_variant.lower() != user_variant.lower():
                    response["variant"] = user_variant
                    emit_to_user(user=user_id, message=f"Corrected variant extraction to {user_variant}", thread_id=thread_id)
                elif not llm_variant:
                    response["variant"] = user_variant
            return response
        except Exception:
            emit_to_user(user=user_id, message="Error formatting query", thread_id=thread_id)
            return {}

    def get_user_projects(self, token: str) -> List[Dict[str, Any]]:
        url = f"{HYPOTHESIS_DATA_API}/projects"
        try:
            response = self._make_api_request("GET", url, token)
            valid, error = self._validate_response(response, required_keys=["projects"])
            if not valid:
                return []
            return response["projects"]
        except Exception:
            return []

    @staticmethod
    def _normalize_field(s: str) -> str:
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        return normalized.replace("_tissue", "").replace("_cell", "").replace("_cells", "")

    def _match_project_fields(self, token: str, project: Dict[str, Any], variant: str, tissue: str):
        project_id = project.get("id")
        url = f"{HYPOTHESIS_DATA_API}/projects"
        details = self._make_api_request("GET", url, token, params={"id": project_id})
        if "error" in details:
            return False, False, [], []

        project_variants = [h.get("variant") for h in details.get("hypotheses", [])]
        project_tissues = [t.get("name") for t in details.get("ldsc", {}).get("tissues", [])]

        norm_variant = self._normalize_field(variant)
        norm_tissue = self._normalize_field(tissue)
        has_variant = any(norm_variant == self._normalize_field(v) for v in project_variants)
        has_tissue = any(norm_tissue == self._normalize_field(t) for t in project_tissues)
        return has_variant, has_tissue, project_variants, project_tissues

    def _collect_all_variants(self, token: str, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_variants = []
        for project in projects:
            project_id = project.get("id")
            project_name = project.get("name")
            url = f"{HYPOTHESIS_DATA_API}/projects"
            details = self._make_api_request("GET", url, token, params={"id": project_id})
            if "error" not in details:
                for h in details.get("hypotheses", []):
                    all_variants.append({"variant": h.get("variant"), "project_name": project_name})
        return all_variants

    def validate_project_context(self, token: str, variant: str, tissue: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        projects = self.get_user_projects(token)
        if not projects:
            return None, {"error_type": "no_projects", "variant": variant, "tissue": tissue}

        variant_found_in = []
        tissue_found_in = []
        searched_project_names = []

        for project in projects:
            project_id = project.get("id")
            project_name = project.get("name")
            searched_project_names.append(project_name)

            has_variant, has_tissue, project_variants, project_tissues = self._match_project_fields(
                token, project, variant, tissue
            )
            if not project_variants and not project_tissues:
                continue

            norm_variant = self._normalize_field(variant)
            norm_tissue = self._normalize_field(tissue)

            if has_variant:
                actual_v = next((v for v in project_variants if self._normalize_field(v) == norm_variant), variant)
                variant_found_in.append({
                    "project_id": project_id, "project_name": project_name,
                    "actual_variant": actual_v, "tissues": project_tissues,
                })
            if has_tissue:
                actual_t = next((t for t in project_tissues if self._normalize_field(t) == norm_tissue), tissue)
                tissue_found_in.append({
                    "project_id": project_id, "project_name": project_name,
                    "actual_tissue": actual_t, "variants": project_variants,
                })
            if has_variant and has_tissue:
                return project_id, None

        if len(variant_found_in) == 0:
            all_variants = self._collect_all_variants(token, projects)
            return None, {
                "error_type": "variant_not_found", "variant": variant, "tissue": tissue,
                "all_variants": all_variants, "searched_projects": searched_project_names,
            }

        return None, {
            "error_type": "mismatch", "variant": variant, "tissue": tissue,
            "variant_projects": variant_found_in, "tissue_projects": tissue_found_in,
        }

    def _format_validation_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        error_type = error_details.get("error_type")
        variant = error_details.get("variant")
        tissue = error_details.get("tissue")

        if error_type == "no_projects":
            return {"text": "Since there's no project created yet, you can use the platform UI to generate a new hypothesis. In the meantime, I can still assist you based on the hypotheses you've already generated."}

        if error_type == "variant_not_found":
            searched_projects = ", ".join(error_details.get("searched_projects", []))
            all_variants = error_details.get("all_variants", [])
            variant_list = "\n".join([f"- {v['variant']} ({v['project_name']})" for v in all_variants])
            return {"text": (
                f"No hypothesis is generated: I searched your projects ({searched_projects if searched_projects else 'none'}), "
                f"but variant **{variant}** was not found.\n\n"
                f"**Available variants:**\n{variant_list if variant_list else '(none)'}"
            )}

        if error_type == "mismatch":
            variant_projects = error_details.get("variant_projects", [])
            tissue_projects = error_details.get("tissue_projects", [])
            variant_project_name = variant_projects[0]["project_name"] if variant_projects else "Unknown"
            tissue_project_name = tissue_projects[0]["project_name"] if tissue_projects else "Unknown"
            available_tissues = variant_projects[0]["tissues"] if variant_projects else []
            tissue_list = "\n".join([f"- {t}" for t in available_tissues])
            return {"text": (
                f"No hypothesis is generated: **{variant}** is in {variant_project_name}, but **{tissue}** is in {tissue_project_name}.\n"
                f"They must be in the same project.\n\n"
                f"For **{variant}**, use these tissues:\n{tissue_list if tissue_list else '(none)'}"
            )}

        return {"text": f"No hypothesis is generated: I couldn't find a project containing both **{variant}** and **{tissue}**."}

    def _start_enrichment(self, token: str, params: Dict[str, Any], user_id: str, thread_id=None) -> Dict[str, Any]:
        emit_to_user(user=user_id, message=f"Starting enrichment for {params.get('variant')}...", thread_id=thread_id)
        step1_res = self._step_1_enrich(token, params)
        if "error" in step1_res:
            return {"text": f"I tried to start the enrichment, but failed: {step1_res['error']}"}

        hypothesis_id = step1_res["hypothesis_id"]

        emit_to_user(user=user_id, message="Waiting for analysis to complete...", thread_id=thread_id)
        step2_res = self._step_2_poll(token, hypothesis_id)
        if "error" in step2_res:
            return {"text": f"Enrichment started (ID: {hypothesis_id}), but failed during processing: {step2_res['error']}"}

        enrich_id = step2_res["enrich_id"]

        emit_to_user(user=user_id, message="Fetching enrichment results...", thread_id=thread_id)
        step3_res = self._step_3_get_results(token, enrich_id)
        if "error" in step3_res:
            return {"text": f"Analysis completed, but failed to retrieve results: {step3_res['error']}"}

        go_terms = step3_res.get("GO_terms", [])
        if not go_terms:
            return {"text": "Analysis completed, but no significant GO terms were found."}

        return {"hypothesis_id": hypothesis_id, "enrich_id": enrich_id, "go_terms": go_terms}

    def _finish_enrichment(self, token: str, hypothesis_id: str, enrich_id: str,
                            go_id: str, user_id: str, thread_id=None) -> Dict[str, Any]:
        emit_to_user(user=user_id, message="Generating final hypothesis...", thread_id=thread_id)
        step4_res = self._step_4_generate(token, enrich_id, go_id)
        if "error" in step4_res:
            return {"text": f"Failed to generate final hypothesis summary: {step4_res['error']}"}
        return {
            "text": step4_res["summary"],
            "resource": {"id": hypothesis_id, "type": "hypothesis", "graph": step4_res["graph"]},
        }

    def generate_hypothesis(self, token: str, user_query: str, user_id: str, thread_id=None) -> Dict[str, Any]:
        emit_to_user(user=user_id, message="Analyzing your query...", thread_id=thread_id)

        params = self.format_user_query(user_query, user_id, thread_id=thread_id)
        if not params:
            return {"text": "Could not understand the biological parameters in your query."}

        validated_project_id, error_details = self.validate_project_context(token, params["variant"], params["tissue_name"])

        if validated_project_id:
            params["project_id"] = validated_project_id
            emit_to_user(user=user_id, message=f"Found related project: {validated_project_id}", thread_id=thread_id)
        else:
            return self._format_validation_error(error_details)

        return self._start_enrichment(token, params, user_id, thread_id=thread_id)
