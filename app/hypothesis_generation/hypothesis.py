from typing import Dict, Any, Tuple, Optional, List, Union
from app.prompts.hypothesis_prompt import hypothesis_format_prompt, hypothesis_response
from app.socket_manager import emit_to_user
from app.agents.proposal_requirements import FILE_FORMAT_REQUIREMENTS
import logging
import os
import re
import difflib
import requests
import time

logger = logging.getLogger(__name__)

# Load API endpoints from environment variables
HYPOTHESIS_MAIN_ENDPOINT = os.getenv('HYPOTHESIS_MAIN_ENDPOINT')
HYPOTHESIS_DATA_API = os.getenv('HYPOTHESIS_DATA_API')
HYPOTHESIS_SERVICE_BASE_URL = os.getenv('HYPOTHESIS_SERVICE_BASE_URL')

_RSID_RE = re.compile(r"\brs\d+\b", re.IGNORECASE)

_TISSUE_LABELS = {
    "fetal_inhibitory_neuron_1": "fetal inhibitory neurons",
    "adipose_subcutaneous": "subcutaneous adipose tissue",
    "t_lymphocyte_1_cd8": "CD8 T lymphocytes",
}


def _extract_rsid(query: str) -> Optional[str]:
    m = _RSID_RE.search(query or "")
    return m.group(0) if m else None


def _humanize_tissue(name: Optional[str]) -> str:
    if not name:
        return "an unspecified tissue"
    key = name.strip().lower().strip("_")
    if key in _TISSUE_LABELS:
        return _TISSUE_LABELS[key]
    return name.replace("_", " ").strip().lower()


class HypothesisGeneration:
    """
    Handles generation and processing of hypotheses based on user queries.
    Interacts with the Hypothesis API to generate hypotheses and retrieve related information.
    """

    def __init__(self, llm) -> None:
        """
        Initialize the HypothesisGeneration class with an LLM instance.
        
        Args:
            llm: Language model instance for generating formatted queries and responses
        """
        self.llm = llm
        logger.info("HypothesisGeneration initialized with LLM")

    def _make_api_request(self,
                         method: str,
                         url: str,
                         token: str,
                         params: Optional[Dict] = None,
                         data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Helper method to make API requests with proper error handling.
        """
        headers = {
            "Authorization": f"Bearer {token}"
        }
        try:
            logger.debug(f"Making {method} request to {url} with data {data} and params {params}")
            if data and method.upper() == "POST":
                # Use json=data to send application/json
                response = requests.post(url, json=data, headers=headers)    
            elif method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
                
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.HTTPError as e:
            detail = None
            try:
                body = e.response.json() if e.response is not None else None
                if isinstance(body, dict):
                    detail = body.get("detail") or body.get("error")
            except ValueError:
                pass
            status = e.response.status_code if e.response is not None else "?"
            message = detail or f"HTTP {status}"
            logger.error(f"API request failed: {message}")
            return {"error": message}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": "Request failed -- could not reach the hypothesis service"}

    def _validate_response(self, response: Dict[str, Any], required_keys: List[str] = []) -> Tuple[bool, str]:
        """
        Validate API response status and content.
        """
        if "error" in response:
             return False, response["error"]
        
        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        
        return True, ""

    '''DISABLED -- the old "run it for real" generate flow (_step_1_enrich
    through _step_4_generate below, and format_user_query through
    generate_hypothesis further down). Kept here, not deleted, only
    because a separate, older (pre-/v1) /query endpoint still calls
    agent() without propose_only=True and would otherwise reach this
    code -- see app/agents/subgraphs/hypothesis.py's generate_node/
    confirm_node, whose own old branches are disabled the same way, for
    where this was wired in. It has real, unfixed bugs (POST /hypothesis
    sending a JSON body instead of query params; HYPOTHESIS_MAIN_ENDPOINT/
    HYPOTHESIS_DATA_API in .env pointing at an unreachable address) and
    must never run automatically regardless -- this whole path performs
    the actual mutating POST /enrich and POST /hypothesis calls, which is
    exactly what "propose, never run" forbids. propose_hypothesis()
    further down is the real, current, safe implementation.

    def _step_1_enrich(self, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Start Enrichment"""
        logger.info(f"Step 1: Starting enrichment with params: {params}")
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("POST", url, token, data=params)
        
        valid, error = self._validate_response(response, required_keys=["hypothesis_id"])
        if not valid:
            return {"error": f"Enrichment start failed: {error}"}
            
        return response

    def _step_2_poll(self, token: str, hypothesis_id: str) -> Dict[str, Any]:
        """Step 2: Polling Loop with Retry Mechanism"""
        logger.info(f"Step 2: Polling status for hypothesis ID: {hypothesis_id}")
        
        # Retry configuration: 6 attempts * 10 seconds = 60 seconds max wait
        max_retries = 6
        retry_delay = 10

        url = f"{HYPOTHESIS_MAIN_ENDPOINT}/hypothesis"
        
        for attempt in range(max_retries):
        
         
            response = self._make_api_request("GET", url, token, params={"id": hypothesis_id})
            
            valid, error = self._validate_response(response, required_keys=["status"])
            if not valid:
                 return {"error": f"Status check failed: {error}"}
            
            status = response.get("status")
            logger.info(f"Polling attempt {attempt + 1}/{max_retries}: Status is '{status}'")

            if status == "completed":
                if "enrich_id" not in response:
                     return {"error": "Completed status but missing enrich_id"}
                return response
            elif status == "pending":
                if attempt < max_retries - 1:
                    logger.info(f"Status is pending. Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    return {"error": "Enrichment timed out after maximum retries"}
            else:
                return {"error": f"Unknown status: {status}"}
                
        return {"error": "Enrichment timed out"}

    def _step_3_get_results(self, token: str, enrich_id: str) -> Dict[str, Any]:
        """Step 3: Get Enrichment Results"""
        logger.info(f"Step 3: Fetching results for enrich ID: {enrich_id}")
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("GET", url, token, params={"id": enrich_id})
        
        valid, error = self._validate_response(response, required_keys=["GO_terms", "causal_gene"])
        if not valid:
            return {"error": f"Result fetch failed: {error}"}
            
        return response

    def _step_4_generate(self, token: str, enrich_id: str, go_term_id: str) -> Dict[str, Any]:
        """Step 4: Generate Final Hypothesis"""
        logger.info(f"Step 4: Generating hypothesis for enrich ID: {enrich_id} and GO: {go_term_id}")
        url = f"{HYPOTHESIS_MAIN_ENDPOINT}/hypothesis"
        response = self._make_api_request("POST", url, token, data={"id": enrich_id, "go": go_term_id})
        
        valid, error = self._validate_response(response, required_keys=["summary", "graph"])
        if not valid:
            return {"error": f"Final generation failed: {error}"}
            
        return response
    '''


    def get_by_hypothesis_id(self, token: str, hypothesis_id: str, user_id, query=None, thread_id=None) -> Dict[str, Any]:
        """
        Retrieve hypothesis information by ID.
        """
        logger.info(f"Retrieving hypothesis by ID: {hypothesis_id}")
        logger.info(f"Token: {token}")
        emit_to_user(user=user_id, message=f"Retrieving hypothesis by ID: {hypothesis_id}", thread_id=thread_id)

        try:
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{HYPOTHESIS_SERVICE_BASE_URL}/hypothesis"
            response = requests.get(url, params={"id": hypothesis_id}, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Hypothesis GET response status: {data.get('status')}")

            summary = data.get("summary", "")
            graph = data.get("graph", {})
            result = data.get("result", {})

            if not summary and not graph:
                logger.warning(f"Hypothesis {hypothesis_id} returned no data")
                return {"text": "NO summaries provided"}

            context_parts = []

            if summary:
                context_parts.append(f"Summary:\n{summary}")

            meta = (
                f"Variant: {data.get('variant', 'N/A')} | "
                f"Phenotype: {data.get('phenotype', 'N/A')} | "
                f"Status: {data.get('status', 'N/A')}"
            )
            context_parts.append(f"Hypothesis Metadata:\n{meta}")

            nodes = graph.get("nodes", [])
            if nodes:
                node_lines = [f"  - {n.get('name', n.get('id'))} (type: {n.get('type')})" for n in nodes]
                context_parts.append("Graph Nodes:\n" + "\n".join(node_lines))

            edges = graph.get("edges", [])
            if edges:
                edge_lines = [f"  - {e.get('source')} --[{e.get('label')}]--> {e.get('target')}" for e in edges]
                context_parts.append("Graph Relationships:\n" + "\n".join(edge_lines))

            go_terms = result.get("GO_terms", [])
            if go_terms:
                go_lines = [
                    f"  - {g.get('name')} (p={g.get('p', 'N/A'):.4f}, genes: {', '.join(g.get('genes', []))})"
                    for g in go_terms[:5]
                ]
                context_parts.append("Top GO Terms:\n" + "\n".join(go_lines))

            full_context = "\n\n".join(context_parts)
            logger.info(f"Built hypothesis context with {len(nodes)} nodes, {len(edges)} edges, {len(go_terms)} GO terms")

            return {
                "text": full_context,
                "resource": {
                    "id": hypothesis_id,
                    "type": "hypothesis",
                    "graph": graph,
                }
            }
        except Exception as e:
            logger.error(f"Failed to retrieve hypothesis by ID {hypothesis_id}: {e}")
            emit_to_user(user=user_id, message="Error retrieving hypothesis", thread_id=thread_id)
            return {"text": "NO summaries provided"}

    '''DISABLED -- the rest of the old "run it for real" generate flow
    (format_user_query through generate_hypothesis below). Same reason and
    same caveats as the _step_1_enrich..._step_4_generate block disabled
    above -- not deleted only because the older, pre-/v1 /query endpoint
    still calls agent() with propose_only defaulting to False and would
    otherwise reach it; must never run automatically regardless, since it
    performs the actual mutating generate calls. propose_hypothesis()
    below is the real, current, safe implementation.

    def format_user_query(self, query: str, user_id, thread_id=None) -> Dict[str, Any]:
        """
        Format user query using the LLM to extract relevant parameters.
        """
        logger.info(f"Formatting user query: {query}")

        try:
            prompt = hypothesis_format_prompt.format(question=query)
            response = self.llm.generate(prompt)

            if not response:
                logger.warning("LLM returned empty response for query formatting")
                emit_to_user(user=user_id, message="Warning: Empty response from query formatting", thread_id=thread_id)
                return {}

            # Guard: ensure the LLM returned a dict, not a raw string
            if isinstance(response, str):
                logger.warning(f"LLM returned a string instead of a dict: {response}")
                emit_to_user(user=user_id, message="Warning: Could not parse extraction response", thread_id=thread_id)
                return {}

            logger.info(f"Successfully formatted query with {len(response)} parameters")

            # POST-PROCESSING VALIDATION: Override LLM if it hallucinated the variant
            import re
            regex_variants = re.findall(r'\brs\d+\b', query, re.IGNORECASE)

            if regex_variants:
                user_variant = regex_variants[0]
                llm_variant = response.get("variant")

                if llm_variant and llm_variant.lower() != user_variant.lower():
                    logger.warning(
                        f"LLM hallucination detected! User said '{user_variant}' but LLM extracted "
                        f"'{llm_variant}'. Overriding."
                    )
                    response["variant"] = user_variant
                    emit_to_user(user=user_id, message=f"Corrected variant extraction to {user_variant}", thread_id=thread_id)
                elif not llm_variant:
                    logger.warning(f"LLM missed variant. Found '{user_variant}' via regex.")
                    response["variant"] = user_variant

            return response

        except Exception as e:
            logger.error(f"Error formatting user query: {str(e)}")
            emit_to_user(user=user_id, message="Error formatting query", thread_id=thread_id)
            return {}

    def get_user_projects(self, token: str) -> List[Dict[str, Any]]:
        """
        Fetch the list of projects available to the user.
        """
        url = f"{HYPOTHESIS_DATA_API}/projects"
        try:
            response = self._make_api_request("GET", url, token)
            valid, error = self._validate_response(response, required_keys=["projects"])
            if not valid:
                logger.error(f"Failed to fetch projects: {error}")
                return []
            return response["projects"]
        except Exception as e:
            logger.error(f"Error fetching user projects: {e}")
            return []

    @staticmethod
    def _normalize_field(s: str) -> str:
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        normalized = normalized.replace("_tissue", "").replace("_cell", "").replace("_cells", "")
        return normalized

    def _match_project_fields(self, token: str, project: Dict[str, Any], variant: str, tissue: str):
        """Fetch project details and return (has_variant, has_tissue, project_variants, project_tissues)."""
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
        """Collect all available variants across all projects."""
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
        """
        Phase 3 Validation: Check if the variant and tissue exist in the same project.

        Returns:
            Tuple[Optional[str], Optional[Dict]]:
                - First element: project_id if found, else None
                - Second element: Error details if validation failed, else None
        """
        logger.info(f"Validating project context for Variant: {variant}, Tissue: {tissue}")

        projects = self.get_user_projects(token)

        # Scenario 1: No projects exist
        if not projects:
            logger.warning("No projects found for user.")
            return None, {
                "error_type": "no_projects",
                "variant": variant,
                "tissue": tissue
            }

        # Track which projects contain each component
        variant_found_in = []  # List of {project_id, project_name, variants, tissues}
        tissue_found_in = []   # List of {project_id, project_name, variants, tissues}
        searched_project_names = [] # Names of all projects searched

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

            # Track findings
            if has_variant:
                # Find the actual variant name from the project for consistent suggesting
                actual_v = next((v for v in project_variants if self._normalize_field(v) == norm_variant), variant)
                variant_found_in.append({
                    "project_id": project_id,
                    "project_name": project_name,
                    "actual_variant": actual_v,
                    "tissues": project_tissues
                })

            if has_tissue:
                # Find the actual tissue name from the project
                actual_t = next((t for t in project_tissues if self._normalize_field(t) == norm_tissue), tissue)
                tissue_found_in.append({
                    "project_id": project_id,
                    "project_name": project_name,
                    "actual_tissue": actual_t,
                    "variants": project_variants
                })

            # Success case: Both found in same project
            if has_variant and has_tissue:
                logger.info(f"Validation Successful: Found matched context in project {project_id}")
                return project_id, None

        # If we reach here, validation failed. Determine the specific error type.
        logger.warning("Validation Failed: No single project contains both variant and tissue.")

        # Scenario 2: Variant not found anywhere
        if len(variant_found_in) == 0:
            all_variants = self._collect_all_variants(token, projects)
            return None, {
                "error_type": "variant_not_found",
                "variant": variant,
                "tissue": tissue,
                "all_variants": all_variants,
                "searched_projects": searched_project_names
            }

        # Scenario 3: Mismatch (variant and tissue exist, but in different projects)
        return None, {
            "error_type": "mismatch",
            "variant": variant,
            "tissue": tissue,
            "variant_projects": variant_found_in,
            "tissue_projects": tissue_found_in
        }

    def _format_validation_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Format a validation error dict into a user-facing text response."""
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

        # Fallback (should not reach here)
        return {"text": f"No hypothesis is generated: I couldn't find a project containing both **{variant}** and **{tissue}**."}

    def _start_enrichment(self, token: str, params: Dict[str, Any], user_id: str, thread_id=None) -> Dict[str, Any]:
        # Step 1: Start Enrichment
        emit_to_user(user=user_id, message=f"Starting enrichment for {params.get('variant')}...", thread_id=thread_id)
        step1_res = self._step_1_enrich(token, params)
        if "error" in step1_res:
            logger.error(step1_res["error"])
            return {"text": f"I tried to start the enrichment, but failed: {step1_res['error']}"}

        hypothesis_id = step1_res["hypothesis_id"]

        # Step 2: Polling
        emit_to_user(user=user_id, message="Waiting for analysis to complete...", thread_id=thread_id)
        step2_res = self._step_2_poll(token, hypothesis_id)
        if "error" in step2_res:
            logger.error(step2_res["error"])
            return {"text": f"Enrichment started (ID: {hypothesis_id}), but failed during processing: {step2_res['error']}"}

        enrich_id = step2_res["enrich_id"]

        # Step 3: Get Results
        emit_to_user(user=user_id, message="Fetching enrichment results...", thread_id=thread_id)
        step3_res = self._step_3_get_results(token, enrich_id)
        if "error" in step3_res:
            logger.error(step3_res["error"])
            return {"text": f"Analysis completed, but failed to retrieve results: {step3_res['error']}"}

        go_terms = step3_res.get("GO_terms", [])
        if not go_terms:
            return {"text": "Analysis completed, but no significant GO terms were found."}

        return {"hypothesis_id": hypothesis_id, "enrich_id": enrich_id, "go_terms": go_terms}

    def _finish_enrichment(self, token: str, hypothesis_id: str, enrich_id: str,
                           go_id: str, user_id: str, thread_id=None) -> Dict[str, Any]:
        """Step 4: generate the final hypothesis for whichever GO term was
        chosen (by the user, via confirmation)."""
        emit_to_user(user=user_id, message="Generating final hypothesis...", thread_id=thread_id)
        step4_res = self._step_4_generate(token, enrich_id, go_id)
        if "error" in step4_res:
            logger.error(step4_res["error"])
            return {"text": f"Failed to generate final hypothesis summary: {step4_res['error']}"}

        return {
            "text": step4_res["summary"],
            "resource": {
                "id": hypothesis_id,
                "type": "hypothesis",
                "graph": step4_res["graph"],
            }
        }

    def generate_hypothesis(self, token: str, user_query: str, user_id: str, thread_id=None) -> Dict[str, Any]:
        """
        Orchestrates the NLP-driven hypothesis generation workflow.
        """
        logger.info(f"Starting NLP-driven hypothesis generation for: {user_query}")
        emit_to_user(user=user_id, message="Analyzing your query...", thread_id=thread_id)

        # 1. NLP Extraction
        params = self.format_user_query(user_query, user_id, thread_id=thread_id)
        if not params:
            return {"text": "Could not understand the biological parameters in your query."}

        # Ensure we have the minimum required fields
        if "variant" not in params or "tissue_name" not in params:
             # Fallback or ask for clarification? For now, error.
             pass

        # Log the extracted parameters so they are visible in the Docker logs
        logger.info(
            f"Extracted parameters: variant='{params.get('variant')}', "
            f"tissue='{params.get('tissue_name')}'"
        )

        # Validate Project Context (Phase 3)
        validated_project_id, error_details = self.validate_project_context(token, params["variant"], params["tissue_name"])

        if validated_project_id:
             params["project_id"] = validated_project_id
             logger.info(f"Project context valid ation successful. Using project ID: {validated_project_id}")
             # Add project info to user message
             emit_to_user(user=user_id, message=f"Found related project: {validated_project_id}", thread_id=thread_id)
        else:
             # Validation failed - format specific error message based on error type
             logger.warning(f"Project validation failed for {params['variant']} in {params['tissue_name']}")
             return self._format_validation_error(error_details)

        return self._start_enrichment(token, params, user_id, thread_id=thread_id)
    '''


    def _list_projects_raw(self, token: str) -> List[Dict[str, Any]]:
        """GET /projects -- read-only."""
        url = f"{HYPOTHESIS_SERVICE_BASE_URL}/projects"
        response = self._make_api_request("GET", url, token)
        if isinstance(response, list):
            return response
        return response.get("projects", []) if isinstance(response, dict) else []

    def _get_project_detail_raw(self, token: str, project_id: str) -> Dict[str, Any]:
        """GET /projects?id=<id> -- read-only."""
        url = f"{HYPOTHESIS_SERVICE_BASE_URL}/projects"
        response = self._make_api_request("GET", url, token, params={"id": project_id})
        return response if isinstance(response, dict) and "error" not in response else {}

    @staticmethod
    def _match_project_by_phenotype(query: str, projects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        query_lower = (query or "").lower()
        query_words = set(re.findall(r"[a-z0-9]+", query_lower))
        best, best_score = None, 0.0
        for p in projects:
            phenotype = (p.get("phenotype") or "").lower()
            if not phenotype:
                continue
            if phenotype in query_lower:
                return p
            phenotype_words = set(re.findall(r"[a-z0-9]+", phenotype))
            overlap = (
                len(phenotype_words & query_words) / len(phenotype_words)
                if phenotype_words else 0.0
            )
            ratio = difflib.SequenceMatcher(None, phenotype, query_lower).ratio()
            score = max(overlap, ratio)
            if score > best_score:
                best, best_score = p, score
        return best if best_score >= 0.5 else None

    @staticmethod
    def _pick_best_tissue(project_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tissues = (project_detail.get("ldsc") or {}).get("tissues") or []
        if not tissues:
            return None
        return min(tissues, key=lambda t: t.get("p_value", float("inf")))

    @staticmethod
    def _match_named_tissue(requested: Optional[str], project_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A tissue the user explicitly asked for, matched against this
        project's own REAL ldsc.tissues list -- never a free-text guess.
        None means either nothing was requested, or what was requested
        genuinely isn't one of the tissues this project actually
        analyzed; the caller falls back to auto-picking in that case."""
        tissues = (project_detail.get("ldsc") or {}).get("tissues") or []
        if not requested or not tissues:
            return None
        requested_lower = requested.lower()
        requested_words = set(re.findall(r"[a-z0-9]+", requested_lower))
        best, best_score = None, 0.0
        for t in tissues:
            label = _humanize_tissue(t.get("name")).lower()
            if label in requested_lower or requested_lower in label:
                return t
            label_words = set(re.findall(r"[a-z0-9]+", label))
            overlap = len(label_words & requested_words) / len(label_words) if label_words else 0.0
            if overlap > best_score:
                best, best_score = t, overlap
        return best if best_score >= 0.5 else None

    @staticmethod
    def _match_credible_set_by_variant(project_detail: Dict[str, Any], rsid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not rsid:
            return None
        rsid_lower = rsid.lower()
        for cs in project_detail.get("credible_sets") or []:
            lead = cs.get("lead_variant")
            lead_rsid = lead.get("rs_id") if isinstance(lead, dict) else lead
            if lead_rsid and lead_rsid.lower() == rsid_lower:
                return cs
        return None

    @staticmethod
    def _new_project_requirements_note() -> str:
        """One line describing what creating a new GWAS project needs,
        built from the one confirmed source of truth
        (FILE_FORMAT_REQUIREMENTS) rather than reinvented here. Only the
        GWAS file's accepted formats are actually confirmed from the real
        service's source; phenotype/population/reference genome are named
        here as likely requirements (every existing project has them) but
        aren't verified yet against the real upload endpoint's exact
        parameter list."""
        hyp = FILE_FORMAT_REQUIREMENTS.get("hypothesis", {})
        extensions = hyp.get("accepted_extensions") or []
        ext_text = "/".join(e.lstrip(".") for e in extensions) if extensions else "a supported"
        return (
            f"upload a GWAS summary-statistics file ({ext_text}) along with its "
            "phenotype, population, and reference genome, through the platform UI."
        )

    def propose_hypothesis(self, token: str, user_query: str, user_id: str, thread_id=None,
                            requested_tissue: Optional[str] = None,
                            requested_phenotype: Optional[str] = None,
                            requested_variant: Optional[str] = None,
                            attached_file_note: Optional[str] = None) -> Dict[str, Any]:
        """Propose-only: match to an existing GWAS project by phenotype,
        pick its tissue, and try to match a lead variant to one of its
        credible sets. requested_phenotype/requested_variant/
        requested_tissue are the LLM-extracted structured parameters (the
        tool schema's job, not this function's) -- user_query is only a
        fallback safety net for whichever of those didn't reach this call
        filled in. Tissue is never auto-picked when a real requested_tissue
        match exists; falls back to the lowest-p-value tissue when none
        was requested or the requested one wasn't actually analyzed here.
        Returns {"text", "json_format"} -- never a resource, since nothing
        was generated. json_format is the proposal's self-contained data:
        project_id, project_name, phenotype, credible_set_id (None if
        unmatched), variant, tissue, seed. Safe calls only (GET /projects,
        GET /projects?id=)."""
        if not HYPOTHESIS_SERVICE_BASE_URL:
            logger.error("HYPOTHESIS_SERVICE_BASE_URL is not set")
            return {"text": "The hypothesis service isn't configured right now, so I can't look up GWAS projects.", "json_format": None}

        emit_to_user(user=user_id, message="Looking for a matching GWAS project...", thread_id=thread_id)

        attached_suffix = f"\n\n{attached_file_note}" if attached_file_note else ""

        projects = self._list_projects_raw(token)
        if not projects:
            return {
                "text": "There's no GWAS project available yet to build a hypothesis from." + attached_suffix,
                "json_format": None,
            }

        project = self._match_project_by_phenotype(requested_phenotype or user_query, projects)
        if not project:
            names = ", ".join(p.get("phenotype", "?") for p in projects)
            return {
                "text": (f"I couldn't match this to any of the available GWAS projects ({names}). "
                         f"I can't create a new one myself, but you can generate a new hypothesis "
                         f"from your own GWAS data: {self._new_project_requirements_note()} "
                         "Want to do that instead, or try a different phenotype?") + attached_suffix,
                "json_format": None,
            }

        project_id = project["id"]
        project_name = project.get("name", project_id)
        phenotype = project.get("phenotype", "")
        detail = self._get_project_detail_raw(token, project_id)

        tissue_mismatch_note = ""
        matched_tissue = self._match_named_tissue(requested_tissue, detail)
        if matched_tissue:
            tissue_name = matched_tissue.get("name")
        else:
            best_tissue = self._pick_best_tissue(detail)
            tissue_name = best_tissue.get("name") if best_tissue else None
            if requested_tissue:
                analyzed = (detail.get("ldsc") or {}).get("tissues") or []
                analyzed_labels = ", ".join(_humanize_tissue(t.get("name")) for t in analyzed)
                tissue_mismatch_note = (
                    f" You asked for {requested_tissue}, but only "
                    f"{analyzed_labels or 'no tissue'} {'was' if len(analyzed) <= 1 else 'were'} "
                    f"analyzed for this project — using {_humanize_tissue(tissue_name)} instead."
                )
        tissue_label = _humanize_tissue(tissue_name)

        rsid = requested_variant or _extract_rsid(user_query)
        matched_cs = self._match_credible_set_by_variant(detail, rsid)

        data = {
            "project_id": project_id,
            "project_name": project_name,
            "phenotype": phenotype,
            "credible_set_id": matched_cs.get("_id") if matched_cs else None,
            "variant": rsid,
            "tissue": tissue_name,
            "seed": 0,
        }

        if matched_cs:
            text = (
                f"That one I can fill in completely — {rsid} leads one of the "
                f"{project_name} project's credible sets, and its heritability "
                f"lands on {tissue_label}. Review it and generate."
            )
        elif rsid:
            text = (
                f"{phenotype} matches the sample GWAS project, and its "
                f"heritability lands on {tissue_label}, so I have filled both "
                f"in. I could not match {rsid} to one of its credible sets — "
                "pick the locus on the Manhattan and generate."
            )
        else:
            text = (
                f"{phenotype} matches the sample GWAS project, and its "
                f"heritability lands on {tissue_label}. Name a lead variant "
                "to match it to one of the project's credible sets, or pick "
                "the locus on the Manhattan and generate."
            )

        return {"text": text + tissue_mismatch_note + attached_suffix, "json_format": data}