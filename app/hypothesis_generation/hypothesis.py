from typing import Dict, Any, Tuple, Optional, List
from app.socket_manager import emit_to_user
from app.agents.proposal_requirements import FILE_FORMAT_REQUIREMENTS
import logging
import os
import re
import difflib
import requests

logger = logging.getLogger(__name__)

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

    def __init__(self, llm) -> None:
        self.llm = llm
        logger.info("HypothesisGeneration initialized with LLM")

    def _make_api_request(self,
                         method: str,
                         url: str,
                         token: str,
                         params: Optional[Dict] = None,
                         data: Optional[Dict] = None) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token}"
        }
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
        if "error" in response:
             return False, response["error"]

        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"

        return True, ""

    def get_by_hypothesis_id(self, token: str, hypothesis_id: str, user_id, query=None, thread_id=None) -> Dict[str, Any]:
        logger.info(f"Retrieving hypothesis by ID: {hypothesis_id}")
        emit_to_user(user=user_id, message=f"Retrieving hypothesis by ID: {hypothesis_id}", thread_id=thread_id)

        try:
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{HYPOTHESIS_SERVICE_BASE_URL}/hypothesis"
            response = requests.get(url, params={"id": hypothesis_id}, headers=headers)
            response.raise_for_status()
            data = response.json()

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

    _GWAS_SEARCH_GENERIC_SUFFIXES = {"disease", "disorder", "syndrome", "condition"}
    _SEX_FEMALE_WORDS = {"women", "woman", "female", "females"}
    _SEX_MALE_WORDS = {"men", "man", "male", "males"}
    _SEX_CONNECTORS = {"in", "for", "among"}

    @classmethod
    def _extract_requested_sex(cls, phenotype: str) -> Tuple[str, Optional[str]]:
        words = (phenotype or "").split()
        sex = None
        out: List[str] = []
        for w in words:
            wl = w.lower().strip(",.")
            if wl in cls._SEX_FEMALE_WORDS or wl in cls._SEX_MALE_WORDS:
                sex = "female" if wl in cls._SEX_FEMALE_WORDS else "male"
                if out and out[-1].lower() in cls._SEX_CONNECTORS:
                    out.pop()
                continue
            out.append(w)
        stripped = " ".join(out).strip()
        return (stripped or phenotype, sex)

    @classmethod
    def _gwas_search_term(cls, phenotype: str) -> str:
        words = (phenotype or "").strip().split()
        while words and words[-1].lower() in cls._GWAS_SEARCH_GENERIC_SUFFIXES:
            words.pop()
        if not words:
            return phenotype
        stem = " ".join(words)
        if stem.lower().endswith("'s"):
            stem = stem[:-2]
        elif stem.lower().endswith("s") and not stem.lower().endswith("ss"):
            stem = stem[:-1]
        return stem

    def _search_gwas_files_raw(self, token: str, phenotype: str, sex: Optional[str] = None) -> Dict[str, Any]:
        url = f"{HYPOTHESIS_SERVICE_BASE_URL}/gwas-files"
        params = {"search": phenotype, "skip": 0, "limit": 10}
        if sex:
            params["sex"] = sex
        response = self._make_api_request("GET", url, token, params=params)
        if isinstance(response, dict) and "error" not in response:
            files = response.get("gwas_files", []) or []
            return {"files": files, "total": response.get("total_files", len(files))}
        return {"files": [], "total": 0}

    def _list_projects_raw(self, token: str) -> List[Dict[str, Any]]:
        url = f"{HYPOTHESIS_SERVICE_BASE_URL}/projects"
        response = self._make_api_request("GET", url, token)
        if isinstance(response, list):
            return response
        return response.get("projects", []) if isinstance(response, dict) else []

    def _get_project_detail_raw(self, token: str, project_id: str) -> Dict[str, Any]:
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

    def list_credible_set_variants(self, token: str, project_id: str) -> Dict[str, Any]:
        detail = self._get_project_detail_raw(token, project_id)
        if not detail:
            return {"error": f"Couldn't read project {project_id}."}

        variants = []
        for cs in detail.get("credible_sets") or []:
            lead = cs.get("lead_variant")
            if isinstance(lead, dict):
                rsid = lead.get("rs_id")
                if not rsid:
                    continue
                variants.append({
                    "credible_set_id": cs.get("_id"),
                    "variant": rsid,
                    "chromosome": lead.get("chromosome"),
                    "position": lead.get("position"),
                    "log_pvalue": lead.get("log_pvalue"),
                    "posterior_prob": lead.get("posterior_prob"),
                    "beta": lead.get("beta"),
                })
            elif lead:
                variants.append({"credible_set_id": cs.get("_id"), "variant": lead})
        variants.sort(key=lambda v: v.get("log_pvalue") if v.get("log_pvalue") is not None else -1, reverse=True)
        return {"project_id": project_id, "variants": variants}

    _ICD10_PREFIX_RE = re.compile(r"^diagnoses\s*-\s*main\s*ICD1?0?\s*:\s*\S+\s*", re.IGNORECASE)

    @classmethod
    def _trim_phenotype_for_name(cls, phenotype: str) -> str:
        trimmed = cls._ICD10_PREFIX_RE.sub("", phenotype).strip()
        return trimmed.split("/")[0].strip() or phenotype

    @classmethod
    def _suggest_project_name(cls, gwas_file: Dict[str, Any]) -> Optional[str]:
        phenotype = gwas_file.get("phenotype")
        if not phenotype:
            return None
        name = cls._trim_phenotype_for_name(phenotype)
        sex = (gwas_file.get("sex") or "").lower()
        if sex == "female":
            return f"{name} in women"
        if sex == "male":
            return f"{name} in men"
        return name

    @staticmethod
    def _new_project_requirements_note() -> str:
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

        searched_phenotype = requested_phenotype or user_query
        project = self._match_project_by_phenotype(searched_phenotype, projects)
        if not project:
            names = ", ".join(p.get("phenotype", "?") for p in projects)
            base_phenotype, requested_sex = self._extract_requested_sex(searched_phenotype)
            search_result = self._search_gwas_files_raw(
                token, self._gwas_search_term(base_phenotype), sex=requested_sex
            )
            gwas_files, total_files = search_result["files"], search_result["total"]
            if gwas_files:
                recommended = gwas_files[0]
                detail = (
                    f" ({recommended.get('source', '?')}, sex={recommended.get('sex', '?')}, "
                    f"build={recommended.get('genome_build', '?')})"
                )
                return {
                    "text": (f"'{searched_phenotype}' isn't one of your existing projects, but you "
                             f"can create a new hypothesis from the public catalog instead -- there "
                             f"{'is' if total_files == 1 else 'are'} {total_files} GWAS file(s) "
                             f"available, here's one: **{recommended.get('phenotype') or recommended.get('filename', '?')}**"
                             f"{detail}."
                             ) + attached_suffix,
                    "kind": "hypothesis-project",
                    "json_format": {
                        "gwas_file": recommended.get("filename") or recommended.get("id"),
                        "project_name": self._suggest_project_name(recommended),
                        "phenotype": recommended.get("phenotype"),
                    },
                    "recommended_file": {
                        "filename": recommended.get("filename") or recommended.get("id"),
                        "phenotype": recommended.get("phenotype"),
                        "sex": recommended.get("sex"),
                        "source": recommended.get("source"),
                        "genome_build": recommended.get("genome_build"),
                    },
                }
            return {
                "text": (f"'{searched_phenotype}' doesn't match any of the available GWAS "
                         f"projects ({names}). I can't create a new one myself, but you can "
                         f"generate a new hypothesis from your own GWAS data: "
                         f"{self._new_project_requirements_note()} "
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
                f"heritability lands on {tissue_label}. However, {rsid} "
                f"doesn't match any of this project's own analyzed credible "
                f"sets, so I can't confirm it as a lead variant here — name "
                "a different variant, pick one from the proposal card, or "
                "tell me to choose one for you from this project's real data."
            )
        else:
            text = (
                f"{phenotype} matches the sample GWAS project, and its "
                f"heritability lands on {tissue_label}. Name a lead variant "
                "to match it to one of the project's credible sets, pick one "
                "from the proposal card, or just tell me to choose one for "
                "you from this project's own real data."
            )

        return {"text": text + tissue_mismatch_note + attached_suffix, "json_format": data}
