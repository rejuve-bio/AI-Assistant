import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.storage.redis import redis_manager

logger = logging.getLogger(__name__)

HYPOTHESIS_DATA_API = os.getenv("HYPOTHESIS_DATA_API")


class HypothesisProjectMixin:
    """Project/variant validation and user-facing error formatting for hypothesis generation."""

    # --- Sample project helpers ---

    @staticmethod
    def _is_sample_project(project: Dict[str, Any]) -> bool:
        return "sample" in (project.get("name") or "").lower()

    def _get_sample_project_info(self, token: str, projects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for p in projects:
            if self._is_sample_project(p):
                url = f"{HYPOTHESIS_DATA_API}/projects"
                details = self._make_api_request("GET", url, token, params={"id": p.get("id")})
                if "error" not in details:
                    tissues = [
                        {"name": t.get("name", ""), "p": t.get("p") or t.get("p_value") or t.get("pvalue")}
                        for t in details.get("ldsc", {}).get("tissues", []) if t.get("name")
                    ]
                    variants = [h.get("variant") for h in details.get("hypotheses", []) if h.get("variant")]
                    return {"id": p.get("id"), "name": p.get("name"), "tissues": tissues, "variants": variants}
        return None

    # --- Project data ---

    def get_user_projects(self, token: str) -> List[Dict[str, Any]]:
        url = f"{HYPOTHESIS_DATA_API}/projects"
        try:
            response = self._make_api_request("GET", url, token)
            valid, error = self._validate_response(response, required_keys=["projects"])
            if not valid:
                logger.error("Failed to fetch projects: %s", error)
                return []
            return response["projects"]
        except Exception as e:
            logger.error("Error fetching user projects: %s", e)
            return []

    # --- Tissue / field helpers ---

    @staticmethod
    def _tissue_name(t) -> str:
        return t["name"] if isinstance(t, dict) else t

    @staticmethod
    def _tissue_display(t) -> str:
        if isinstance(t, dict):
            p = t.get("p")
            return f"{t['name']} (p={p:.4e})" if p is not None else t["name"]
        return t

    @staticmethod
    def _normalize_field(s) -> str:
        if not isinstance(s, str):
            return ""
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        normalized = normalized.replace("_tissue", "").replace("_cell", "").replace("_cells", "")
        return normalized

    def _match_project_fields(self, token: str, project: Dict[str, Any], variant: str, tissue: str):
        """Return (has_variant, has_tissue, project_variants, project_tissues, project_hypotheses)."""
        project_id = project.get("id")
        url = f"{HYPOTHESIS_DATA_API}/projects"
        details = self._make_api_request("GET", url, token, params={"id": project_id})
        if "error" in details:
            return False, False, [], [], []
        project_hypotheses = details.get("hypotheses", [])
        project_variants = [h.get("variant") for h in project_hypotheses]
        project_tissues = [
            {"name": t.get("name", ""), "p": t.get("p") or t.get("p_value") or t.get("pvalue")}
            for t in details.get("ldsc", {}).get("tissues", []) if t.get("name")
        ]
        norm_variant = self._normalize_field(variant) if variant else ""
        norm_tissue = self._normalize_field(tissue) if tissue else ""
        has_variant = bool(norm_variant) and any(norm_variant == self._normalize_field(v) for v in project_variants if v)
        has_tissue = bool(norm_tissue) and any(norm_tissue == self._normalize_field(t["name"]) for t in project_tissues)
        return has_variant, has_tissue, project_variants, project_tissues, project_hypotheses

    def _collect_all_variants(self, token: str, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_variants = []
        for project in projects:
            url = f"{HYPOTHESIS_DATA_API}/projects"
            details = self._make_api_request("GET", url, token, params={"id": project.get("id")})
            if "error" not in details:
                for h in details.get("hypotheses", []):
                    all_variants.append({"variant": h.get("variant"), "project_name": project.get("name")})
        return all_variants

    # --- Project context validation ---

    @staticmethod
    def _extract_hyp_id(h):
        return h.get("id") or h.get("hypothesis_id") or h.get("graph_id") or h.get("hyp_id")

    def _evaluate_variant_match(
        self, token, project_id, project_name, variant, tissue,
        is_sample, has_variant, has_tissue, project_variants, project_tissues,
        project_hypotheses, projects, variant_found_in
    ):
        if not has_variant:
            return None
        norm_variant = self._normalize_field(variant)
        actual_v = next((v for v in project_variants if v and self._normalize_field(v) == norm_variant), variant)
        existing_hyp = next(
            (h for h in project_hypotheses
             if self._extract_hyp_id(h) and h.get("variant")
             and self._normalize_field(h.get("variant", "")) == norm_variant),
            None,
        )
        existing_hyp_id = self._extract_hyp_id(existing_hyp) if existing_hyp else None
        existing_tissue = (existing_hyp or {}).get("tissue_selected", "")
        variant_found_in.append({
            "project_id": project_id,
            "project_name": project_name,
            "actual_variant": actual_v,
            "tissues": project_tissues,
            "is_sample": is_sample,
            "existing_hyp_id": existing_hyp_id,
        })
        if is_sample and existing_hyp_id:
            if has_tissue and self._normalize_field(tissue) != self._normalize_field(existing_tissue):
                logger.info("Different tissue requested (%s vs existing %s) — allowing new generation", tissue, existing_tissue)
                return project_id, None
            return None, {
                "error_type": "existing_hypothesis",
                "variant": actual_v,
                "hypothesis_id": existing_hyp_id,
                "project_id": project_id,
                "tissue_used": existing_tissue,
                "available_tissues": project_tissues,
            }
        if has_tissue:
            if is_sample:
                logger.info("Validation successful — sample project %s", project_id)
                return project_id, None
            sample_info = self._get_sample_project_info(token, projects)
            return None, {
                "error_type": "non_sample_project",
                "variant": variant,
                "project_name": project_name,
                "sample_info": sample_info,
            }
        return None

    def validate_project_context(self, token: str, variant: str, tissue: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate variant + tissue against user's projects.
        Returns (project_id, None) on success, or (None, error_details) on failure.
        """
        logger.info("Validating project context: variant=%s tissue=%s", variant, tissue)
        projects = self.get_user_projects(token)
        if not projects:
            return None, {"error_type": "no_projects", "variant": variant, "tissue": tissue}

        variant_found_in: List[Dict[str, Any]] = []
        searched_project_names: List[str] = []

        for project in projects:
            project_id = project.get("id")
            project_name = project.get("name")
            searched_project_names.append(project_name)
            is_sample = self._is_sample_project(project)

            has_variant, has_tissue, project_variants, project_tissues, project_hypotheses = self._match_project_fields(
                token, project, variant, tissue
            )
            if not project_variants and not project_tissues:
                continue

            result = self._evaluate_variant_match(
                token, project_id, project_name, variant, tissue,
                is_sample, has_variant, has_tissue, project_variants, project_tissues,
                project_hypotheses, projects, variant_found_in,
            )
            if result is not None:
                return result

        if not variant_found_in:
            all_variants = self._collect_all_variants(token, projects)
            return None, {
                "error_type": "variant_not_found",
                "variant": variant,
                "tissue": tissue,
                "all_variants": all_variants,
                "searched_projects": searched_project_names,
            }

        sample_match = next((v for v in variant_found_in if v["is_sample"]), None)

        if not tissue:
            if sample_match:
                return None, {
                    "error_type": "tissue_missing",
                    "variant": sample_match["actual_variant"],
                    "project_id": sample_match["project_id"],
                    "project_name": sample_match["project_name"],
                    "available_tissues": sample_match["tissues"],
                }
            sample_info = self._get_sample_project_info(token, projects)
            return None, {
                "error_type": "non_sample_project",
                "variant": variant,
                "project_name": variant_found_in[0]["project_name"],
                "sample_info": sample_info,
            }

        if sample_match:
            return None, {
                "error_type": "tissue_not_found",
                "variant": variant,
                "tissue": tissue,
                "project_id": sample_match["project_id"],
                "project_name": sample_match["project_name"],
                "available_tissues": sample_match["tissues"],
            }

        sample_info = self._get_sample_project_info(token, projects)
        return None, {
            "error_type": "non_sample_project",
            "variant": variant,
            "project_name": variant_found_in[0]["project_name"],
            "sample_info": sample_info,
        }

    # --- User-facing error formatting ---

    def _format_variant_not_found(self, variant: str, all_variants: list) -> Dict[str, Any]:
        sample_variants = [v for v in all_variants if "sample" in v.get("project_name", "").lower()]
        non_sample = [v for v in all_variants if "sample" not in v.get("project_name", "").lower()]
        if sample_variants:
            sv = sample_variants[0]
            return {"text": (
                f"I couldn't find **{variant}** in your projects, but there's a ready-to-use sample — "
                f"**{sv['project_name']}** with variant **{sv['variant']}**. "
                f"You can see the full pipeline in action: from LDSC enrichment to pathway analysis to a final biological hypothesis. "
                f"Would you like to explore it using the sample?"
            )}
        if non_sample:
            variant_list = "\n".join([f"- {v['variant']} ({v['project_name']})" for v in non_sample])
            return {"text": (
                f"I couldn't find **{variant}** in your projects. Here's what you do have:\n\n"
                f"{variant_list}\n\nWould you like to run a hypothesis on one of these instead?"
            )}
        return {"text": f"I couldn't find **{variant}** in any of your projects and no other variants are available yet."}

    def _format_tissue_missing_error(self, error_details: Dict[str, Any], variant) -> Dict[str, Any]:
        project_name = error_details.get("project_name", "your project")
        tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in error_details.get("available_tissues", [])])
        return {"text": (
            f"Found **{variant}** in your **{project_name}** project — we're one step away from generating a hypothesis! "
            f"I just need to know which tissue context to run the analysis on. "
            f"Here's what's available:\n{tissue_list if tissue_list else '(none available)'}\n\n"
            f"Pick one and I'll take it from there."
        )}

    def _format_mismatch_error(self, error_details: Dict[str, Any], variant, tissue) -> Dict[str, Any]:
        variant_projects = error_details.get("variant_projects", [])
        project_name = variant_projects[0]["project_name"] if variant_projects else "Unknown"
        available_tissues = variant_projects[0].get("tissues", []) if variant_projects else []
        tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in available_tissues])
        return {"text": (
            f"**{variant}** is in **{project_name}**, but **{tissue}** was not found in that project.\n\n"
            f"Available tissues in **{project_name}**:\n{tissue_list if tissue_list else '(none)'}"
        )}

    def _format_existing_hypothesis_error(self, error_details: Dict[str, Any], variant) -> Dict[str, Any]:
        tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in error_details.get("available_tissues", [])])
        return {"text": (
            f"There's already a generated hypothesis for **{variant}** in the sample project — pulling it up for you now.\n\n"
            f"If you'd like to run a new analysis with a different tissue, here's what's available:\n"
            f"{tissue_list if tissue_list else '(none available)'}\n\nJust pick one and I'll get started."
        )}

    def _format_non_sample_project_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        project_name = error_details.get("project_name", "your project")
        sample_info = error_details.get("sample_info")
        sample_name = sample_info["name"] if sample_info else "the sample project"
        sample_variant = sample_info["variants"][0] if sample_info and sample_info.get("variants") else None
        sample_note = (
            f"\n\nIn the meantime, I can run the full pipeline on **{sample_name}** using **{sample_variant}** "
            f"so you can see how it works — want to give it a try?"
        ) if sample_variant else f"\n\nI can walk you through it using **{sample_name}** if you'd like."
        return {"text": (
            f"Hypothesis generation for **{project_name}** needs to be kicked off from the platform — "
            f"I can only run it for the sample project from here."
            f"{sample_note}"
        )}

    def _format_tissue_not_found_error(self, error_details: Dict[str, Any], tissue) -> Dict[str, Any]:
        project_name = error_details.get("project_name", "the sample project")
        tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in error_details.get("available_tissues", [])])
        return {"text": (
            f"I couldn't find **{tissue}** in **{project_name}**. "
            f"Here are the available tissues:\n{tissue_list if tissue_list else '(none available)'}\n\n"
            f"Pick one and I'll run the analysis."
        )}

    def _format_validation_error(self, error_details: Dict[str, Any]) -> Dict[str, Any]:
        error_type = error_details.get("error_type")
        variant = error_details.get("variant")
        tissue = error_details.get("tissue")

        if error_type == "no_projects":
            return {"text": "Since there's no project created yet, you can use the platform UI to generate a new hypothesis. In the meantime, I can still assist you based on the hypotheses you've already generated."}

        if error_type == "variant_not_found":
            return self._format_variant_not_found(variant, error_details.get("all_variants", []))

        if error_type == "tissue_missing":
            return self._format_tissue_missing_error(error_details, variant)
        if error_type == "mismatch":
            return self._format_mismatch_error(error_details, variant, tissue)
        if error_type == "existing_hypothesis":
            return self._format_existing_hypothesis_error(error_details, variant)
        if error_type == "non_sample_project":
            return self._format_non_sample_project_error(error_details)
        if error_type == "tissue_not_found":
            return self._format_tissue_not_found_error(error_details, tissue)

        return {"text": f"No hypothesis is generated: I couldn't find a project containing both **{variant}** and **{tissue}**."}
