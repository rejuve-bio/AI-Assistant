import logging
import re
from typing import Any, Dict, List, Optional

from app.prompts.hypothesis_prompt import go_term_selection_prompt, hypothesis_format_prompt, tissue_selection_prompt
from app.socket_manager import emit_to_user
from app.storage.redis import redis_manager

from .hypothesis_api import HypothesisAPIMixin
from .hypothesis_project import HypothesisProjectMixin

logger = logging.getLogger(__name__)


class HypothesisGeneration(HypothesisAPIMixin, HypothesisProjectMixin):

    def __init__(self, llm) -> None:
        self.llm = llm
        self._pending_fallback: Dict[str, Dict[str, Any]] = {}
        logger.info("HypothesisGeneration initialized with LLM")

    # --- Pending state (tissue / GO / sample-offer) ---

    def set_pending_tissue(self, user_id: str, variant: str, project_id: str, available_tissues: list) -> None:
        if redis_manager.is_available:
            redis_manager.set_pending_hypothesis(user_id, variant, project_id, available_tissues)
        else:
            self._pending_fallback[user_id] = {"variant": variant, "project_id": project_id, "available_tissues": available_tissues}

    def has_pending_tissue_for(self, user_id: str) -> bool:
        return redis_manager.get_pending_hypothesis(user_id) is not None if redis_manager.is_available else user_id in self._pending_fallback

    def clear_pending(self, user_id: str) -> None:
        redis_manager.clear_pending_hypothesis(user_id) if redis_manager.is_available else self._pending_fallback.pop(user_id, None)

    def _get_pending(self, user_id: str) -> Optional[Dict[str, Any]]:
        return redis_manager.get_pending_hypothesis(user_id) if redis_manager.is_available else self._pending_fallback.get(user_id)

    def set_pending_go(self, user_id: str, enrich_id: str, hypothesis_id: str, go_terms: list, tissue: str = "", variant: str = "", project_id: str = "") -> None:
        if redis_manager.is_available:
            redis_manager.set_pending_go(user_id, enrich_id, hypothesis_id, go_terms, tissue=tissue, variant=variant, project_id=project_id)
        else:
            self._pending_fallback[f"go:{user_id}"] = {"enrich_id": enrich_id, "hypothesis_id": hypothesis_id, "go_terms": go_terms, "tissue": tissue, "variant": variant, "project_id": project_id}

    def has_pending_go_for(self, user_id: str) -> bool:
        return redis_manager.get_pending_go(user_id) is not None if redis_manager.is_available else f"go:{user_id}" in self._pending_fallback

    def _get_pending_go(self, user_id: str) -> Optional[Dict[str, Any]]:
        return redis_manager.get_pending_go(user_id) if redis_manager.is_available else self._pending_fallback.get(f"go:{user_id}")

    def _clear_pending_go(self, user_id: str) -> None:
        redis_manager.clear_pending_go(user_id) if redis_manager.is_available else self._pending_fallback.pop(f"go:{user_id}", None)

    def set_pending_sample_offer(self, user_id: str, variant: str, sample_project_id: str, sample_tissues: list) -> None:
        if redis_manager.is_available:
            redis_manager.set_pending_sample_offer(user_id, variant, sample_project_id, sample_tissues)
        else:
            self._pending_fallback[f"sample_offer:{user_id}"] = {"variant": variant, "sample_project_id": sample_project_id, "sample_tissues": sample_tissues}

    def has_pending_sample_offer_for(self, user_id: str) -> bool:
        return redis_manager.get_pending_sample_offer(user_id) is not None if redis_manager.is_available else f"sample_offer:{user_id}" in self._pending_fallback

    def _get_pending_sample_offer(self, user_id: str) -> Optional[Dict[str, Any]]:
        return redis_manager.get_pending_sample_offer(user_id) if redis_manager.is_available else self._pending_fallback.get(f"sample_offer:{user_id}")

    def _clear_pending_sample_offer(self, user_id: str) -> None:
        redis_manager.clear_pending_sample_offer(user_id) if redis_manager.is_available else self._pending_fallback.pop(f"sample_offer:{user_id}", None)

    # --- Interaction handlers ---

    def handle_sample_offer_response(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        pending = self._get_pending_sample_offer(user_id)
        if not pending:
            return None
        q = query.lower().strip()
        is_new = len(query.split()) > 6 or "?" in query or q.startswith(("what", "how", "why", "find", "show", "explain", "tell", "is ", "are ", "can "))
        if is_new:
            self._clear_pending_sample_offer(user_id)
            return None
        is_no  = any(w in q for w in ("no", "nope", "don't", "dont", "not now", "skip", "cancel", "never mind", "nevermind"))
        is_yes = any(w in q for w in ("yes", "sure", "okay", "ok", "go ahead", "yeah", "yep", "please", "start", "let's", "lets", "do it"))
        if is_no:
            self._clear_pending_sample_offer(user_id)
            return {"text": "No problem! When you're ready, you can set up your own project on the platform first.", "agents_completed": ["hypothesis_agent"]}
        if is_yes:
            self._clear_pending_sample_offer(user_id)
            tissues = pending["sample_tissues"]
            if tissues:
                self.set_pending_tissue(user_id, pending["variant"], pending["sample_project_id"], tissues)
                tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in tissues])
                return {"text": f"Let's run it on the sample project! Which tissue context would you like to use?\n\n{tissue_list}\n\nJust pick one and I'll kick off the analysis.", "agents_completed": ["hypothesis_agent"]}
            return {"text": "I couldn't find available tissues for the sample project. Please try again later.", "agents_completed": ["hypothesis_agent"]}
        return {"text": "Just say **yes** to run the sample project, or **no** if you'd prefer not to.", "agents_completed": ["hypothesis_agent"]}

    @staticmethod
    def _build_history_ctx(history: list) -> str:
        if not history:
            return ""
        lines = []
        for h in history[-2:]:
            q = h.get("question", "")
            a = (h.get("context") or {}).get("answer", "")
            if q:
                lines.append(f"User previously: {q}")
            if a:
                lines.append(f"Assistant previously: {a[:200]}")
        return "Recent conversation context:\n" + "\n".join(lines) + "\n\n" if lines else ""

    def _llm_pick_from_list(self, query: str, options: List[str], prompt_template: str, label: str, history: list = None) -> Optional[int]:
        numbered = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        history_ctx = self._build_history_ctx(history)
        try:
            text = self.llm.generate(history_ctx + prompt_template.format(query=query, **{label: numbered}))
            text = text.strip() if isinstance(text, str) else str(text).strip()
            if text == "NEW_QUESTION":
                return None
            if text in ("UNCLEAR", "SMALL_TALK"):
                return -1 if text == "UNCLEAR" else -2
            idx = int(text) - 1
            if 0 <= idx < len(options):
                return idx
        except Exception:
            pass
        return -1

    def handle_go_selection(self, user_id: str, query: str, token: str, history: list = None) -> Optional[Dict[str, Any]]:
        pending = self._get_pending_go(user_id)
        if not pending:
            return None
        go_terms = pending["go_terms"]
        idx = self._llm_pick_from_list(query, [f"{t['name']} ({t['id']}) — p={t['p']:.4f}" for t in go_terms], go_term_selection_prompt, "go_list", history=history)
        if idx is None:
            self._clear_pending_go(user_id)
            return None
        if idx == -1:
            go_list = "\n".join([f"{i+1}. {t['name']} ({t['id']}) — p={t['p']:.4f}" for i, t in enumerate(go_terms)])
            return {"text": f"I didn't catch that. Reply with a number (1–{len(go_terms)}) or the GO term name:\n{go_list}", "agents_completed": ["hypothesis_agent"]}
        chosen = go_terms[idx]
        enrich_id, hypothesis_id, tissue = pending["enrich_id"], pending["hypothesis_id"], pending.get("tissue", "")
        self._clear_pending_go(user_id)
        result = self._finalize_with_go(token, enrich_id, chosen, user_id)
        if "resource" in result and isinstance(result["resource"], dict):
            result["resource"]["id"] = hypothesis_id
            if redis_manager.is_available:
                if tissue:
                    redis_manager.set_hypothesis_meta(hypothesis_id, tissue, chosen["name"], chosen["id"])
                redis_manager.set_generated_hypothesis(user_id, pending.get("variant", ""), pending.get("project_id", ""), hypothesis_id)
        result.setdefault("agents_completed", ["hypothesis_agent"])
        return result

    def handle_tissue_selection(self, user_id: str, query: str, token: str, history: list = None) -> Optional[Dict[str, Any]]:
        pending = self._get_pending(user_id)
        if not pending:
            return None
        available_tissues = pending["available_tissues"]
        idx = self._llm_pick_from_list(query, [self._tissue_display(t) for t in available_tissues], tissue_selection_prompt, "tissue_list", history=history)
        if idx == -2: return None
        if idx is None:
            self.clear_pending(user_id)
            return None
        if idx == -1:
            tissue_list = "\n".join([f"{i+1}. {self._tissue_display(t)}" for i, t in enumerate(available_tissues)])
            return {"text": f"I wasn't able to match **\"{query}\"** to any of the available tissues.\n\nHere are the tissues you can pick from:\n{tissue_list}\n\nReply with a number or the tissue name.", "agents_completed": ["hypothesis_agent"]}
        chosen = self._tissue_name(available_tissues[idx])
        self.clear_pending(user_id)
        result = self._run_enrichment_pipeline(token, {"variant": pending["variant"], "tissue_name": chosen, "project_id": pending["project_id"]}, user_id)
        result.setdefault("agents_completed", ["hypothesis_agent"])
        return result

    # --- Query formatting ---

    def format_user_query(self, query: str, user_id) -> Dict[str, Any]:
        logger.info("Formatting user query: %s", query)
        try:
            response = self.llm.generate(hypothesis_format_prompt.format(question=query))
            if not response or isinstance(response, str):
                emit_to_user(user=user_id, message="Warning: Could not parse extraction response")
                return {}
            regex_variants = re.findall(r'\brs\d+\b', query, re.IGNORECASE)
            if regex_variants:
                llm_variant = response.get("variant")
                if isinstance(llm_variant, list):
                    llm_variant = llm_variant[0] if llm_variant else None
                user_variant = regex_variants[0]
                if not llm_variant or llm_variant.lower() != user_variant.lower():
                    logger.warning("Overriding LLM variant '%s' with regex match '%s'", llm_variant, user_variant)
                response["variant"] = regex_variants  # preserve all regex-found variants
            return response
        except Exception as e:
            logger.error("Error formatting user query: %s", e)
            return {}

    # --- Pipeline execution ---

    def _run_enrichment_pipeline(self, token: str, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        emit_to_user(user=user_id, message=f"Starting enrichment for {params.get('variant')}...")
        step1 = self._step_1_enrich(token, params)
        if "error" in step1:
            return {"text": f"I tried to start the enrichment, but failed: {step1['error']}"}
        emit_to_user(user=user_id, message="Waiting for analysis to complete...")
        step2 = self._step_2_poll(token, step1["hypothesis_id"])
        if "error" in step2:
            return {"text": f"Enrichment started, but failed during processing: {step2['error']}"}
        emit_to_user(user=user_id, message="Fetching enrichment results...")
        step3 = self._step_3_get_results(token, step2["enrich_id"])
        if "error" in step3:
            return {"text": f"Analysis completed, but failed to retrieve results: {step3['error']}"}
        go_terms = step3.get("GO_terms", [])
        if not go_terms:
            return {"text": "Analysis completed, but no significant GO terms were found."}
        self.set_pending_go(user_id, step2["enrich_id"], step1["hypothesis_id"], go_terms,
                            tissue=params.get("tissue_name", ""), variant=params.get("variant", ""), project_id=params.get("project_id", ""))
        go_list = "\n".join([f"{i+1}. **{t['name']}** ({t['id']}) — genes: {', '.join(t.get('genes', [])[:4])}{'...' if len(t.get('genes', [])) > 4 else ''} — p={t['p']:.4f}" for i, t in enumerate(go_terms)])
        return {"text": f"The enrichment analysis is done! I found **{len(go_terms)} significant biological pathways** linked to **{params.get('variant')}**.\n\nPick the GO term you'd like to build the hypothesis around:\n{go_list}\n\nJust reply with the number or name. Heads up — the final generation step can take up to 30 seconds, so hang tight after you choose!", "agents_completed": ["hypothesis_agent"]}

    def _finalize_with_go(self, token: str, enrich_id: str, go_term: dict, user_id: str) -> Dict[str, Any]:
        emit_to_user(user=user_id, message=f"Generating hypothesis for '{go_term['name']}' — this may take up to 30 seconds, please wait...")
        step4 = self._step_4_generate(token, enrich_id, go_term["id"])
        if "error" in step4:
            return {"text": f"Failed to generate final hypothesis summary: {step4['error']}"}
        return {"text": step4["summary"], "resource": {"id": step4.get("hypothesis_id", enrich_id), "type": "hypothesis", "graph": step4["graph"]}, "agents_completed": ["hypothesis_agent"]}

    def _handle_multiple_variants(self, token, params, user_id, variants, tissue) -> Dict[str, Any]:
        found = []
        failed_variants = []
        last_error_details = None
        for v in variants:
            validated_project_id, error_details = self.validate_project_context(token, v, tissue)
            if validated_project_id:
                found.append((v, validated_project_id))
            else:
                failed_variants.append(v)
                last_error_details = error_details

        parts = []
        if found:
            first_v, first_pid = found[0]
            queued = [v for v, _ in found[1:]] + failed_variants
            result = self._run_enrichment_pipeline(token, {**params, "variant": first_v, "project_id": first_pid}, user_id)
            parts.append(result.get("text", ""))
            if queued:
                parts.append(f"*(Starting with **{first_v}**. Once done, ask me to run: {', '.join(queued)})*")
        elif failed_variants and last_error_details:
            error_result = self._format_validation_error(last_error_details)
            failed_list = " and ".join(f"**{v}**" for v in failed_variants)
            parts.append(f"I couldn't find {failed_list} in your projects.\n\n{error_result.get('text', '')}")
            if (last_error_details or {}).get("error_type") == "non_sample_project":
                sample_info = last_error_details.get("sample_info")
                if sample_info:
                    self.set_pending_sample_offer(user_id, variant=failed_variants[0], sample_project_id=sample_info.get("id", ""), sample_tissues=sample_info.get("tissues", []))
        return {"text": "\n\n".join(parts), "agents_completed": ["hypothesis_agent"]}

    def _handle_existing_hypothesis(self, token, user_id, variant, tissue, error_details) -> Dict[str, Any]:
        hypothesis_id = error_details["hypothesis_id"]
        tissue_used = error_details.get("tissue_used", "")
        available_tissues = error_details.get("available_tissues", [])
        other_tissues = [t for t in available_tissues if self._tissue_name(t) != tissue_used] if tissue_used else available_tissues
        if other_tissues:
            self.set_pending_tissue(user_id, variant=error_details["variant"], project_id=error_details["project_id"], available_tissues=other_tissues)
        other_tissue_list = "\n".join([f"- {self._tissue_display(t)}" for t in other_tissues])
        if tissue and self._normalize_field(tissue) == self._normalize_field(tissue_used):
            msg = (
                f"**{tissue_used}** was already used for the existing hypothesis on **{error_details['variant']}**. "
                f"To generate a new one, pick a different tissue:\n\n{other_tissue_list}"
                if other_tissue_list
                else f"**{tissue_used}** was already used and there are no other tissues available."
            )
            return {"text": msg, "agents_completed": ["hypothesis_agent"], "is_existing_hypothesis": True}
        existing = self.get_by_hypothesis_id(token, hypothesis_id, user_id)
        hyp_meta = redis_manager.get_hypothesis_meta(hypothesis_id) if redis_manager.is_available else None
        go_used = (hyp_meta or {}).get("go_term_name", "")
        go_term_part = f", **{go_used}** GO term" if go_used else ""
        meta_line = f"*(Generated using: **{tissue_used}** tissue{go_term_part})*\n\n" if tissue_used else ""
        offer_line = f"\n\n---\nWant to generate a new hypothesis with a different tissue? Here are your options:\n{other_tissue_list}" if other_tissue_list else ""
        return {"text": f"{meta_line}{existing.get('text', '')}{offer_line}", "resource": existing.get("resource"), "agents_completed": ["hypothesis_agent"], "is_existing_hypothesis": True}

    def _handle_single_variant_error(self, user_id, variant, error_details, error_type) -> Dict[str, Any]:
        error_resp = self._format_validation_error(error_details)
        if error_type in ("tissue_missing", "tissue_not_found"):
            self.set_pending_tissue(user_id, variant=variant, project_id=error_details.get("project_id") or "", available_tissues=error_details.get("available_tissues", []))
        elif error_type == "non_sample_project":
            sample_info = error_details.get("sample_info")
            if sample_info:
                self.set_pending_sample_offer(user_id, variant=variant, sample_project_id=sample_info.get("id", ""), sample_tissues=sample_info.get("tissues", []))
        return error_resp

    def generate_hypothesis(self, token: str, user_query: str, user_id: str) -> Dict[str, Any]:
        logger.info("Starting hypothesis generation for: %s", user_query)
        emit_to_user(user=user_id, message="Analyzing your query...")
        params = self.format_user_query(user_query, user_id)
        if not params:
            return {"text": "Could not understand the biological parameters in your query."}
        variant = params.get("variant")
        tissue = params.get("tissue_name") or ""
        if isinstance(tissue, list):
            tissue = ""
        if not variant:
            return {"text": "Could not extract a genetic variant from your query. Please specify a variant (e.g. rs1421085)."}

        variants = variant if isinstance(variant, list) else [variant]
        if len(variants) > 1:
            return self._handle_multiple_variants(token, params, user_id, variants, tissue)

        variant = variants[0]
        params["variant"] = variant
        validated_project_id, error_details = self.validate_project_context(token, variant, tissue)
        if validated_project_id:
            params["project_id"] = validated_project_id
            return self._run_enrichment_pipeline(token, params, user_id)
        error_type = error_details.get("error_type") if error_details else None
        if error_type == "existing_hypothesis":
            return self._handle_existing_hypothesis(token, user_id, variant, tissue, error_details)
        return self._handle_single_variant_error(user_id, variant, error_details, error_type)
