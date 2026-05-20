from ..prompts.classifier_prompt import (
    aggeregator_prompt,
    VALIDATION_PROMPT,
    PLANNER_PROMPT,
    agent_descriptions,
)
from ..prompts.dependency_prompts import DEPENDENCY_SUMMARIZATION_PROMPT
from ..prompts.rag_prompts import CLARIFYING_QUESTIONS_PROMPT
from app.tools.platform.annotation.annotated_graph import Graph
from app.tools.platform.rag.rag import RAG
from app.tools.platform.annotation.summarizer import Graph_Summarizer
from app.tools.platform.hypothesis.hypothesis import HypothesisGeneration
from ..socket_manager import emit_to_user
from app.tools.platform.galaxy.galaxy import GalaxyHandler
from app.tools.platform.biogpt.biogpt import BioGPTAgentOpenVINO
from app.tools.platform.web_search import WebSearch
from app.tools.biomni import BiomniFunctionRetriever
from .code_executor import CodeExecutor
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from langgraph.types import Send
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import importlib
import json
import operator
import os
import logging

logger = logging.getLogger(__name__)

# ── Biomni direct-call argument extractor ───────────────────────────────────
BIOMNI_ARG_EXTRACT_PROMPT = """Extract argument values from this task for a function call.

Task: {task}
Function: {function_name}
Parameters: {params}

Respond with ONLY a JSON object of argument values. No explanation, no markdown.
Examples:
  {{"gene_name": "BRCA1"}}
  {{"gene_list": ["TP53", "MYC", "KRAS"], "collection": "H"}}
  {{"target_gene": "EGFR", "top_k": 20}}"""

# ── Dynamic replanner ───────────────────────────────────────────────────────
REPLAN_PROMPT = """You are checking whether a multi-step biomedical query is fully answered.

User query: {user_query}

Outputs collected so far:
{collected_outputs}

Is the query fully answered by these outputs?

Respond with EXACTLY one of:

SUFFICIENT

or, if something critical is clearly missing:

NEED_MORE:
[
  {{"id": {next_id}, "agent": "<agent>", "type": "informative", "input": "<specific instruction>", "depends_on": []}},
  ...
]

Rules:
- Maximum 2 additional steps.
- Use SUFFICIENT unless an entire major part of the query is completely unanswered.
- Valid agents: rag_agent, biogpt_agent, annotation_agent (sub_type: annotation_general),
  web_search_agent (sub_type: pubmed), biomni_agent, code_executor (type: action, tool: python).
- For biomni_agent: set input to the exact database lookup needed.
- Do NOT add steps that re-fetch what was already retrieved."""


def _merge_dicts(existing: Dict, new: Dict) -> Dict:
    merged = dict(existing) if existing else {}
    if new:
        merged.update(new)
    return merged


class AgentState(TypedDict):
    # Core
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    response: Dict[str, Any]
    error: str

    # Input context
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    pipeline_details: Dict[str, Any]
    query_types: List[str]

    # DAG plan (set by classifier)
    plan: Optional[List[Dict[str, Any]]]

    # Per-Send: which step this executor instance is handling
    current_step: Optional[Dict[str, Any]]

    # Parallel-safe accumulators (merged automatically by LangGraph reducers)
    completed_step_ids: Annotated[List[int], operator.add]
    step_outputs: Annotated[Dict[int, str], _merge_dicts]
    step_agent_outputs: Annotated[List[Dict[str, Any]], operator.add]

    # Action step retry state
    action_retry_count: int

    # Session ID — unique per query, scopes output files
    session_id: Optional[str]

    # Last N conversation turns fed in from MongoDB
    conversation_history: Optional[List[Dict[str, Any]]]

    # Replanning loop control (max 1 replan round)
    replan_count: int

    # Final aggregation
    agents_completed: Annotated[List[str], operator.add]
    agents_to_run: List[str]
    suggested_questions: Optional[List[str]]


class Orchestrator:
    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.store = mongo_db_manager
        self.embedding_model = embedding_model

        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.biogpt = BioGPTAgentOpenVINO(llm=advanced_llm)
        self.web_search = WebSearch(advanced_llm)
        biomni_retriever = BiomniFunctionRetriever(embedding_model)
        self.code_executor = CodeExecutor(advanced_llm, basic_llm, biomni_retriever=biomni_retriever)

        logger.info(f"Orchestrator initialized with llm: {type(advanced_llm).__name__}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Classifier + DAG Planner
    # ─────────────────────────────────────────────────────────────────────────

    def get_content_summaries(self, user_id, content_ids=None):
        content_summaries = []
        try:
            all_content = self.store.get_user_content_files(user_id)
        except Exception:
            return []

        filtered = (
            [c for c in all_content if c.get("content_id") in content_ids]
            if content_ids else all_content
        )

        for c in filtered:
            if c.get("content_type") == "pdf":
                content_summaries.append({
                    "content_id": c.get("content_id"),
                    "content_type": "pdf",
                    "filename": c.get("filename"),
                    "summary": c.get("summary") or "",
                })
            elif c.get("content_type") == "web":
                content_summaries.append({
                    "content_id": c.get("content_id"),
                    "content_type": "web",
                    "url": c.get("url"),
                    "title": c.get("title"),
                    "summary": c.get("summary") or "",
                })
        return content_summaries

    def _rewrite_query_if_followup(self, query: str, history: list) -> str:
        if not history:
            return query

        turns = ""
        for turn in history:
            turns += f"User: {turn.get('user', '')}\n"
            turns += f"Assistant: {turn.get('assistant_answer', '')[:600]}\n\n"

        prompt = (
            f"Conversation so far:\n{turns}"
            f"New message: \"{query}\"\n\n"
            f"If the new message references something from the conversation (e.g. 'explain further', 'that analysis', 'same thing', vague pronouns), "
            f"rewrite it as a complete self-contained question. "
            f"If it is already self-contained and unrelated to prior context, return it unchanged. "
            f"Return ONLY the final question, nothing else."
        )
        try:
            rewritten = self.basic_llm.generate(prompt).strip().strip('"')
            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten
        except Exception:
            return query

    def classify_query(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        user_id = state["user_id"]
        history = state.get("conversation_history") or []

        query = self._rewrite_query_if_followup(query, history)
        logger.info(f"Classifying query: {query}")

        content_summaries = self.get_content_summaries(user_id, state.get("content_ids"))

        try:
            raw = self.advanced_llm.generate(
                VALIDATION_PROMPT.format(query=query, agent_descriptions=agent_descriptions)
            )
            validation = raw if isinstance(raw, dict) else json.loads(
                raw.replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            logger.warning(f"Validation parse error: {e}, defaulting to valid")
            validation = {"is_valid": True}
        if not validation.get("is_valid", True):
            refusal = validation.get("refusal_message", "I can only help with biological queries.")
            return {"response": {"text": refusal, "json_format": None}, "plan": [], "messages": [AIMessage(content=refusal)]}
        previous_attempt = ""
        if history:
            last = history[-1]
            prev_q = last.get("user", "")
            prev_a = last.get("assistant_answer", "")
            if prev_a:
                previous_attempt = (
                    f"PREVIOUS ATTEMPT:\n"
                    f"User asked: {prev_q}\n"
                    f"Response was: {prev_a[:800]}"
                )

        try:
            raw = self.advanced_llm.generate(
                PLANNER_PROMPT.format(
                    query=query,
                    agent_descriptions=agent_descriptions,
                    content_summaries=content_summaries,
                    previous_attempt=previous_attempt,
                )
            )
            plan_result = raw if isinstance(raw, dict) else json.loads(raw.replace("```json", "").replace("```", "").strip())
            steps = plan_result.get("steps", [])
        except Exception as e:
            logger.error(f"Planning parse error: {e}, falling back to rag_agent")
            steps = [{"id": 1, "agent": "rag_agent", "type": "informative", "input": query, "depends_on": []}]
        if not steps:
            steps = [{"id": 1, "agent": "rag_agent", "type": "informative", "input": query, "depends_on": []}]

        plan_summary = " → ".join(f"[{s['id']}]{s['agent']}" + (f"({s.get('sub_type','') or s.get('tool','')})" if s.get('sub_type') or s.get('tool') else "") for s in steps)
        logger.info(f"[PLAN] {len(steps)} steps: {plan_summary}")
        return {
            "plan": steps,
            "completed_step_ids": [],
            "step_outputs": {},
            "step_agent_outputs": [],
            "action_retry_count": 0,
            "agents_to_run": [s["agent"] for s in steps],
            "messages": [HumanMessage(content=f"Plan: {len(steps)} steps")],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — DAG Scheduler: find ready steps, dispatch via Send
    # ─────────────────────────────────────────────────────────────────────────

    def dag_scheduler(self, state: AgentState):
        plan = state.get("plan", [])
        completed = set(state.get("completed_step_ids", []))

        if not plan:
            return "aggregator"

        ready = [
            step for step in plan
            if step["id"] not in completed
            and all(dep in completed for dep in step.get("depends_on", []))
        ]

        if not ready:
            if len(completed) >= len(plan):
                logger.info("All steps completed → aggregator")
                return "aggregator"
            # Some steps are blocked — shouldn't happen in a valid DAG
            logger.warning("No ready steps but plan incomplete. Routing to aggregator.")
            return "aggregator"

        step_outputs = state.get("step_outputs", {})
        sends = []
        for step in ready:
            resolved_input = self._resolve_tokens(
                step.get("input", state["user_query"]), step_outputs
            )
            resolved_step = {**step, "resolved_input": resolved_input}
            logger.info(f"Dispatching step {step['id']}: {step['agent']}")
            sends.append(Send("step_executor", {**state, "current_step": resolved_step}))

        return sends

    def _resolve_tokens(self, raw_input: str, step_outputs: Dict[int, str]) -> str:
        result = raw_input
        for step_id, output in step_outputs.items():
            token = "{" + f"step_{step_id}_output" + "}"
            if token in result:
                context = self._prepare_dependency_context(output, step_id)
                result = result.replace(token, context)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Step Executor: runs one step (informative or action)
    # ─────────────────────────────────────────────────────────────────────────

    def step_executor(self, state: AgentState) -> Dict[str, Any]:
        current_step = state.get("current_step", {})
        step_id = current_step.get("id", 0)
        agent_name = current_step.get("agent", "rag_agent")
        step_type = current_step.get("type", "informative")
        step_input = current_step.get("resolved_input") or current_step.get("input") or state["user_query"]

        logger.info(f"Executing step {step_id}: {agent_name} ({step_type})")
        emit_to_user(
            user=state["user_id"],
            message=f"Step {step_id}: Running {agent_name.replace('_', ' ').title()}...",
        )

        try:
            if step_type == "action":
                result = self._run_action_step(state, current_step, step_input)
            else:
                result = self._run_informative_step(state, current_step, step_input)
        except Exception as e:
            logger.error(f"Step {step_id} ({agent_name}) crashed: {e}", exc_info=True)
            result = {
                "text": f"Error in {agent_name}: {str(e)}",
                "source": agent_name,
                "json_format": None,
            }

        output_text = result.get("text", "") or ""
        if isinstance(output_text, list):
            output_text = " | ".join(
                p.get("content", "") if isinstance(p, dict) else str(p) for p in output_text
            )

        return {
            "completed_step_ids": [step_id],
            "step_outputs": {step_id: output_text},
            "step_agent_outputs": [{
                "step_id": step_id,
                "agent": agent_name,
                "type": step_type,
                "source": result.get("source", agent_name),
                "text": output_text,
                "json_format": result.get("json_format"),
                "files": result.get("files", []),
                "provenance": result.get("provenance"),
            }],
            "agents_completed": [agent_name],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Sync Node: after all parallel executors finish, decide next
    # ─────────────────────────────────────────────────────────────────────────

    def sync_node(self, state: AgentState) -> Dict[str, Any]:
        completed = set(state.get("completed_step_ids", []))
        total = len(state.get("plan", []))
        logger.info(f"Sync: {len(completed)}/{total} steps done")
        return {}  # state already updated by step_executor reducers

    def should_continue_dag(self, state: AgentState) -> str:
        completed = set(state.get("completed_step_ids", []))
        plan = state.get("plan", [])
        if len(completed) < len(plan):
            return "dag_scheduler"
        # All current steps done — replan once if we haven't yet
        if state.get("replan_count", 0) == 0:
            logger.info("All steps done → replanner (first check)")
            return "replanner"
        logger.info("All steps done → aggregator (already replanned)")
        return "aggregator"

    # ─────────────────────────────────────────────────────────────────────────
    # Informative agent dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _run_informative_step(self, state: AgentState, step: Dict, step_input: str) -> Dict:
        agent = step.get("agent", "rag_agent")
        sub_type = step.get("sub_type", "")

        tool_dispatch = {
            "rag_agent":               lambda: self._rag_step(state, step_input),
            "annotation_agent":        lambda: self._annotation_step(state, step_input, sub_type),
            "galaxy_agent":            lambda: self._galaxy_step(state, step_input),
            "biogpt_agent":            lambda: self._biogpt_step(state, step_input),
            "hypothesis_agent":        lambda: self._hypothesis_step(state, step_input),
            "content_retrieval_agent": lambda: self._content_retrieval_step(state, step_input),
            "web_search_agent":        lambda: self._web_search_step(state, step_input, sub_type),
            "biomni_agent":            lambda: self._biomni_lookup_step(state, step_input),
        }

        handler = tool_dispatch.get(agent)
        if handler:
            return handler()
        logger.warning(f"Unknown tool: {agent}, falling back to rag")
        return self._rag_step(state, step_input)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual informative agent implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _rag_step(self, state: AgentState, step_input: str) -> Dict:
        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            response = self.rag.get_result_from_rag(
                step_input, state["user_id"], content_ids=state.get("content_ids")
            )
            text = response.get("text", str(response)) if isinstance(response, dict) else str(response)
            return {"text": text, "source": "knowledge base", "json_format": None}
        except Exception as e:
            logger.error(f"RAG step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": "knowledge base", "json_format": None}

    def _build_annotation_text(self, json_format: dict) -> str:
        if not json_format:
            return "The annotation structure was created successfully (see structured data)."
        nodes = json_format.get("nodes", [])
        substituted = []
        for n in nodes:
            if n.get("status") is False and n.get("suggestion") and not n.get("not_validated"):
                props = n.get("properties", {})
                suggestion = n.pop("suggestion")
                for key in props:
                    substituted.append((props[key], suggestion))
                    props[key] = suggestion
                    break
                n["status"] = True
                n.pop("validation_error", None)

        failed = [n for n in nodes if n.get("status") is False]
        text = "The annotation structure was created successfully (see structured data)."

        if substituted:
            if len(substituted) == 1:
                orig, sub = substituted[0]
                text += f' Note: "{orig}" was not found in the database. The structure was created for the closest match "{sub}" instead — did you mean "{sub}"?'
            else:
                pairs = ", ".join(f'"{o}" → "{s}"' for o, s in substituted)
                text += f" Note: The following were not found and substituted with the closest match: {pairs}."

        if failed:
            missing_parts = []
            all_suggestions = []
            for n in failed:
                not_validated = n.get("not_validated")
                suggestions = n.get("suggestions", {})
                if not_validated:
                    items = not_validated if isinstance(not_validated, list) else [not_validated]
                    for item in items:
                        missing_parts.append(f'"{item}"')
                        suggestion = suggestions.get(item)
                        if suggestion:
                            all_suggestions.append((item, suggestion))
                else:
                    props = n.get("properties", {})
                    name = next(iter(props.values()), n.get("type", "unknown"))
                    missing_parts.append(f'"{name}"')
                    suggestion = n.get("suggestion")
                    if suggestion:
                        all_suggestions.append((name, suggestion))
            verb = "was" if len(missing_parts) == 1 else "were"
            joined = ", ".join(missing_parts)
            text += f" Note: {joined} {verb} not found in the database but {verb} included in the structure based on the provided information."
            if all_suggestions:
                if len(all_suggestions) == 1:
                    text += f" Did you mean \"{all_suggestions[0][1]}\"?"
                else:
                    did_you_mean = ", ".join(f'"{o}" → "{s}"' for o, s in all_suggestions)
                    text += f" Did you mean: {did_you_mean}?"
        return text

    def _annotation_step(self, state: AgentState, step_input: str, sub_type: str) -> Dict:
        query_type = sub_type if sub_type in ("annotation_biological", "annotation_general") else "annotation_biological"
        try:
            emit_to_user(user=state["user_id"], message="Querying annotation database...")
            pipeline_resp = self.annotation_graph.process_annotation_query(
                query=step_input, user_id=state["user_id"], query_type=query_type
            )
            if pipeline_resp.get("success"):
                json_format = pipeline_resp.get("json_format")
                provenance = pipeline_resp.get("provenance")

                if json_format:
                    # annotation_biological: JSON structure for frontend visualization
                    text = self._build_annotation_text(json_format)
                else:
                    # annotation_general: actual data returned with provenance
                    text = pipeline_resp.get("summary", "")

                result = {"text": text, "json_format": json_format, "source": "annotation database"}
                if provenance:
                    result["provenance"] = provenance
                return result

            return {"text": pipeline_resp.get("error", "Annotation failed"), "source": "annotation database", "json_format": None}
        except Exception as e:
            logger.error(f"Annotation step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": "annotation database", "json_format": None}

    def _galaxy_step(self, state: AgentState, step_input: str) -> Dict:
        try:
            emit_to_user(user=state["user_id"], message="Retrieving Galaxy information...")
            response = self.galaxy_handler.get_galaxy_info(step_input, state["user_id"], state["token"])
            text = response.get("text", str(response)) if isinstance(response, dict) else str(response)
            return {"text": text, "source": "Galaxy platform", "json_format": None}
        except Exception as e:
            logger.error(f"Galaxy step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": "Galaxy platform", "json_format": None}

    def _biogpt_step(self, state: AgentState, step_input: str) -> Dict:
        try:
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            text = self.biogpt.generate_answer(step_input)
            return {"text": text, "source": "BioGPT", "json_format": None}
        except Exception as e:
            logger.error(f"BioGPT step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": "BioGPT", "json_format": None}

    def _hypothesis_step(self, state: AgentState, step_input: str) -> Dict:
        try:
            emit_to_user(user=state["user_id"], message="Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"], user_query=step_input, user_id=state["user_id"]
            )
            text = response.get("text", str(response)) if isinstance(response, dict) else str(response)
            return {"text": text, "source": "hypothesis generator", "json_format": None}
        except Exception as e:
            logger.error(f"Hypothesis step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": "hypothesis generator", "json_format": None}

    def _content_retrieval_step(self, state: AgentState, step_input: str) -> Dict:
        user_id = state["user_id"]
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        emit_to_user(user=user_id, message="Retrieving content...")
        content_parts = []

        try:
            if graph_id:
                gs = self._answer_from_graph_summaries(step_input, user_id, resource, token, graph_id)
                if gs:
                    content_parts.append({"source": f"graph:{graph_id}", "content": gs.get("text", "")})

            if urls:
                ur = self.galaxy_handler.get_galaxy_info(query=step_input, user_id=user_id, token=token, urls=urls)
                if ur:
                    for f in (urls if isinstance(urls, list) else [urls]):
                        content_parts.append({"source": f"file:{f}", "content": ur.get("text", str(ur))})

            if content_ids:
                rc = self.rag.get_result_from_rag(step_input, user_id, content_ids)
                if rc:
                    content_parts.append({
                        "source": f"content IDs: {', '.join(content_ids)}",
                        "content": rc.get("text", str(rc)),
                    })

            return {"text": content_parts, "source": "content retrieval", "json_format": None}
        except Exception as e:
            logger.error(f"Content retrieval step error: {e}", exc_info=True)
            return {"text": [], "source": "content retrieval", "json_format": None}

    def _web_search_step(self, state: AgentState, step_input: str, sub_type: str) -> Dict:
        try:
            emit_to_user(user=state["user_id"], message=f"Searching {sub_type or 'web'}...")
            result = self.web_search.search(query=step_input, sub_type=sub_type)
            return {"text": result, "source": f"web:{sub_type or 'general'}", "json_format": None}
        except Exception as e:
            logger.error(f"Web search step error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "source": f"web:{sub_type}", "json_format": None}

    # ─────────────────────────────────────────────────────────────────────────
    # Action agent dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_file_paths(self, user_id: str, content_ids) -> list:
        """Resolve content_ids to absolute file paths on disk."""
        if not content_ids or not self.store:
            return []
        paths = []
        for cid in (content_ids if isinstance(content_ids, list) else [content_ids]):
            try:
                record = self.store.get_content_file_by_id(user_id, cid)
                if not record:
                    continue
                # Prefer explicit file_path stored in DB
                fp = record.get("file_path")
                if fp and os.path.exists(fp):
                    paths.append(fp)
                    continue
                # Fallback: PDFs are always at pdfs_uploaded/pdfs/{content_id}.pdf
                if record.get("content_type") == "pdf":
                    pdf_path = os.path.join("pdfs_uploaded", "pdfs", f"{cid}.pdf")
                    if os.path.exists(pdf_path):
                        paths.append(os.path.abspath(pdf_path))
            except Exception as e:
                logger.warning(f"Could not resolve file path for content_id {cid}: {e}")
        return paths

    def _run_action_step(self, state: AgentState, step: Dict, step_input: str) -> Dict:
        tool = step.get("tool", "python")
        step_id = step.get("id", 0)
        user_id = state["user_id"]

        file_paths = self._resolve_file_paths(user_id, state.get("content_ids"))
        if file_paths:
            logger.info(f"Action step {step_id}: passing {len(file_paths)} file(s) to sandbox")

        emit_to_user(user=user_id, message=f"Executing {tool} code...")

        result = self.code_executor.execute(
            step_input=step_input,
            tool=tool,
            step_id=step_id,
            user_id=user_id,
            retry_count=state.get("action_retry_count", 0),
            file_paths=file_paths,
            session_id=state.get("session_id"),
        )

        files = result.get("output_files", [])

        # Inline informative: biological interpretation of the result
        if result.get("success") and result.get("stdout"):
            interp_prompt = (
                f"The following is the output of a {tool} execution step in a biomedical analysis pipeline.\n"
                f"User request: {step_input}\n\n"
                f"Output:\n{result['stdout'][:2000]}\n\n"
                f"Briefly explain what this output means biologically. "
                f"If it contains statistics, numbers or file names, interpret them in plain language."
            )
            try:
                interpretation = self.basic_llm.generate(interp_prompt)
                combined = f"{result['stdout']}\n\n---\nInterpretation: {interpretation}"
            except Exception:
                combined = result["stdout"]
        else:
            combined = result.get("stderr") or result.get("error") or "Execution produced no output."

        return {
            "text": combined,
            "source": f"code executor ({tool})",
            "json_format": None,
            "files": files,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Biomni direct DB/data-lake lookup (no code generation)
    # ─────────────────────────────────────────────────────────────────────────

    def _biomni_lookup_step(self, state: AgentState, step_input: str) -> Dict:
        """Directly call a Biomni database or data-lake function for this step.

        The retriever picks the best-matching function, the LLM extracts the
        arguments from step_input, and we call the function and return the result.
        """
        try:
            emit_to_user(user=state["user_id"], message="Querying biomedical databases...")
            retriever = self.code_executor.biomni_retriever
            if not retriever or not retriever._schemas:
                return {"text": "Biomni retriever not available.", "source": "biomni", "json_format": None}

            candidates = retriever._rank_by_embedding(step_input, top_k=3)
            if not candidates:
                return {"text": "No matching Biomni function found.", "source": "biomni", "json_format": None}

            schema = candidates[0]
            params = schema.get("parameters", {})
            params_desc = ", ".join(f"{k}: {v}" for k, v in params.items())

            # Ask LLM to extract argument values
            arg_prompt = BIOMNI_ARG_EXTRACT_PROMPT.format(
                task=step_input,
                function_name=schema["name"],
                params=params_desc,
            )
            args_raw = self.basic_llm.generate(arg_prompt)
            try:
                # Strip markdown fences if present
                clean = args_raw.strip().strip("```json").strip("```").strip()
                kwargs = json.loads(clean)
            except Exception:
                # Fallback: use the first param with step_input as value
                first_key = next(iter(params), None)
                kwargs = {first_key: step_input} if first_key else {}

            # Dynamically import and call the function
            mod = importlib.import_module(f"app.tools.biomni.{schema['module']}")
            func = getattr(mod, schema["name"])
            logger.info(f"biomni_lookup: calling {schema['module']}.{schema['name']}({kwargs})")
            result = func(**kwargs)

            # Format result as text
            text = json.dumps(result, indent=2, default=str)
            provenance = {
                "database": f"Biomni — {schema['name']}",
                "biomni_apis": [schema["name"]],
                "source_databases": [result.get("source", schema["module"])],
                "node_types_queried": [],
            }
            return {"text": text, "source": f"biomni:{schema['name']}", "json_format": None, "provenance": provenance}

        except Exception as e:
            logger.error(f"Biomni lookup step error: {e}", exc_info=True)
            return {"text": f"Error calling Biomni function: {e}", "source": "biomni", "json_format": None}

    # ─────────────────────────────────────────────────────────────────────────
    # Replanner — checks if plan is complete or needs more steps
    # ─────────────────────────────────────────────────────────────────────────

    def replanner(self, state: AgentState) -> Dict[str, Any]:
        """After all current steps finish, ask the LLM if more steps are needed.

        Adds new steps to the plan (with new IDs) if something critical is missing.
        Increments replan_count so this runs at most once per query.
        """
        replan_count = state.get("replan_count", 0) + 1

        step_agent_outputs = state.get("step_agent_outputs", [])
        collected = "\n\n".join(
            f"[Step {o['step_id']} — {o.get('source', o['agent'])}]: {str(o.get('text', ''))[:600]}"
            for o in sorted(step_agent_outputs, key=lambda x: x.get("step_id", 0))
        )
        current_plan = state.get("plan", [])
        next_id = max((s["id"] for s in current_plan), default=0) + 1

        prompt = REPLAN_PROMPT.format(
            user_query=state.get("user_query", ""),
            collected_outputs=collected[:3000],
            next_id=next_id,
        )

        try:
            verdict = self.basic_llm.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Replanner LLM call failed: {e}")
            return {"replan_count": replan_count}

        if verdict.upper().startswith("SUFFICIENT"):
            logger.info("Replanner: SUFFICIENT — going to aggregator")
            return {"replan_count": replan_count}

        if verdict.upper().startswith("NEED_MORE:"):
            raw_json = verdict[len("NEED_MORE:"):].strip()
            try:
                new_steps = json.loads(raw_json)
                if isinstance(new_steps, list) and new_steps:
                    updated_plan = current_plan + new_steps
                    logger.info(f"Replanner: adding {len(new_steps)} step(s): {[s['agent'] for s in new_steps]}")
                    return {"plan": updated_plan, "replan_count": replan_count}
            except Exception as e:
                logger.warning(f"Replanner failed to parse new steps: {e}")

        return {"replan_count": replan_count}

    def after_replan(self, state: AgentState) -> str:
        """Route after replanner: if new unfinished steps exist → dag_scheduler, else → aggregator."""
        completed = set(state.get("completed_step_ids", []))
        plan = state.get("plan", [])
        unfinished = [s for s in plan if s["id"] not in completed]
        if unfinished:
            return "dag_scheduler"
        return "aggregator"

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregator
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        user_query = state.get("user_query", "")
        step_agent_outputs = state.get("step_agent_outputs", [])

        logger.info(f"Aggregating {len(step_agent_outputs)} step outputs")

        # Sort by step_id to preserve logical order
        step_agent_outputs = sorted(step_agent_outputs, key=lambda x: x.get("step_id", 0))

        json_format = next(
            (o["json_format"] for o in step_agent_outputs if o.get("json_format")), None
        )

        all_output_files = []
        for o in step_agent_outputs:
            all_output_files.extend(o.get("files") or [])

        sources_info = []
        for output in step_agent_outputs:
            content = output.get("text", "")
            if isinstance(content, list):
                content = " | ".join(
                    p.get("content", "") if isinstance(p, dict) else str(p) for p in content
                )
            content = str(content).strip() if content else ""
            if not content:
                continue
            sources_info.append(f"[{output.get('source', 'unknown')}]: {content}")

        if not sources_info:
            return {
                "response": {
                    "text": "I don't have enough information from the available sources to answer this.",
                    "json_format": json_format,
                }
            }

        combined = "\n\n".join(sources_info)
        json_note = "\n\nNote: Structured annotation data is also available." if json_format else ""

        files_note = ""
        if all_output_files:
            file_names = [os.path.basename(f) for f in all_output_files]
            files_note = f"\n\nGenerated files: {', '.join(file_names)}"

        # Build provenance note from all steps that returned source info
        provenance_lines = []
        for output in step_agent_outputs:
            prov = output.get("provenance")
            if not prov:
                continue
            db = prov.get("database", "")
            source_dbs = prov.get("source_databases", [])
            node_types = prov.get("node_types_queried", [])
            biomni_tools = prov.get("biomni_tools", [])
            biomni_apis = prov.get("biomni_apis", [])
            not_in_neo4j = prov.get("not_in_neo4j", [])

            if db and source_dbs:
                provenance_lines.append(
                    f"- {db} | data types: {', '.join(node_types)} | "
                    f"sourced from: {', '.join(source_dbs)}"
                )
            if biomni_tools:
                provenance_lines.append(f"- Biomni analysis tools used: {', '.join(biomni_tools)}")
            if biomni_apis:
                provenance_lines.append(f"- External APIs queried: {', '.join(biomni_apis)}")
            if not_in_neo4j:
                provenance_lines.append(
                    f"- Not found in our database — sourced externally: {', '.join(not_in_neo4j)}"
                )

        provenance_note = (
            "\n\nData sources used:\n" + "\n".join(provenance_lines)
            if provenance_lines else ""
        )

        plan = state.get("plan", [])
        execution_context = ""
        if len(plan) > 1:
            execution_context = "\n\nExecution Flow:\n"
            completed = set(state.get("completed_step_ids", []))
            for step in plan:
                status = "✓" if step["id"] in completed else "⊘"
                execution_context += f"{status} Step {step['id']}: {step['agent']}\n"

        try:
            prompt = aggeregator_prompt.format(
                user_query=user_query,
                combined_responses=combined,
                json_note=json_note,
                files_note=files_note,
                execution_context=execution_context,
                provenance_note=provenance_note,
            )
            aggregated = self.advanced_llm.generate(prompt)
            if isinstance(aggregated, str) and aggregated.strip() == user_query.strip():
                aggregated = combined  # echo guard
        except Exception as e:
            logger.error(f"Aggregation LLM error: {e}", exc_info=True)
            aggregated = combined

        return {"response": {"text": aggregated, "json_format": json_format, "files": all_output_files}}

    # ─────────────────────────────────────────────────────────────────────────
    # Finalizer
    # ─────────────────────────────────────────────────────────────────────────

    def finalize_response(self, state: AgentState) -> Dict[str, Any]:
        response = state.get("response", {})
        user_id = state.get("user_id")
        query = state.get("user_query", "")

        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
        response.setdefault("json_format", None)

        if state.get("resource"):
            response["resource"] = state["resource"]

        response_text = response.get("text", "")
        if response_text and len(response_text.strip()) > 20:
            try:
                prompt = CLARIFYING_QUESTIONS_PROMPT.format(
                    user_query=query, assistant_response=response_text
                )
                result = self.basic_llm.generate(prompt)
                questions = []
                if result:
                    for line in result.strip().split("\n"):
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith(("-", "•"))):
                            q = line.split(".", 1)[-1].strip() if "." in line and line[0].isdigit() else line.strip("- •").strip()
                            if q and len(q) > 5:
                                questions.append(q)
                if questions:
                    response["suggested_questions"] = questions[:5]
            except Exception as e:
                logger.error(f"Clarifying questions error: {e}")

        emit_to_user(user=user_id, message=response, status="completed")
        return {"response": response}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_dependency_context(self, dep_output: str, dependency_id: int) -> str:
        MAX_CHARS = 1500
        if len(dep_output) <= MAX_CHARS:
            return dep_output
        try:
            summary = self.basic_llm.generate(
                DEPENDENCY_SUMMARIZATION_PROMPT.format(content=dep_output)
            )
            return f"[Summary from step {dependency_id}]: {summary}"
        except Exception:
            return dep_output[:MAX_CHARS] + "... [truncated]"

    def _answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        try:
            if resource == "annotation":
                result = self.graph_summarizer.summary(token=token, graph_id=graph_id, user_query=query)
            elif resource == "hypothesis":
                result = self.hypothesis_generation.get_by_hypothesis_id(token, graph_id, user_id, query)
            else:
                return {"text": "Invalid resource type.", "json_format": None}
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return {"text": text, "json_format": None}
        except Exception as e:
            logger.error(f"Graph summary error: {e}", exc_info=True)
            return {"text": f"Error: {e}", "json_format": None}
