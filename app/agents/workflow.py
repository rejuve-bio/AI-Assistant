import json
import logging
import re
import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.agents.state import AgentState, UNRESOLVED_MARKER, RESET
from app.agents.tools import TOOL_RUNNERS, grounding_violation, openai_tool_defs
from app.agents.proposal_requirements import FILE_FORMAT_REQUIREMENTS
from app.prompts.orchestrator_prompts import (
    ONE_ROUND_CHECK_PROMPT,
    BLOCKING_QUESTION_CHECK_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10
URL_IN_TEXT_RE = re.compile(r"https?://\S+")



class WorkflowMixin:
    def _create_workflow(self) -> StateGraph:
        logger.info("Creating LangGraph workflow: dynamic tool-calling orchestrator")

        workflow = StateGraph(AgentState)
        workflow.add_node("seed", self._seed)
        workflow.add_node("orchestrator", self._orchestrator)
        workflow.add_node("run_tool", self._run_tool)
        workflow.add_node("finalizer", self._finalize_response)

        workflow.set_entry_point("seed")
        workflow.add_edge("seed", "orchestrator")

        workflow.add_conditional_edges(
            "orchestrator", self._route_from_orchestrator,
            {"finalizer": "finalizer", "run_tool": "run_tool"},
        )
        workflow.add_conditional_edges(
            "run_tool", self._route_after_tool,
            {"orchestrator": "orchestrator", "finalizer": "finalizer"},
        )
        workflow.add_edge("finalizer", END)
        return workflow

    def _thread_resource_summaries(self, resources: list, token, user_id, thread_id) -> list:
        lines = []
        for r in resources:
            kind, rid = r.get("kind"), r.get("id")
            if not kind or not rid:
                continue
            try:
                if kind == "hypothesis":
                    result = self.hypothesis_generation.get_by_hypothesis_id(token, rid, user_id, thread_id=thread_id)
                elif kind == "annotation":
                    result = self.graph_summarizer.summary(graph_id=rid, token=token)
                else:
                    continue
            except Exception as e:
                logger.warning(f"Couldn't summarize thread resource {kind}:{rid}: {e}")
                continue

            text = result.get("text", "") if isinstance(result, dict) else str(result or "")
            if not text and isinstance(result, dict):
                summary = result.get("summary")
                if isinstance(summary, dict):
                    parts = []
                    if summary.get("summary"):
                        parts.append(summary["summary"])
                    node_labels = summary.get("node_count_by_label") or []
                    edge_labels = summary.get("edge_count_by_label") or []
                    if summary.get("node_count") is not None:
                        by_label = ", ".join(f"{n.get('count')} {n.get('label')}" for n in node_labels) if node_labels else None
                        parts.append(f"{summary['node_count']} nodes" + (f" ({by_label})" if by_label else ""))
                    if summary.get("edge_count") is not None:
                        by_label = ", ".join(f"{e.get('count')} {e.get('label')}" for e in edge_labels) if edge_labels else None
                        parts.append(f"{summary['edge_count']} edges" + (f" ({by_label})" if by_label else ""))
                    text = ", ".join(parts)
                elif isinstance(summary, str):
                    text = summary
            if not text or text == "NO summaries provided":
                continue
            short = text.strip().splitlines()[0][:200]
            lines.append(f"- {kind} {rid}: {short}")
        return lines

    def _seed(self, state: AgentState) -> Dict[str, Any]:

        query = state["user_query"]
        logger.info(f"[STEP] turn started for thread {state.get('thread_id')}: {query!r}")
        try:
            verdict = self.basic_llm.generate(ONE_ROUND_CHECK_PROMPT.format(query=query)).strip().upper()
        except Exception as e:
            logger.warning(f"One-round check failed, using the full loop: {e}")
            verdict = "MULTI_ROUND"

        notes = []
        if state.get("content_ids") or state.get("graph_id") or state.get("urls"):
            notes.append(
                "(Note: content is attached to this conversation -- an uploaded "
                "file, URL, or graph. If the question relates to it, call "
                "retrieve_content rather than asking the user to restate it.)"
            )

        summary_lines = []
        if state.get("thread_id") and state.get("user_id"):
            try:
                summary_lines = self.store.get_proposal_summary(state["thread_id"], state["user_id"])
            except Exception as e:
                logger.warning(f"Could not load proposal summary for thread {state.get('thread_id')}: {e}")
        if summary_lines:
            notes.append(
                "(Note: earlier proposals made in this conversation, for reference if this "
                "question refers back to one of them -- these are summaries only, not live "
                "data, so don't restate them as if freshly generated:\n"
                + "\n".join(summary_lines) + ")"
            )

        thread_resources = state.get("thread_resources") or []
        if thread_resources:
            resource_lines = self._thread_resource_summaries(
                thread_resources, state.get("token"), state.get("user_id"), state.get("thread_id")
            )
            if resource_lines:
                notes.append(
                    "(Note: resources chosen for this thread's context -- short summaries "
                    "only, call the matching tool for the full result if a question needs "
                    "more detail:\n" + "\n".join(resource_lines) + ")"
                )

        attached_files = state.get("attached_files") or []

        if attached_files:
            names = ", ".join(f.get("name") or f.get("id", "") for f in attached_files)
            notes.append(
                f"(Note: file(s) attached in this conversation: {names}. They are stored but "
                "not yet parsed or searchable -- you can acknowledge them by name, but you "
                "cannot read their contents or answer questions about what's inside them. If "
                "more than one file could plausibly fill the same role and it isn't clear "
                "which one the user means, ask them to say which one rather than guessing or "
                "ignoring the ambiguity.)"
            )

            hyp_formats = FILE_FORMAT_REQUIREMENTS.get("hypothesis", {})
            accepted = hyp_formats.get("accepted_extensions")
            if accepted and re.search(r"\bgwas\b|\bhypothesis\b|\bcredible set\b|\brs\d+\b", query, re.IGNORECASE):
                mismatched = [
                    f.get("name", "") for f in attached_files
                    if f.get("name") and not any(f["name"].lower().endswith(ext) for ext in accepted)
                ]
                formats_str = ", ".join(accepted)
                if mismatched:
                    notes.append(
                        f"(Note: this looks like a GWAS/hypothesis-related request, but the "
                        f"attached file(s) {', '.join(mismatched)} don't match the formats the "
                        f"real hypothesis service accepts for a GWAS file ({formats_str}). Tell "
                        "the user this plainly rather than assuming the file will work -- "
                        "generating a hypothesis from it isn't built yet regardless, but the "
                        "file format matters once it is.)"
                    )
                else:
                    notes.append(
                        f"(Note: this looks like a GWAS/hypothesis-related request. The real "
                        f"hypothesis service accepts {formats_str} for a GWAS file -- the "
                        "attached file's extension matches. Generating a hypothesis from it "
                        "isn't built yet, so say so, but confirm the format is right.)"
                    )

        attached_urls = set(state.get("urls") or [])
        unattached_urls = set(URL_IN_TEXT_RE.findall(query)) - attached_urls
        if unattached_urls:
            notes.append(
                "(Note: the message contains a URL "
                f"({', '.join(sorted(unattached_urls))}) that was not "
                "attached through the file/URL upload mechanism, so you "
                "cannot read or fetch it. Say plainly that you can't access "
                "a URL pasted directly in the message and ask the user to "
                "attach it instead -- never guess at its content or silently "
                "ignore it.)"
            )

        seed_content = query if not notes else f"{query}\n\n" + "\n".join(notes)

        return {
            "messages": [HumanMessage(content=seed_content)],
            "loop_iterations": 0,
            "one_round_only": "ONE_ROUND" in verdict,
            "proposals_baseline": len(state.get("proposals") or []),
            "annotation_response": RESET,
            "hypothesis_response": RESET,
            "galaxy_response": RESET,
            "rag_response": RESET,
            "content_retrieval_response": RESET,
            "biogpt_response": RESET,
            "pubmed_response": RESET,
            "clinical_trials_response": RESET,
        }

    def _is_blocking_question(self, content: str) -> bool:
        try:
            verdict = self.basic_llm.generate(
                BLOCKING_QUESTION_CHECK_PROMPT.format(content=content)
            ).strip().upper()
        except Exception as e:
            logger.warning(f"Blocking-question check failed, assuming blocking (safer to ask than silently skip): {e}")
            return True
        return "BLOCKING" in verdict

    def _as_chat_messages(self, messages) -> list:
        converted = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
        for m in messages:
            if isinstance(m, HumanMessage):
                converted.append({"role": "user", "content": m.content})
            elif isinstance(m, ToolMessage):
                converted.append({
                    "role": "tool", "tool_call_id": m.tool_call_id, "content": str(m.content),
                })
            elif isinstance(m, AIMessage):
                entry = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m.tool_calls
                    ]
                converted.append(entry)
        return converted

    def _orchestrator(self, state: AgentState) -> Dict[str, Any]:
        """Decide the next step: call tool(s), or respond -- fresh, every turn."""
        iterations = state.get("loop_iterations", 0) + 1
        if iterations > MAX_TOOL_ROUNDS:
            logger.warning("Orchestrator hit the iteration cap — forcing a final answer")
            return {
                "loop_iterations": iterations,
                "messages": [AIMessage(
                    content="I wasn't able to fully resolve this after several steps. "
                            "Here's what I found so far."
                )],
            }

        logger.info(f"[STEP] orchestrator round {iterations}: asking the LLM for the next step")
        chat_messages = self._as_chat_messages(state["messages"])
        result = self.advanced_llm.generate_with_tools(chat_messages, openai_tool_defs())

        content = result.get("content") or ""
        tool_calls = [
            {"name": tc["name"], "args": tc["arguments"], "id": tc["id"]}
            for tc in result.get("tool_calls", [])
        ]
        if tool_calls:
            logger.info(f"[STEP] orchestrator round {iterations} decided to call: " +
                        ", ".join(f"{tc['name']}({tc['args']})" for tc in tool_calls))
        else:
            logger.info(f"[STEP] orchestrator round {iterations} decided to answer directly (no tool call)")

        if not tool_calls and content.strip().endswith("?"):
            if self._is_blocking_question(content):
                logger.info("Final answer is a blocking question with no tool_calls — treating as an implicit ask_human")
                content_is_blocking = True
            else:
                logger.info("Final answer ends in '?' but is an optional closing offer — finalizing normally")
                content_is_blocking = False
        else:
            content_is_blocking = False

        if content_is_blocking:
            tool_calls = [{
                "name": "ask_human", "args": {"question": content},
                "id": f"implicit-{uuid.uuid4().hex[:8]}",
            }]

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return {"messages": [ai_message], "loop_iterations": iterations}

    def _route_from_orchestrator(self, state: AgentState):
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "finalizer"
        return [Send("run_tool", {**state, "current_tool_call": tc}) for tc in tool_calls]

    def _run_tool(self, state: AgentState) -> Dict[str, Any]:
        tc = state["current_tool_call"]
        name, arguments, call_id = tc["name"], tc["args"], tc["id"]
        logger.info(f"[STEP] running tool: {name}({arguments})")

        if name not in TOOL_RUNNERS:
            content = f"{UNRESOLVED_MARKER}'{name}' is not a real tool. Available tools: {', '.join(TOOL_RUNNERS)}."
            return {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=content)]}

        violation = grounding_violation(name, arguments, state)
        if violation:
            logger.warning(f"Rejected ungrounded tool call {name}({arguments}): {violation}")
            return {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=f"{UNRESOLVED_MARKER}{violation}")]}

        prior_result = self._find_prior_tool_result(state, name, arguments)
        if prior_result is not None:
            logger.info(f"[STEP] tool {name} reused a prior result instead of re-running it")
            update = {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=prior_result)]}
            prior_proposal = self._find_prior_proposal(state, name, arguments)
            if prior_proposal is not None:
                update["proposals"] = [prior_proposal]
            return update

        runner = getattr(self, TOOL_RUNNERS[name])
        result = runner(arguments, state)

        preview = (result.get("content") or "")[:200].replace("\n", " ")
        if result.get("error"):
            logger.info(f"[STEP] tool {name} finished with an error: {result['error']}")
        else:
            logger.info(f"[STEP] tool {name} finished: {preview!r}")

        update = dict(result.get("state_update") or {})
        update["messages"] = [ToolMessage(tool_call_id=call_id, name=name, content=result.get("content", ""))]
        if result.get("agent_name"):
            update["agents_completed"] = [result["agent_name"]]
        if result.get("error"):
            update["error"] = result["error"]
        return update

    @staticmethod
    def _find_prior_tool_result(state: AgentState, name: str, arguments: dict):
        messages = state.get("messages", [])
        for i, m in enumerate(messages):
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc["name"] != name or tc["args"] != arguments:
                    continue
                for later in messages[i + 1:]:
                    if isinstance(later, ToolMessage) and later.tool_call_id == tc["id"]:
                        content = later.content
                        if isinstance(content, str) and content.startswith(UNRESOLVED_MARKER):
                            break
                        return content
        return None

    @staticmethod
    def _find_prior_proposal(state: AgentState, name: str, arguments: dict):
        for entry in reversed(state.get("proposals") or []):
            if entry.get("tool") == name and entry.get("arguments") == arguments:
                return {"kind": entry["kind"], "data": entry["data"]}
        return None

    def _route_after_tool(self, state: AgentState) -> str:
        if state.get("one_round_only") and state.get("loop_iterations", 0) <= 1:
            return "finalizer"
        return "orchestrator"
