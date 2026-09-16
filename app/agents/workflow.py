import json
import logging
import re
import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.agents.state import AgentState
from app.agents.tools import TOOL_RUNNERS, grounding_violation, openai_tool_defs

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10
UNRESOLVED_MARKER = "[not run] "
URL_IN_TEXT_RE = re.compile(r"https?://\S+")

ONE_ROUND_CHECK_PROMPT = """A user sent this message to a biology research
assistant with several tools available (gene annotation, literature search,
hypothesis generation, Galaxy, BioGPT, content retrieval):
"{query}"

Decide: can this be FULLY answered by ONE ROUND of tool calls -- whether
that's a single tool, or several tools that are all independent of each
other and can run at once -- with nothing further needed afterward? Or does
answering it require MULTIPLE ROUNDS, where a later tool call depends on an
earlier one's actual result, or it's genuinely unclear which tool(s) apply?

- ONE_ROUND: a single tool, or several independent tools, fully answers this
  in one go. ("annotate BRCA1", "search PubMed for diabetes", "annotate
  BRCA1, and separately, find papers about diabetes")
- MULTI_ROUND: a later step needs to see an earlier step's real result first,
  or it's ambiguous what to do. ("find papers about X, then annotate the
  gene found", anything where one tool's target isn't already named)

Respond with exactly one word: ONE_ROUND or MULTI_ROUND."""

BLOCKING_QUESTION_CHECK_PROMPT = """An assistant just gave this as its final
answer to a user:
"{content}"

This ends with a question. Decide: is the assistant GENUINELY BLOCKED --
it cannot complete the task without the user answering this first (e.g.
"which gene did you mean, X or Y?")? Or has it already finished the task,
and this is just an OPTIONAL closing offer to help further (e.g. "would you
like to know more about X?", "should I look into this further?")?

- BLOCKING: the task is incomplete without this answer.
- OPTIONAL: the task is already done; this only offers to continue.

Respond with exactly one word: BLOCKING or OPTIONAL."""

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are a biology research assistant orchestrating a set of tools. Rules:\n"
    "1. Only call a tool when every one of its required arguments is a REAL, "
    "KNOWN value right now -- never a placeholder or a guess. If a value "
    "isn't known yet, call only the tool(s) whose arguments you DO know, and "
    "wait for their real results before deciding the rest.\n"
    "2. Call multiple tools in the same turn only when they are genuinely "
    "independent of each other.\n"
    "3. If you need to ask the user something, you MUST call ask_human -- "
    "never write the question as your own plain-text answer.\n"
    "4. Once you have enough information, answer directly with no further "
    "tool calls -- that ends the conversation."
)


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

    def _seed(self, state: AgentState) -> Dict[str, Any]:

        query = state["user_query"]
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

        chat_messages = self._as_chat_messages(state["messages"])
        result = self.advanced_llm.generate_with_tools(chat_messages, openai_tool_defs())

        content = result.get("content") or ""
        tool_calls = [
            {"name": tc["name"], "args": tc["arguments"], "id": tc["id"]}
            for tc in result.get("tool_calls", [])
        ]

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

        if name not in TOOL_RUNNERS:
            content = f"{UNRESOLVED_MARKER}'{name}' is not a real tool. Available tools: {', '.join(TOOL_RUNNERS)}."
            return {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=content)]}
        violation = grounding_violation(name, arguments, state)
        if violation:
            logger.warning(f"Rejected ungrounded tool call {name}({arguments}): {violation}")
            return {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=f"{UNRESOLVED_MARKER}{violation}")]}

        prior_result = self._find_prior_tool_result(state, name, arguments)
        if prior_result is not None:
            logger.info(f"Reusing prior result for repeated call {name}({arguments}) instead of re-running it")
            content = (
                f"You already called {name} with these exact arguments earlier in "
                f"this conversation -- here is that result again, no need to call "
                f"it again: {prior_result}"
            )
            return {"messages": [ToolMessage(tool_call_id=call_id, name=name, content=content)]}

        runner = getattr(self, TOOL_RUNNERS[name])
        result = runner(arguments, state)

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
                            break  # a rejection, not a real result -- keep looking
                        return content
        return None

    def _route_after_tool(self, state: AgentState) -> str:
        if state.get("one_round_only") and state.get("loop_iterations", 0) <= 1:
            return "finalizer"
        return "orchestrator"
