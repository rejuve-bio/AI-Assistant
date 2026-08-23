"""LangGraph construction using the registry and typed orchestration contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from .contracts import AgentState, AssistantRequest, AssistantResponse
from .planner import ExecutionPolicy, QueryPlanner
from .registry import AgentRegistry


class AssistantWorkflow:
    """Owns LangGraph wiring; it has no Flask, database, or agent-service knowledge."""

    def __init__(
        self,
        planner: QueryPlanner,
        registry: AgentRegistry,
        aggregate: Callable[[AgentState], dict[str, Any]],
        finalize: Callable[[AgentState], dict[str, Any]],
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._registry = registry
        self._aggregate = aggregate
        self._finalize = finalize
        self._policy = policy or ExecutionPolicy()
        self._app = self._build().compile()

    def invoke(self, request: AssistantRequest) -> AssistantResponse:
        result = self._app.invoke(request.initial_state())
        return AssistantResponse.from_workflow(
            result.get("response"), result.get("agents_completed", []),
        )

    def _build(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("classifier", self._planner.classify)
        workflow.add_node("router", self._router)
        workflow.add_node("aggregator", self._aggregate)
        workflow.add_node("finalizer", self._finalize)
        for name in self._registry.names:
            workflow.add_node(name, self._registry.handler_for(name))

        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "router")
        destinations = {name: name for name in self._registry.names}
        destinations.update({"aggregator": "aggregator"})
        workflow.add_conditional_edges("router", self._policy.next_step, destinations)
        for name in self._registry.names:
            workflow.add_edge(name, "router")
        workflow.add_edge("aggregator", "finalizer")
        workflow.add_edge("finalizer", END)
        return workflow

    @staticmethod
    def _router(_: AgentState) -> dict[str, Any]:
        return {}
