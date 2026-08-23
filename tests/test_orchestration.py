"""Unit tests for the framework-agnostic assistant orchestration boundary."""

from __future__ import annotations

import pytest

from app.orchestration.contracts import AgentName, AssistantRequest
from app.orchestration.planner import ExecutionPolicy, QueryPlanner
from app.orchestration.registry import AgentDefinition, AgentRegistry
from app.orchestration.workflow import AssistantWorkflow


class StubLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def _handler(_state):
    return {}


def _registry() -> AgentRegistry:
    return AgentRegistry(
        [AgentDefinition(name, _handler, f"{name.value}_response") for name in AgentName]
    )


def _planner(response) -> QueryPlanner:
    return QueryPlanner(
        llm=StubLLM(response),
        classifier_prompt="query={query}; content={content_summaries}",
        content_summaries=lambda _user_id, _content_ids: [],
        registry=_registry(),
    )


def test_request_builds_a_complete_initial_state():
    request = AssistantRequest(message="What is RNA?", user_id="user-1", token="jwt")

    state = request.initial_state()

    assert state["user_query"] == "What is RNA?"
    assert state["agents_to_run"] == []
    assert state["agents_completed"] == []
    assert state["response"] == {"text": "", "json_format": None}


def test_planner_uses_structured_classifier_output_and_validates_agents():
    update = _planner({"query_types": ["literature"]}).classify(
        AssistantRequest(message="RNA literature", user_id="user-1").initial_state()
    )

    assert update["query_types"] == ["literature"]
    assert update["agents_to_run"] == [
        AgentName.RAG.value,
        AgentName.PUBMED.value,
        AgentName.CLINICAL_TRIALS.value,
    ]


def test_hypothesis_resource_bypasses_classifier_and_uses_existing_graph():
    planner = _planner({"query_types": ["rag"]})
    update = planner.classify(
        AssistantRequest(
            message="Explain this graph", user_id="user-1", resource="hypothesis", graph_id="graph-1"
        ).initial_state()
    )

    assert update["query_types"] == ["hypothesis_generation"]
    assert update["agents_to_run"] == [AgentName.CONTENT_RETRIEVAL.value]


def test_execution_policy_preserves_graph_dependency_and_safe_parallelism():
    state = AssistantRequest(message="FTO", user_id="user-1", graph_id="graph-1").initial_state()
    state["agents_to_run"] = [
        AgentName.CONTENT_RETRIEVAL.value,
        AgentName.ANNOTATION.value,
        AgentName.BIOGPT.value,
    ]

    assert ExecutionPolicy().next_step(state) == [
        AgentName.CONTENT_RETRIEVAL.value,
        AgentName.BIOGPT.value,
    ]


def test_registry_rejects_duplicate_agent_names():
    with pytest.raises(ValueError, match="unique"):
        AgentRegistry([
            AgentDefinition(AgentName.RAG, _handler, "rag_response"),
            AgentDefinition(AgentName.RAG, _handler, "other_response"),
        ])


def test_workflow_executes_a_registered_agent_and_normalizes_its_response():
    def rag_handler(_state):
        return {
            "rag_response": {"text": "RNA is a nucleic acid."},
            "agents_completed": [AgentName.RAG.value],
        }

    registry = AgentRegistry([AgentDefinition(AgentName.RAG, rag_handler, "rag_response")])
    planner = QueryPlanner(
        llm=StubLLM({"query_types": ["rag"]}),
        classifier_prompt="{query}",
        content_summaries=lambda _user_id, _content_ids: [],
        registry=registry,
    )
    workflow = AssistantWorkflow(
        planner=planner,
        registry=registry,
        aggregate=lambda state: {"response": state["rag_response"]},
        finalize=lambda state: {"response": state["response"]},
    )

    response = workflow.invoke(AssistantRequest(message="What is RNA?", user_id="user-1"))

    assert response.text == "RNA is a nucleic acid."
    assert response.agents_completed == [AgentName.RAG.value]
