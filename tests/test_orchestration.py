"""Unit tests for the framework-agnostic assistant orchestration boundary."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.orchestration.contracts import AgentName, AssistantRequest
from app.orchestration.agents.biogpt import BioGPTQueryAgent
from app.orchestration.agents.dependencies import AgentDependencies
from app.orchestration.agents.rag import RagQueryAgent
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


def test_rag_agent_injects_pubmed_fallback_without_knowing_the_workflow():
    rag = MagicMock()
    rag.get_result_from_rag.return_value = {"text": "No relevant information found.", "confidence": 0.1}
    emit_status = MagicMock()
    dependencies = AgentDependencies(
        rag=rag, biogpt=MagicMock(), basic_llm=MagicMock(), advanced_llm=MagicMock(),
        annotation_graph=MagicMock(), hypothesis_generation=MagicMock(),
        galaxy_handler=MagicMock(), graph_summarizer=MagicMock(),
        store=MagicMock(), emit_status=emit_status,
    )

    state = AssistantRequest(message="RNA", user_id="user-1").initial_state()
    state["agents_to_run"] = ["rag_agent"]
    update = RagQueryAgent(dependencies).execute(state)

    assert update["agents_to_run"] == ["rag_agent", "pubmed_agent"]
    assert update["agents_completed"] == ["rag_agent"]
    assert update["rag_response"]["confidence"] == 0.0


def test_biogpt_agent_uses_only_injected_dependencies():
    biogpt = MagicMock()
    biogpt.generate_answer.return_value = "RNA carries genetic information."
    dependencies = AgentDependencies(
        rag=MagicMock(), biogpt=biogpt, basic_llm=MagicMock(), advanced_llm=MagicMock(),
        annotation_graph=MagicMock(), hypothesis_generation=MagicMock(),
        galaxy_handler=MagicMock(), graph_summarizer=MagicMock(),
        store=MagicMock(), emit_status=MagicMock(),
    )

    update = BioGPTQueryAgent(dependencies).execute(
        AssistantRequest(message="What is RNA?", user_id="user-1").initial_state()
    )

    assert update["biogpt_response"]["text"] == "RNA carries genetic information."
    assert update["agents_completed"] == ["biogpt_agent"]
