"""Narrow dependency contracts shared by concrete agent handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


StatusEmitter = Callable[..., None]


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    """Services injected into agents by the application composition root."""

    rag: Any
    biogpt: Any
    basic_llm: Any
    advanced_llm: Any
    annotation_graph: Any
    hypothesis_generation: Any
    galaxy_handler: Any
    graph_summarizer: Any
    store: Any  # MongoDB manager
    emit_status: StatusEmitter
