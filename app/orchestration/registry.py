"""The extension point for concrete assistant agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import AgentName, AgentState

AgentHandler = Callable[[AgentState], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """A concrete executable capability exposed to the workflow."""

    name: AgentName
    handler: AgentHandler
    response_key: str


class AgentRegistry:
    """Single source of truth for executable agents and workflow node names."""

    def __init__(self, definitions: Iterable[AgentDefinition]) -> None:
        registered = tuple(definitions)
        self._definitions = {definition.name.value: definition for definition in registered}
        if not self._definitions:
            raise ValueError("At least one agent definition is required")
        if len(self._definitions) != len(registered):
            raise ValueError("Agent names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definition_for(self, name: str) -> AgentDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ValueError(f"Unknown agent requested by plan: {name}") from exc

    def handler_for(self, name: str) -> AgentHandler:
        return self.definition_for(name).handler
