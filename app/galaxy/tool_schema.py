"""
Dynamic tool input schema resolution from the Galaxy API.

Replaces the static tool_info.py dictionary by querying Galaxy's
``show_tool(tool_id, io_details=True)`` endpoint at runtime, so
*any* installed tool can be executed — not just the 11 hardcoded ones.
"""

import logging

logger = logging.getLogger(__name__)


class ToolSchemaResolver:
    """Dynamically resolve Galaxy tool input schemas via the BioBlend API."""

    def __init__(self, gi_client):
        """
        Parameters
        ----------
        gi_client : bioblend.galaxy.GalaxyInstance
            A *low-level* BioBlend GalaxyInstance (the ``.gi`` attribute of
            the objects-layer ``GalaxyInstance``).
        """
        self.gi = gi_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_schema(self, tool_id: str) -> dict:
        """Fetch the full input/output schema for a Galaxy tool.

        Returns a simplified dict with tool metadata, parsed inputs
        (name, type, default, whether optional) and outputs.
        """
        tool_details = self.gi.tools.show_tool(tool_id, io_details=True)
        return {
            "tool_id": tool_id,
            "name": tool_details.get("name", tool_id),
            "description": tool_details.get("description", ""),
            "inputs": self._parse_inputs(tool_details.get("inputs", [])),
            "outputs": [
                {
                    "name": o.get("name", ""),
                    "format": o.get("format", "data"),
                }
                for o in tool_details.get("outputs", [])
            ],
        }

    def build_tool_inputs(
        self,
        tool_id: str,
        dataset_id: str | None = None,
        user_params: dict | None = None,
    ) -> dict:
        """Build a Galaxy-compatible ``inputs`` dict for *tool_id*.

        Strategy:
        1. Fetch the tool schema from Galaxy.
        2. For every ``data``-type parameter, plug in *dataset_id*.
        3. For every other parameter, use *user_params* if provided,
           otherwise fall back to the Galaxy default.

        Parameters
        ----------
        tool_id : str
            The Galaxy tool identifier.
        dataset_id : str, optional
            The History Dataset Association (HDA) id to feed as input data.
        user_params : dict, optional
            User-supplied parameter overrides keyed by parameter name.

        Returns
        -------
        dict
            Ready-to-use ``inputs`` dict for ``GalaxyInstance.tools.run_tool``.
        """
        schema = self.get_tool_schema(tool_id)
        inputs: dict = {}
        user_params = user_params or {}

        for param in schema["inputs"]:
            name = param["name"]
            param_type = param["type"]

            # Data inputs — wire up the dataset
            if param_type == "data" and dataset_id is not None:
                inputs[name] = {"src": "hda", "id": dataset_id}

            # Data collection inputs
            elif param_type == "data_collection" and dataset_id is not None:
                inputs[name] = {"src": "hdca", "id": dataset_id}

            # Explicit user override
            elif name in user_params:
                inputs[name] = user_params[name]

            # Galaxy default (skip truly optional params with no default)
            elif param["default"] is not None:
                inputs[name] = param["default"]

        logger.info(
            "Built inputs for tool %s (%s): %d params resolved",
            schema["name"],
            tool_id,
            len(inputs),
        )
        return inputs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_inputs(raw_inputs: list) -> list[dict]:
        """Flatten Galaxy's nested input spec into a simple list of dicts."""
        parsed: list[dict] = []

        for inp in raw_inputs:
            # Galaxy wraps conditional/repeat/section blocks — recurse into them
            if inp.get("type") in ("conditional", "repeat", "section"):
                children = inp.get("inputs") or inp.get("cases", [{}])[0].get("inputs", [])
                parsed.extend(ToolSchemaResolver._parse_inputs(children))
                continue

            parsed.append(
                {
                    "name": inp.get("name", ""),
                    "label": inp.get("label", inp.get("name", "")),
                    "type": inp.get("type", "text"),
                    "optional": inp.get("optional", False),
                    "default": inp.get("value"),
                    "help": inp.get("help", ""),
                }
            )

        return parsed
