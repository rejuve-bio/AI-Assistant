import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _target_dict(target_type: str, target_id: str, target_version: Optional[str]) -> Dict[str, Any]:
    target = {"type": target_type, "id": target_id}
    if target_version:
        target["version"] = target_version
    return target


def _branch_applies(inp: Dict[str, Any], filled: Dict[str, Any]) -> bool:
    when = inp.get("when")
    if not when:
        return True
    return all(filled.get(k) == v for k, v in when.items())


def _fill_inputs(
    inputs_schema: List[Dict[str, Any]], requested_values: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]]]:
    filled: Dict[str, Any] = dict(requested_values)
    unresolved: List[str] = []
    summary: List[Dict[str, str]] = []

    data_gating_keys = {
        k for inp in inputs_schema if inp.get("type") in ("data", "data_collection")
        for k in (inp.get("when") or {})
    }

    for_branches = dict(filled)
    for inp in inputs_schema:
        path = inp.get("path")
        if path in for_branches or inp.get("type") in ("data", "data_collection"):
            continue
        if "default" in inp:
            for_branches[path] = inp["default"]

    for inp in inputs_schema:
        path = inp.get("path")
        if not _branch_applies(inp, for_branches):
            continue
        is_file_decider = path in data_gating_keys
        has_default = "default" in inp and inp.get("type") not in ("data", "data_collection")
        required = not inp.get("optional", False) and (is_file_decider or not has_default)

        if path not in filled and inp.get("type") == "data":
            matches = inp.get("matches") or []
            if len(matches) == 1:
                filled[path] = {"src": "hda", "id": matches[0]["id"]}

        if path in filled:
            value = filled[path]
            display = value.get("id") if isinstance(value, dict) else value
            summary.append({"label": inp.get("label", path), "value": str(display)})
        elif required:
            unresolved.append(path)

    return filled, unresolved, summary


def propose_galaxy_run(
    api,
    token: str,
    thread_id: str,
    target_type: str,
    target_id: str,
    target_version: Optional[str],
    requested_values: Dict[str, Any],
) -> Dict[str, Any]:
    target = _target_dict(target_type, target_id, target_version)

    description = api.describe(token, thread_id, target)
    if isinstance(description, dict) and description.get("error"):
        return {"text": f"Couldn't read {target_id}'s requirements: {description['error']}", "json_format": None}

    inputs_schema = description.get("inputs") or []
    described_target = description.get("target") or target
    real_name = described_target.get("name", target_id)

    filled, unresolved, summary = _fill_inputs(inputs_schema, requested_values)

    if not unresolved and filled:
        validation = api.validate(token, thread_id, target, filled, {})
        if isinstance(validation, dict) and validation.get("errors"):
            for path, message in validation["errors"].items():
                unresolved.append(f"{path} ({message})")
                filled.pop(path, None)

    full_description = api.describe(token, thread_id, target, full=True)
    all_values: List[Dict[str, Any]] = []
    defaults_count = 0
    if not (isinstance(full_description, dict) and full_description.get("error")):
        for inp in full_description.get("inputs") or []:
            path = inp.get("path")
            is_default = path not in filled
            if is_default:
                defaults_count += 1
            value = filled.get(path, inp.get("default"))
            display = value.get("id") if isinstance(value, dict) else value
            all_values.append({
                "section": inp.get("section"),
                "label": inp.get("label", path),
                "value": display,
                "default": is_default,
            })

    data = {
        "target": described_target,
        "inputs": filled,
        "parameters": {},
        "summary": summary,
        "all_values": all_values,
        "defaults_count": defaults_count,
        "unresolved": unresolved,
    }

    if unresolved:
        first = unresolved[0]
        inp = next((i for i in inputs_schema if i.get("path") == first), None)
        label = inp.get("label", first) if inp else first
        text = f"{real_name}: I still need {label} to proceed. What should it be?"
        if len(unresolved) > 1:
            text += f" ({len(unresolved)} setting(s) still needed in total -- I'll ask the rest after this one.)"
    else:
        text = f"{real_name} is ready to run with what's been set. Review it and run."

    return {"text": text, "json_format": data}
