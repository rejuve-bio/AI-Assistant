import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GALAXY_READY_STATES = {"ok", "deferred"}
_GALAXY_FAILED_STATES = {"error", "failed_metadata", "discarded"}

_PAIR_PATTERN = re.compile(
    r"^(?P<sample>.+?)[._-](?:R)?(?P<read>[12])(?P<suffix>\.[^.]+(?:\.gz)?)$",
    re.IGNORECASE,
)


def pair_attached_files(files: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    by_sample: Dict[str, Dict[str, Dict[str, str]]] = {}
    unmatched: List[Dict[str, str]] = []

    for f in files:
        name = f.get("name") or ""
        m = _PAIR_PATTERN.match(name)
        if not m:
            unmatched.append(f)
            continue
        sample = m.group("sample")
        read = m.group("read")
        by_sample.setdefault(sample, {})[f"r{read}"] = f

    pairs: List[Dict[str, Any]] = []
    unpaired: List[Dict[str, str]] = list(unmatched)
    for sample, reads in by_sample.items():
        if "r1" in reads and "r2" in reads:
            pairs.append({"sample": sample, "r1": reads["r1"], "r2": reads["r2"]})
        else:
            unpaired.extend(reads.values())

    return pairs, unpaired


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


_FORMAT_SYNONYMS = {
    "tsv": "tabular", "csv": "tabular", "txt": "tabular",
    "fq": "fastq", "fq.gz": "fastq.gz",
    "fa": "fasta", "fa.gz": "fasta.gz",
    "fna": "fasta", "fna.gz": "fasta.gz",
}

_FASTQ_EXTS = {"fastq", "fq"}


def _file_matches_formats(name: str, formats: List[str]) -> bool:
    name = (name or "").lower()
    is_gz = name.endswith(".gz")
    base = name[:-3] if is_gz else name
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    formats_l = {f.lower() for f in formats}

    if ext:
        mapped = _FORMAT_SYNONYMS.get(ext, ext)
        candidates = {mapped, ext}
        candidates = {c + (".gz" if is_gz else "") for c in candidates}
        if candidates & formats_l:
            return True

    if ext in _FASTQ_EXTS:
        return any(f.startswith("fastq") and f.endswith(".gz") == is_gz for f in formats_l)

    return False


def _wait_for_galaxy_dataset(api, token: str, thread_id: str, galaxy_id: str,
                              attempts: int = 8, delay: float = 1.0) -> Optional[str]:
    for _ in range(attempts):
        files = api.thread_files(token, thread_id)
        for item in (files or {}).get("items", []):
            if item.get("src") == "hda" and item.get("id") == galaxy_id:
                state = item.get("state")
                if state in _GALAXY_READY_STATES or state in _GALAXY_FAILED_STATES:
                    return state
                break
        time.sleep(delay)
    return None


def _deliver_for_unresolved(
    api, file_service, token: str, thread_id: str,
    inputs_schema: List[Dict[str, Any]], filled: Dict[str, Any],
    unresolved: List[str], candidates: List[Dict[str, str]],
    used_galaxy_ids: Optional[set] = None, excluded_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    used_ids = set()
    used_galaxy_ids = set(used_galaxy_ids or set())
    excluded = set(excluded_paths or [])
    still_unresolved = list(unresolved)
    errors: List[str] = []
    for path in list(unresolved):
        if path in excluded:
            still_unresolved.remove(path)
            continue
        inp = next((i for i in inputs_schema if i.get("path") == path), None)
        if not inp or inp.get("type") not in ("data", "data_collection"):
            continue
        formats = inp.get("formats") or []
        format_matches = [
            f for f in candidates
            if f.get("id") not in used_ids and _file_matches_formats(f.get("name", ""), formats)
        ]
        resolved = False
        for candidate in format_matches:
            delivery = file_service.deliver_to_galaxy(token, candidate["id"], thread_id)
            galaxy_id = (((delivery or {}).get("delivery") or {}).get("galaxy") or {}).get("id")
            if not galaxy_id:
                errors.append(f"{inp.get('label', path)}: couldn't deliver {candidate.get('name')} to Galaxy ({(delivery or {}).get('error', 'unknown error')})")
                continue
            if galaxy_id in used_galaxy_ids:
                used_ids.add(candidate["id"])
                continue
            state = _wait_for_galaxy_dataset(api, token, thread_id, galaxy_id)
            if state in _GALAXY_FAILED_STATES:
                errors.append(f"{inp.get('label', path)}: {candidate.get('name')} failed in Galaxy (state: {state})")
                continue
            if state not in _GALAXY_READY_STATES:
                errors.append(f"{inp.get('label', path)}: {candidate.get('name')} is still being delivered to Galaxy, try again shortly")
                continue
            filled[path] = {"src": "hda", "id": galaxy_id}
            used_ids.add(candidate["id"])
            used_galaxy_ids.add(galaxy_id)
            still_unresolved.remove(path)
            resolved = True
            break
        if not resolved:
            continue
    return filled, still_unresolved, errors


def _fill_inputs(
    inputs_schema: List[Dict[str, Any]], requested_values: Dict[str, Any],
    excluded_paths: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]], set]:
    filled: Dict[str, Any] = dict(requested_values)
    unresolved: List[str] = []
    summary: List[Dict[str, str]] = []
    used_galaxy_ids: set = set()
    excluded = set(excluded_paths or [])

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
        if path in filled or path not in data_gating_keys or "default" not in inp:
            continue
        if inp.get("type") in ("data", "data_collection"):
            continue
        trial_branches = dict(for_branches)
        trial_branches[path] = inp["default"]
        gated = [
            other for other in inputs_schema
            if other.get("type") in ("data", "data_collection")
            and path in (other.get("when") or {})
            and _branch_applies(other, trial_branches)
        ]
        if gated and all(other.get("matches") for other in gated):
            filled[path] = inp["default"]
            for_branches[path] = inp["default"]

    for inp in inputs_schema:
        path = inp.get("path")
        if not _branch_applies(inp, for_branches):
            continue
        is_file_decider = path in data_gating_keys
        has_default = "default" in inp and inp.get("type") not in ("data", "data_collection")
        required = not inp.get("optional", False) and (is_file_decider or not has_default)

        if path in excluded and path not in filled:
            summary.append({"label": inp.get("label", path), "value": "(left out, as asked)"})
            continue

        if path not in filled and inp.get("type") == "data":
            matches = inp.get("matches") or []
            if len(matches) == 1 and matches[0]["id"] not in used_galaxy_ids:
                filled[path] = {"src": "hda", "id": matches[0]["id"]}
                used_galaxy_ids.add(matches[0]["id"])

        if path in filled:
            value = filled[path]
            display = value.get("id") if isinstance(value, dict) else value
            summary.append({"label": inp.get("label", path), "value": str(display)})
        elif required:
            unresolved.append(path)

    return filled, unresolved, summary, used_galaxy_ids


def _propose_single_run(
    api, token: str, thread_id: str, target: Dict[str, Any], described_target: Dict[str, Any],
    inputs_schema: List[Dict[str, Any]], real_name: str, requested_values: Dict[str, Any],
    file_service=None, uploaded_candidates: Optional[List[Dict[str, str]]] = None,
    excluded_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    filled, unresolved, summary, used_galaxy_ids = _fill_inputs(
        inputs_schema, requested_values, excluded_paths=excluded_paths
    )

    delivery_errors: List[str] = []
    if unresolved and file_service and uploaded_candidates:
        filled, unresolved, delivery_errors = _deliver_for_unresolved(
            api, file_service, token, thread_id, inputs_schema, filled, unresolved, uploaded_candidates,
            used_galaxy_ids=used_galaxy_ids, excluded_paths=excluded_paths,
        )
        for path in list(filled):
            inp = next((i for i in inputs_schema if i.get("path") == path), None)
            if inp and not any(s["label"] == inp.get("label", path) for s in summary):
                value = filled[path]
                display = value.get("id") if isinstance(value, dict) else value
                summary.append({"label": inp.get("label", path), "value": str(display)})

    validation_errors: Dict[str, str] = {}
    if not unresolved and filled:
        validation = api.validate(token, thread_id, target, filled, {})
        if isinstance(validation, dict) and validation.get("errors"):
            for path, message in validation["errors"].items():
                unresolved.append(path)
                validation_errors[path] = message
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
        "validation_errors": validation_errors,
    }

    if unresolved:
        first = unresolved[0]
        inp = next((i for i in inputs_schema if i.get("path") == first), None)
        label = inp.get("label", first) if inp else first
        reason = validation_errors.get(first)
        if reason:
            text = f"{real_name}: {label} was rejected -- {reason} What should it be instead?"
        elif inp and inp.get("type") in ("data", "data_collection"):
            formats = inp.get("formats") or []
            formats_note = f" (accepted formats: {', '.join(formats)})" if formats else ""
            text = (
                f"{real_name}: I need a file for {label}{formats_note}, but none of the "
                "files already in this thread match -- attach one, or tell me which "
                "already-uploaded file to use."
            )
        else:
            text = f"{real_name}: I still need {label} to proceed. What should it be?"
        if len(unresolved) > 1:
            text += f" ({len(unresolved)} setting(s) still needed in total -- I'll ask the rest after this one.)"
    else:
        text = f"{real_name} is ready to run with what's been set. Review it and run."
    if delivery_errors:
        text += " " + " ".join(delivery_errors)

    return {"text": text, "json_format": data}


def _unmatched_required_data_paths(
    inputs_schema: List[Dict[str, Any]], requested_values: Dict[str, Any],
    excluded_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    excluded = set(excluded_paths or [])
    for_branches = dict(requested_values)
    for inp in inputs_schema:
        path = inp.get("path")
        if path in for_branches or inp.get("type") in ("data", "data_collection"):
            continue
        if "default" in inp:
            for_branches[path] = inp["default"]

    out = []
    for inp in inputs_schema:
        path = inp.get("path")
        if inp.get("type") != "data" or inp.get("optional") or not _branch_applies(inp, for_branches):
            continue
        if path in requested_values or path in excluded:
            continue
        if len(inp.get("matches") or []) == 1:
            continue
        out.append(inp)
    return out


def _normalize_requested_values(
    inputs_schema: List[Dict[str, Any]], requested_values: Dict[str, Any]
) -> Dict[str, Any]:
    real_paths = {inp.get("path") for inp in inputs_schema}
    by_label = {
        (inp.get("label") or "").strip().lower(): inp.get("path")
        for inp in inputs_schema if inp.get("label")
    }
    normalized: Dict[str, Any] = {}
    for key, value in requested_values.items():
        if key in real_paths:
            normalized[key] = value
            continue
        by_label_match = by_label.get((key or "").strip().lower())
        if by_label_match:
            logger.info(f"requested_values key {key!r} isn't a real path -- remapped by label to {by_label_match!r}")
            normalized[by_label_match] = value
            continue
        by_type = [
            inp.get("path") for inp in inputs_schema
            if inp.get("type") == key and inp.get("path") not in normalized
        ]
        if len(by_type) == 1:
            logger.info(f"requested_values key {key!r} isn't a real path -- remapped by type to {by_type[0]!r}")
            normalized[by_type[0]] = value
            continue
        logger.warning(f"requested_values key {key!r} doesn't match any real input path, label, or unique type -- dropped")
    return normalized


def propose_galaxy_run(
    api,
    token: str,
    thread_id: str,
    target_type: str,
    target_id: str,
    target_version: Optional[str],
    requested_values: Dict[str, Any],
    file_service=None,
    uploaded_candidates: Optional[List[Dict[str, str]]] = None,
    excluded_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    target = _target_dict(target_type, target_id, target_version)

    description = api.describe(token, thread_id, target)
    if isinstance(description, dict) and description.get("error"):
        return {"text": f"Couldn't read {target_id}'s requirements: {description['error']}", "json_format": None}

    inputs_schema = description.get("inputs") or []
    described_target = description.get("target") or target
    real_name = described_target.get("name", target_id)

    requested_values = _normalize_requested_values(inputs_schema, requested_values)

    needed = _unmatched_required_data_paths(inputs_schema, requested_values, excluded_paths=excluded_paths)
    candidates = list(uploaded_candidates or [])

    file_groups: List[List[Dict[str, str]]] = []
    if len(needed) == 1 and file_service and candidates:
        formats = needed[0].get("formats") or []
        matching = [f for f in candidates if _file_matches_formats(f.get("name", ""), formats)]
        by_name: Dict[str, Dict[str, str]] = {}
        for f in matching:
            by_name[f.get("name", "")] = f
        deduped = list(by_name.values())
        file_groups = [[f] for f in deduped]
        leftover = [f for f in candidates if f not in matching]
    else:
        leftover = candidates

    if len(file_groups) <= 1:
        group_candidates = file_groups[0] if file_groups else candidates
        return _propose_single_run(
            api, token, thread_id, target, described_target, inputs_schema, real_name,
            requested_values, file_service=file_service, uploaded_candidates=group_candidates,
            excluded_paths=excluded_paths,
        )

    results = [
        _propose_single_run(
            api, token, thread_id, target, described_target, inputs_schema, real_name,
            requested_values, file_service=file_service, uploaded_candidates=group,
            excluded_paths=excluded_paths,
        )
        for group in file_groups
    ]
    proposals = [r["json_format"] for r in results if r.get("json_format")]
    ready = sum(1 for r in proposals if not r.get("unresolved"))
    names = ", ".join(
        "+".join(f.get("name", "?") for f in group) for group in file_groups
    )
    text = (
        f"{real_name}: built {len(proposals)} separate run(s), once per "
        f"{'pair' if len(file_groups[0]) == 2 else 'file'} ({names}) -- "
        f"{ready} ready to run, {len(proposals) - ready} still need something."
    )
    if leftover:
        text += f" {len(leftover)} other attached file(s) didn't match this tool's inputs and were left out."
    return {"text": text, "proposals": proposals}
