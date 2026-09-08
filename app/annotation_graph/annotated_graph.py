import copy
import difflib
import json
import logging
import os
from dotenv import load_dotenv
from app.annotation_graph.neo4j_handler import Neo4jConnection
from app.annotation_graph.schema_handler import SchemaHandler
from app.llm_handle.llm_models import LLMInterface
from app.prompts.annotation_prompts import (
    EXTRACT_RELEVANT_INFORMATION_PROMPT,
    JSON_CONVERSION_PROMPT,
    ORGANISM_DETECTION_PROMPT,
    SELECT_PROPERTY_VALUE_PROMPT,
    SELECT_PROPERTY_VALUES_BATCH_PROMPT,
    CONFIRMATION_CLASSIFICATION_PROMPT,
    DESCRIBE_ANNOTATION_PROMPT,
)
from app.socket_manager import emit_to_user


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class Graph:
    def __init__(self, llm: LLMInterface, schema_handler: SchemaHandler, fly_schema_handler: SchemaHandler = None) -> None:
        self.llm = llm
        self.schema_handler = schema_handler
        self.enhanced_schema = schema_handler.enhanced_schema
        self.neo4j = Neo4jConnection(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

        # Fly organism resources (schema + separate Neo4j connection)
        self.fly_schema_handler = fly_schema_handler
        self.fly_enhanced_schema = fly_schema_handler.enhanced_schema if fly_schema_handler else None
        self.fly_neo4j = Neo4jConnection(
            uri=os.getenv("FLY_NEO4J_URI"),
            username=os.getenv("FLY_NEO4J_USERNAME"),
            password=os.getenv("FLY_NEO4J_PASSWORD"),
        ) if os.getenv("FLY_NEO4J_URI") else None

        self._node_id_property = {
            "gene": "gene_name",
            "transcript": "transcript_id",
            "exon": "exon_id",
            "protein": "protein_name",
            "pathway": "pathway_name",
            "enhancer": "enhancer_id",
        }

    def query_knowledge_graph(self, json_query, token):
        """
        Query the knowledge graph service.

        Args:
            json_query (dict): The JSON query to be sent.

        Returns:
            dict: The JSON response from the knowledge graph service or an error message.
        """
        if isinstance(json_query, str):
            logger.info("passed json is a string changing it to a dicitionary")
            json_query = json.loads(json_query)

        logger.info("Starting knowledge graph query...")
        source = "ai-assistant"
        limit = 100

        params = {"source": source, "limit": limit, "properties": True}
        payload = {"requests": json_query}

        try:
            logger.debug(
                f"Sending request to {self.kg_service_url} with payload: {payload}"
            )
            response = requests.post(
                self.kg_service_url + "/query",
                json=payload,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            json_response = response.json()
            # logger.info(f"Successfully queried the knowledge graph. 'nodes count': {len(json_response.get('nodes'))} 'edges count': {len(json_response.get('edges', []))}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying knowledge graph: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return {"error": f"Failed to query knowledge graph: {str(e)}"}

    def validated_json(self, query, user_id):
        logger.info(f"Starting annotation query processing for question: '{query}'")

        # Extract relevant information
        relevant_information = self._extract_relevant_information(query)

        # Convert to initial JSON
        emit_to_user(user=user_id, message=f"Validating Constructed Json Format...")
        initial_json = self._convert_to_annotation_json(relevant_information, query)

        # Validate and update
        validation = self._validate_and_update(initial_json)

        # If validation failed, return the intermediate steps
        if validation["validation_report"]["validation_status"] == "failed":
            logger.error("Validation is failing *****sending the intial json format")
            return {
                "text": None,
                "json_format": initial_json,
                "organism": "human",
                "resource": {"id": None, "type": "annotation"},
            }

        # Use the updated JSON for subsequent steps
        validated_json = validation["updated_json"]
        # validated_json["question"] = query
        """
            TODO
            add query along with job id to specifiy to what query is the json requested is related to.
            """
        return {
            "text": None,
            "json_format": validated_json,
            "organism": "human",
            "resource": {"id": None, "type": "annotation"},
        }

    def generate_graph(self, query, validated_json, token):
        try:
            graph = self.query_knowledge_graph(validated_json, token)



            response = {
                "text": graph["answer"],
                "resource": {"id": graph["annotation_id"], "type": "annotation"},
            }
            # Store summary in Redis cache for 24 hours
            # redis_manager.create_graph(graph_id=graph_id, graph_summary=summary_text)

            logger.info("Completed query processing.")
            return response

        except Exception as e:
            logger.error(f"An error occurred during graph generation: {e}")
            return {
                "text": f"I apologize, but I wasn't able to generate the graph you requested. Could you please rephrase your question or provide additional details so I can better understand what you're looking for?"
            }

    def _detect_organism(self, query) -> str:
        q_lower = query.lower()
        fly_keywords = [
            # organism names
            "drosophila", "melanogaster", "dmel", "fruit fly", "fly schema",
            # flybase id prefixes
            "fbgn", "fbal", "fbtr", "fbbt", "fbdv", "fbrf",
            # fly-exclusive gene names (absent in human)
            " wg ", " hh ", " dpp ", " arm ", " ci ", " nkd ", " ptc ", " smo ",
            " bsk ", " hep ", " yki ", " sd ", " eve ", " ftz ", " vg ",
            " en ", " sev ", " boss ", " cos2 ", " puc ", " lats ", " wts ",
            # fly anatomy / tissue / cell terms
            "wing disc", "eye disc", "leg disc", "imaginal disc",
            "fat body", "hemocyte", "plasmatocyte", "crystal cell",
            "oenocyte", "malpighian", "salivary gland polytene",
            "dorsal vessel", "ring gland", "follicle cell", "nurse cell",
            # fly cell lines
            "s2 cell",
        ]
        if any(kw in q_lower for kw in fly_keywords):
            return "fly"
        try:
            prompt = ORGANISM_DETECTION_PROMPT.format(query=query)
            result = self.llm.generate(prompt)
            organism = str(result).strip().lower()
            if "fly" in organism:
                return "fly"
            return "human"
        except Exception as e:
            logger.warning(f"Organism detection failed, defaulting to human: {e}")
            return "human"

    def _extract_relevant_information(self, query, enhanced_schema=None):
        try:
            logger.info("Extracting relevant information from the query.")
            schema = enhanced_schema if enhanced_schema is not None else self.enhanced_schema
            prompt = EXTRACT_RELEVANT_INFORMATION_PROMPT.format(
                schema=schema, query=query
            )
            extracted_info = self.llm.generate(prompt)
            logger.info(f"Extracted data: \n{extracted_info}")
            return extracted_info
        except Exception as e:
            logger.error(f"Failed to extract relevant information: {e}")
            raise

    def _convert_to_annotation_json(self, relevant_information, query, enhanced_schema=None):
        try:
            logger.info("Converting relevant information to annotation JSON format.")
            schema = enhanced_schema if enhanced_schema is not None else self.enhanced_schema
            prompt = JSON_CONVERSION_PROMPT.format(
                query=query,
                extracted_information=relevant_information,
                schema=schema,
            )
            json_data = self.llm.generate(prompt)
            logger.info(f"Converted JSON:\n{json.dumps(json_data, indent=2)}")
            if not isinstance(json_data, dict):
                raise ValueError(f"Expected a JSON object from the LLM, got {type(json_data).__name__}: {json_data!r}")
            return json_data
        except Exception as e:
            logger.error(f"Failed to convert information to annotation JSON: {e}")
            raise

    def _validate_and_update(self, initial_json, neo4j=None, schema_handler=None):
        try:
            logger.info("Validating and updating the JSON structure.")
            _neo4j = neo4j if neo4j is not None else self.neo4j
            _schema_handler = schema_handler if schema_handler is not None else self.schema_handler
            node_types = {}
            validation_report = {
                "property_changes": [],
                "direction_changes": [],
                "removed_properties": [],
                "failed_nodes": [],
                "validation_status": "success",
            }

            # Create a deep copy to track changes
            updated_json = copy.deepcopy(initial_json)

            # Validate node properties
            if "nodes" not in updated_json:
                raise ValueError("The input JSON must contain a 'nodes' key.")

            # Pre-pass: collect all values that need Neo4j lookup grouped by (node_type, property_key)
            lookup_needed = {}  # (node_type, property_key) -> set of string values
            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})

                # Also validate the `id` field. Always check it against the real database
                # `id` property first (Cypher MATCH always keys on `id` — see json_to_cypher.py),
                # and additionally against the display property (if this type has one) in case
                # the caller typed a name/symbol instead of the raw id.
                node_db_id = node.get("id", "")
                if node_db_id:
                    lookup_needed.setdefault((node_type, "id"), set()).add(node_db_id)
                    id_prop = self._node_id_property.get(node_type.lower())
                    if id_prop:
                        lookup_needed.setdefault((node_type, id_prop), set()).add(node_db_id)

                for property_key, property_value in properties.items():
                    if not property_value and property_value != 0:
                        continue
                    if node.get("is_list") and isinstance(property_value, (str, list)):
                        items = (
                            [i.strip() for i in property_value.split(",") if i.strip()]
                            if isinstance(property_value, str)
                            else property_value
                        )
                        lookup_needed.setdefault((node_type, property_key), set()).update(items)
                    elif isinstance(property_value, str):
                        lookup_needed.setdefault((node_type, property_key), set()).add(property_value)

            # Run one batch Neo4j query per (node_type, property_key)
            similarity_cache = {}  # (node_type, property_key, value) -> [(similar_value, score), ...]
            for (node_type, property_key), values in lookup_needed.items():
                batch = _neo4j.get_similar_property_values_batch(node_type, property_key, list(values))
                for value, matches in batch.items():
                    similarity_cache[(node_type, property_key, value)] = matches

            # Pre-pass: collect non-exact items for a single batched LLM call
            batch_for_llm = {}  # item -> [(candidate, score), ...]
            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})

                # Check id field — skip straight to LLM disambiguation only if neither the
                # real `id` property nor the display property gave an exact match.
                node_db_id = node.get("id", "")
                if node_db_id:
                    direct = similarity_cache.get((node_type, "id", node_db_id), [])
                    if direct and direct[0][0].lower() == node_db_id.lower():
                        pass  # already the real database id — nothing to disambiguate
                    else:
                        id_prop = self._node_id_property.get(node_type.lower())
                        if id_prop:
                            similar = similarity_cache.get((node_type, id_prop, node_db_id), [])
                            if similar:
                                if similar[0][0].lower() != node_db_id.lower():
                                    batch_for_llm[node_db_id] = similar
                            else:
                                # No Neo4j candidates at all — still needs confirmation with empty list
                                batch_for_llm[node_db_id] = []
                        elif direct:
                            batch_for_llm[node_db_id] = direct
                        else:
                            batch_for_llm[node_db_id] = []

                for property_key, property_value in properties.items():
                    if not property_value and property_value != 0:
                        continue
                    if node.get("is_list") and isinstance(property_value, (str, list)):
                        items = (
                            [i.strip() for i in property_value.split(",") if i.strip()]
                            if isinstance(property_value, str) else property_value
                        )
                        for item in items:
                            similar = similarity_cache.get((node_type, property_key, item), [])
                            if similar:
                                if similar[0][0].lower() != item.lower():
                                    batch_for_llm[item] = similar
                            else:
                                batch_for_llm[item] = []
                    elif isinstance(property_value, str):
                        similar = similarity_cache.get((node_type, property_key, property_value), [])
                        if similar:
                            if similar[0][0].lower() != property_value.lower():
                                batch_for_llm[property_value] = similar
                        else:
                            batch_for_llm[property_value] = []

            # One LLM call for all ambiguous items → {item: {"value": ..., "auto_accept": bool} | None}
            llm_picks = self._select_best_matching_values_batch(batch_for_llm)

            for node in updated_json.get("nodes"):
                node_type = node.get("type")
                properties = node.get("properties", {})
                node_id = node.get("node_id")
                node_types[node_id] = node_type
                if not node.get("is_list"):
                    node["status"] = True

                # Validate `id` field if set. Always prefer a match against the real database
                # `id` property — that's what the Cypher MATCH clause keys on (json_to_cypher.py).
                # Only fall back to the display property (gene_name, pathway_name, ...) when the
                # input isn't already a raw id, and flag it for id-resolution below so the
                # display value gets swapped for the node's actual `id` before querying.
                node_db_id = node.get("id", "")
                if node_db_id:
                    id_prop = self._node_id_property.get(node_type.lower())
                    direct_values = similarity_cache.get((node_type, "id", node_db_id), [])

                    if direct_values and direct_values[0][0].lower() == node_db_id.lower():
                        # Input is already the real database id — normalize casing and move on
                        node["id"] = direct_values[0][0]
                    else:
                        similar_values = similarity_cache.get((node_type, id_prop, node_db_id), []) if id_prop else []
                        if similar_values and similar_values[0][0].lower() == node_db_id.lower():
                            # Exact match against the display property — resolve to the real id below
                            node["id"] = similar_values[0][0]
                            node["_id_needs_resolution"] = True
                        else:
                            pick = llm_picks.get(node_db_id)
                            candidates = similar_values or direct_values
                            top = self._closest_candidate(node_db_id, candidates)
                            suggestion = pick["value"] if pick else top
                            if pick and pick.get("auto_accept"):
                                # Trivial difference — silently fix the id
                                node["id"] = pick["value"]
                                if id_prop and similar_values:
                                    node["_id_needs_resolution"] = True
                            elif suggestion:
                                # Genuinely different, with a real candidate — ask user
                                node["status"] = False
                                node["needs_confirmation"] = True
                                node["pending_substitutions"] = {node_db_id: suggestion}
                                validation_report["failed_nodes"].append(
                                    {"node_id": node_id, "reason": f"'{node_db_id}' not found in database"}
                                )
                            else:
                                # Nothing even remotely similar exists — say so plainly instead
                                # of echoing the user's own input back as a fake "suggestion"
                                node["status"] = False
                                node["needs_confirmation"] = False
                                node["validation_error"] = f"'{node_db_id}' not found in the database, and no similar value exists."
                                validation_report["failed_nodes"].append(
                                    {"node_id": node_id, "reason": node["validation_error"]}
                                )

                # Track removed properties
                for property_key in list(properties.keys()):
                    property_value = properties[property_key]

                    if not property_value and property_value != 0:
                        del properties[property_key]
                        validation_report["removed_properties"].append(
                            {
                                "node_type": node_type,
                                "node_id": node_id,
                                "property": property_key,
                                "original_value": property_value,
                            }
                        )
                    elif node.get("is_list") and isinstance(property_value, (str, list)):
                        items = (
                            [i.strip() for i in property_value.split(",") if i.strip()]
                            if isinstance(property_value, str)
                            else property_value
                        )
                        validated_items = []
                        failed_items = []
                        item_suggestions = {}
                        for item in items:
                            similar_values = similarity_cache.get((node_type, property_key, item), [])
                            if similar_values:
                                top = similar_values[0][0]
                                if top.lower() == item.lower():
                                    # Exact case match — auto-accept
                                    validated_items.append(top)
                                else:
                                    pick = llm_picks.get(item)
                                    if pick is None:
                                        # LLM says no clear match — still ask with the closest Neo4j result
                                        failed_items.append(item)
                                        if similar_values:
                                            item_suggestions[item] = self._closest_candidate(item, similar_values)
                                    elif pick.get("auto_accept"):
                                        # Trivial typo/case/punctuation — silently fix
                                        validated_items.append(pick["value"])
                                    else:
                                        # Genuinely different entity — ask user
                                        failed_items.append(item)
                                        item_suggestions[item] = pick["value"]
                            else:
                                failed_items.append(item)

                        confirmable = {
                            item: item_suggestions[item]
                            for item in failed_items if item in item_suggestions
                        }
                        truly_missing = [item for item in failed_items if item not in item_suggestions]

                        properties[property_key] = ", ".join(validated_items + truly_missing)
                        if failed_items:
                            node["status"] = False
                            if confirmable:
                                node["needs_confirmation"] = True
                                node["pending_substitutions"] = confirmable
                                node["all_list_values"] = list(items)
                            if truly_missing:
                                node["not_validated"] = truly_missing
                            validation_report["failed_nodes"].append(
                                {"node_id": node_id, "reason": f"Could not find in database: {failed_items}"}
                            )
                        else:
                            node["status"] = True

                    elif isinstance(property_value, str):
                        similar_values = similarity_cache.get((node_type, property_key, property_value), [])
                        if similar_values:
                            top = similar_values[0][0]
                            if top.lower() == property_value.lower():
                                # Exact case match — auto-accept
                                properties[property_key] = top
                            else:
                                pick = llm_picks.get(property_value)
                                if pick is None:
                                    # LLM unsure — still ask with the closest Neo4j result
                                    best = self._closest_candidate(property_value, similar_values)
                                    node["status"] = False
                                    node["needs_confirmation"] = True
                                    node["pending_substitutions"] = {property_value: best}
                                    validation_report["failed_nodes"].append(
                                        {"node_id": node_id, "reason": f"'{property_value}' not found; nearest is '{best}'"}
                                    )
                                elif pick.get("auto_accept"):
                                    # Trivial difference — silently fix
                                    properties[property_key] = pick["value"]
                                else:
                                    # Genuinely different — ask user
                                    node["status"] = False
                                    node["needs_confirmation"] = True
                                    node["pending_substitutions"] = {property_value: pick["value"]}
                                    validation_report["failed_nodes"].append(
                                        {
                                            "node_id": node_id,
                                            "reason": f"'{property_value}' not found; nearest match is '{pick['value']}'",
                                        }
                                    )
                        else:
                            node["status"] = False
                            node["validation_error"] = f"'{property_value}' not found in the database."
                            validation_report["failed_nodes"].append(
                                {"node_id": node_id, "reason": node["validation_error"]}
                            )

            # Validate edge direction — remove edges that don't exist in the schema
            valid_predicates = []
            for edge in updated_json.get("predicates", []):
                s = node_types.get(edge["source"])
                t = node_types.get(edge["target"])
                rel = edge["type"]
                conn = f"{s}-{rel}-{t}"

                if conn in _schema_handler.processed_schema:
                    valid_predicates.append(edge)
                else:
                    rev = f"{t}-{rel}-{s}"
                    if rev in _schema_handler.processed_schema:
                        # Swap direction and keep
                        validation_report["direction_changes"].append(
                            {"relation_type": rel, "original": f"{s} → {t}", "corrected": f"{t} → {s}"}
                        )
                        edge["source"], edge["target"] = edge["target"], edge["source"]
                        valid_predicates.append(edge)
                    else:
                        # Not in schema at all — drop it silently
                        logger.warning(f"Dropping invalid predicate: {conn}")
                        validation_report.setdefault("removed_predicates", []).append(
                            {"type": rel, "source": s, "target": t}
                        )
            updated_json["predicates"] = valid_predicates

            for node in updated_json.get("nodes", []):
                node.pop("is_list", None)

            # Remove duplicate nodes (same type + properties) the LLM may have hallucinated
            updated_json["nodes"] = self._deduplicate_nodes(
                updated_json.get("nodes", []), updated_json.get("predicates", [])
            )

            logger.info(
                f"Validated and updated JSON: \n{json.dumps(updated_json, indent=2)}"
            )

            return {
                "updated_json": updated_json,
                "validation_report": validation_report,
                "candidates": batch_for_llm,  # {item: [(candidate, score), ...]}
            }

        except Exception as e:
            logger.error(f"Validation and update of JSON failed: {e}")
            validation_report["validation_status"] = "failed"
            validation_report["error_message"] = str(e)
            return {
                "updated_json": initial_json,
                "validation_report": validation_report,
                "candidates": {},
            }

    _LOOKALIKE_CHARS = str.maketrans({"0": "O", "1": "I", "5": "S"})

    def _closest_candidate(self, target: str, candidates: list):
        if not candidates:
            return None
        norm_target = target.upper().translate(self._LOOKALIKE_CHARS)
        return max(
            candidates,
            key=lambda c: difflib.SequenceMatcher(
                None, norm_target, c[0].upper().translate(self._LOOKALIKE_CHARS)
            ).ratio(),
        )[0]

    def _select_best_matching_property_value(self, user_input_value, possible_values):
        try:
            prompt = SELECT_PROPERTY_VALUE_PROMPT.format(
                search_query=user_input_value, possible_values=possible_values
            )
            selected_value = self.llm.generate(prompt)
            logger.info(f"Selected value: {selected_value}")
            return selected_value
        except Exception as e:
            logger.error(f"Failed to select property value: {e}")
            raise

    def _select_best_matching_values_batch(self, items_with_candidates: dict) -> dict:
        """One LLM call for all ambiguous items.
        Returns {item: {"value": str, "auto_accept": bool} | None}
        """
        if not items_with_candidates:
            return {}
        items_repr = {
            item: [{"value": v, "similarity": round(s, 2)} for v, s in candidates[:5]]
            for item, candidates in items_with_candidates.items()
        }
        prompt = SELECT_PROPERTY_VALUES_BATCH_PROMPT.format(
            items_json=json.dumps(items_repr, indent=2)
        )
        result = self.llm.generate(prompt)
        logger.info(f"Batch LLM picks: {result}")
        if isinstance(result, dict):
            out = {}
            for k, v in result.items():
                if self._is_no_match(v):
                    out[k] = None
                elif isinstance(v, dict) and "value" in v and not self._is_no_match(v["value"]):
                    # Guard the inner value too, not just the wrapper: the LLM
                    # sometimes answers "no match" as {"value": "None"} — a dict
                    # whose value is the literal *string* "None". Taken at face
                    # value that gets offered to the user as a real substitution
                    # ("closest match is 'None'"), which is nonsense.
                    out[k] = {"value": v["value"], "auto_accept": bool(v.get("auto_accept", False))}
                else:
                    out[k] = None
            return out
        return {item: None for item in items_with_candidates}

    @staticmethod
    def _is_no_match(value) -> bool:
        """True for every way the LLM spells "no match" — a real null, or one of
        the null-ish strings it produces instead."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in ("", "none", "null", "n/a", "na")
        return False

    def _build_alternatives_text(self, pending: dict) -> str:
        """Show the other Neo4j candidates for each unconfirmed node, so the user
        can pick a different one instead of the top suggestion."""
        candidates = pending.get("candidates", {})
        unconfirmed = pending.get("unconfirmed", [])
        lines = []
        for u in unconfirmed:
            original = u["original"]
            all_cands = candidates.get(original, [])
            # Skip the already-suggested top hit; show the rest
            suggestion = u["suggestion"]
            others = [(v, s) for v, s in all_cands if v != suggestion]
            if others:
                others_str = ", ".join(f"**{v}** ({round(s*100)}% similar)" for v, s in others[:4])
                lines.append(
                    f"Other candidates for **'{original}'** in the database: {others_str}.\n"
                    f"The closest remains **'{suggestion}'**. "
                    f"Reply with the name you'd like to use, or say **yes** to use '{suggestion}', "
                    f"or **no** to build without it."
                )
            else:
                lines.append(
                    f"There are no other similar entries for **'{original}'** in the database. "
                    f"The only close match is **'{suggestion}'**. "
                    f"Say **yes** to use it or **no** to build without it."
                )
        return "\n\n".join(lines)

    def _alternative_candidate_values(self, pending: dict) -> list:
        candidates = pending.get("candidates", {})
        unconfirmed = pending.get("unconfirmed", [])
        seen = []
        for u in unconfirmed:
            all_cands = candidates.get(u["original"], [])
            for value, _score in all_cands[:4]:
                if value not in seen:
                    seen.append(value)
        return seen

    def _classify_confirmation(self, message: str) -> str:
        """Uses the LLM to understand what the user meant — no keyword matching."""
        prompt = CONFIRMATION_CLASSIFICATION_PROMPT.format(message=message)
        try:
            result = self.llm.generate(prompt)
            verdict = result.strip().lower().split()[0] if result else "new_query"
            return verdict if verdict in ("confirm", "reject", "show_alternatives", "new_query") else "new_query"
        except Exception:
            return "new_query"

    def _apply_pending_substitutions(self, pending_json: dict, apply: bool = True, override_value: str = None) -> dict:
        result = copy.deepcopy(pending_json)
        for node in result.get("nodes", []):
            if not (node.get("needs_confirmation") and node.get("pending_substitutions")):
                continue
            if apply:
                subs = node["pending_substitutions"]
                if override_value:
                    subs = {orig: override_value for orig in subs}

                # Handle id-field substitution: move result to the correct property
                node_id_val = node.get("id", "")
                if node_id_val and node_id_val in subs:
                    suggested = subs[node_id_val]
                    id_prop = self._node_id_property.get(node.get("type", "").lower())
                    if id_prop:
                        node.setdefault("properties", {})[id_prop] = suggested
                    node["id"] = ""

                # Handle property-value substitutions
                for prop_key, prop_val in node.get("properties", {}).items():
                    if not isinstance(prop_val, str):
                        continue
                    parts = [p.strip() for p in prop_val.split(",") if p.strip()]
                    replaced = set()
                    new_parts = []
                    for part in parts:
                        replacement = next(
                            (sugg for orig, sugg in subs.items() if part.lower() == orig.lower()),
                            None,
                        )
                        if replacement:
                            new_parts.append(replacement)
                            replaced.add(part.lower())
                        else:
                            new_parts.append(part)
                    existing_lower = {p.lower() for p in new_parts}
                    for orig, sugg in subs.items():
                        if orig.lower() not in replaced and sugg.lower() not in existing_lower:
                            new_parts.append(sugg)
                            existing_lower.add(sugg.lower())
                    node["properties"][prop_key] = ", ".join(new_parts)

            if apply:
                node["status"] = True
                for key in ("needs_confirmation", "pending_substitutions", "all_list_values",
                            "not_validated", "validation_error"):
                    node.pop(key, None)
            else:
                originals = ", ".join(f"'{o}'" for o in (node.get("pending_substitutions") or {}))
                node["status"] = False
                node["validation_error"] = (
                    f"{originals} not found in the database; the suggested match was declined."
                    if originals else "Not found in the database; the suggested match was declined."
                )
                for key in ("needs_confirmation", "pending_substitutions", "all_list_values"):
                    node.pop(key, None)

        # Deduplicate nodes that ended up with identical type + properties after substitution
        result["nodes"] = self._deduplicate_nodes(result.get("nodes", []), result.get("predicates", []))
        return result

    def finalize_annotation_json(self, validated_json: dict):
        if not isinstance(validated_json, dict):
            return validated_json, []

        cleaned = copy.deepcopy(validated_json)
        nodes = cleaned.get("nodes", []) or []

        unresolved_ids = {n.get("node_id") for n in nodes if n.get("status") is False}
        unresolved = [
            {
                "node_id": n.get("node_id"),
                "type": n.get("type"),
                "value": (n.get("properties") or {}).get(
                    self._node_id_property.get((n.get("type") or "").lower(), ""), ""
                ) or n.get("id") or next(iter((n.get("properties") or {}).values()), ""),
                "reason": n.get("validation_error") or "Not found in the database.",
            }
            for n in nodes if n.get("status") is False
        ]

        if unresolved_ids:
            cleaned["predicates"] = [
                p for p in (cleaned.get("predicates", []) or [])
                if p.get("source") not in unresolved_ids and p.get("target") not in unresolved_ids
            ]

        return cleaned, unresolved

    def _deduplicate_nodes(self, nodes: list, predicates: list) -> list:
        """Remove nodes whose type+properties are exact duplicates of an earlier node.
        Predicates that reference a removed duplicate are remapped to the surviving node.
        """
        seen = {}       # (type, frozenset(properties.items())) -> surviving node_id
        removed = {}    # removed node_id -> surviving node_id
        kept = []
        for node in nodes:
            key = (node.get("type", ""), frozenset(
                (k, v) for k, v in node.get("properties", {}).items()
            ))
            if key in seen:
                removed[node["node_id"]] = seen[key]
            else:
                seen[key] = node["node_id"]
                kept.append(node)

        # Remap predicates
        if removed:
            for pred in predicates:
                if pred.get("source") in removed:
                    pred["source"] = removed[pred["source"]]
                if pred.get("target") in removed:
                    pred["target"] = removed[pred["target"]]

        return kept

    def _describe_annotation_result(self, query: str, validated_json: dict) -> str:
        """Generate a meaningful biological description from the query + validated JSON.
        Replaces the generic 'structure created successfully' text so the aggregator
        has real content to work with instead of hallucinating.
        """
        try:
            nodes = validated_json.get("nodes", [])
            predicates = validated_json.get("predicates", [])

            node_id_to_label = {}
            node_lines = []
            failed_nodes = []
            for n in nodes:
                nid  = n.get("node_id", "")
                ntype = n.get("type", "unknown")
                props = n.get("properties", {})
                prop_str = ", ".join(f"{k}: {v}" for k, v in props.items()) if props else "(no properties)"
                if n.get("status") is False:
                    reason = n.get("validation_error") or (
                        f"could not be matched in the database: {n.get('not_validated')}"
                        if n.get("not_validated") else "not found in the database"
                    )
                    failed_nodes.append(f"- {ntype} [{nid}]: {prop_str} — NOT FOUND: {reason}")
                else:
                    node_lines.append(f"- {ntype} [{nid}]: {prop_str}")
                node_id_to_label[nid] = f"{ntype}({prop_str})"

            pred_lines = []
            for p in predicates:
                src = node_id_to_label.get(p.get("source", ""), p.get("source", ""))
                tgt = node_id_to_label.get(p.get("target", ""), p.get("target", ""))
                pred_lines.append(f"- {src} --[{p.get('type', '')}]--> {tgt}")

            structure_summary = "Nodes found in the database:\n" + ("\n".join(node_lines) or "- (none)")
            if failed_nodes:
                structure_summary += "\n\nEntities NOT found in the database:\n" + "\n".join(failed_nodes)
            if pred_lines:
                structure_summary += "\n\nRelationships:\n" + "\n".join(pred_lines)

            failure_instruction = (
                " IMPORTANT: some requested entities were NOT found in the database (listed above). "
                "Say so plainly and name them — do NOT describe this as successful or complete, and do "
                "not imply those entities exist or were annotated."
                if failed_nodes else ""
            )
            prompt = DESCRIBE_ANNOTATION_PROMPT.format(
                query=query,
                structure_summary=structure_summary,
                failure_instruction=failure_instruction,
            )
            result = self.llm.generate(prompt)
            if result and result.strip():
                return result.strip()
            return self._fallback_annotation_text(failed_nodes)
        except Exception as e:
            logger.warning(f"Failed to generate annotation description: {e}")
            return self._fallback_annotation_text(locals().get("failed_nodes") or [])

    @staticmethod
    def _fallback_annotation_text(failed_nodes: list) -> str:
        if failed_nodes:
            return (
                "Some of the entities you asked about could not be found in the database, "
                "so the annotation structure is incomplete."
            )
        return "The annotation structure was built successfully."

    def _build_confirmation_text(self, unconfirmed_nodes: list) -> str:
        if len(unconfirmed_nodes) == 1:
            u = unconfirmed_nodes[0]
            all_vals = u.get("all_list_values") or []
            known_vals = [v for v in all_vals if v != u["original"]]

            base = (
                f"I couldn't find **'{u['original']}'** in the database. "
                f"The closest match I found is **'{u['suggestion']}'**.\n\n"
                f"Should I go ahead and use **'{u['suggestion']}'** in place of **'{u['original']}'**?"
            )
            if known_vals:
                base += f" Or would you like me to build the annotation without it, using only {known_vals}?"
            else:
                base += " Or would you like to cancel and try a different identifier?"
            return base

        lines = ["I couldn't find some of the nodes you mentioned in the database:"]
        for u in unconfirmed_nodes:
            lines.append(
                f"  - **'{u['original']}'** — closest match is **'{u['suggestion']}'**"
            )
        lines.append(
            "\nShould I go ahead with these substitutions? "
            "Or would you prefer I build the annotation skipping the unrecognised nodes?"
        )
        return "\n".join(lines)


    def process_annotation_query(
        self, query, user_id, query_type="annotation_biological"
    ):
        # orchestrate the entire annotation pipeline from user query to final response
        try:
            logger.info(
                f"Starting annotation pipeline for query: '{query}', type: {query_type}"
            )

            return self._handle_biological_query(query, user_id)

        except Exception as e:
            error_msg = f"Unexpected error in annotation pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "json_extraction": "unknown",
                    "cypher_conversion": "unknown",
                    "database_execution": "unknown",
                    "summarization": "unknown",
                },
            }

    def _organism_context(self, query):
        """Pick the schema and Neo4j connection matching the query's organism."""
        if self._detect_organism(query) == "fly" and self.fly_schema_handler:
            logger.info("Organism detected: fly — using fly schema and Neo4j")
            return ("fly", self.fly_enhanced_schema, self.fly_schema_handler,
                    self.fly_neo4j or self.neo4j)
        logger.info("Organism detected: human — using human schema and Neo4j")
        return "human", self.enhanced_schema, self.schema_handler, self.neo4j

    @staticmethod
    def _collect_unconfirmed_nodes(updated_json):
        """Nodes whose values were guessed and need the user to confirm them."""
        unconfirmed = []
        for node in updated_json.get("nodes", []):
            if not (node.get("needs_confirmation") and node.get("pending_substitutions")):
                continue
            for original, suggestion in node["pending_substitutions"].items():
                unconfirmed.append({
                    "node_id": node.get("node_id"),
                    "node_type": node.get("type"),
                    "original": original,
                    "suggestion": suggestion,
                    "all_list_values": node.get("all_list_values", []),
                })
        return unconfirmed

    def _confirmation_needed(self, validation, unconfirmed, organism):
        """Pause the pipeline and hand the caller everything needed to resume."""
        logger.info(f"Returning needs_confirmation for {len(unconfirmed)} node(s)")
        return {
            "success": True,
            "needs_confirmation": True,
            "confirmation_text": self._build_confirmation_text(unconfirmed),
            "pending": {
                "json": validation["updated_json"],
                "candidates": validation.get("candidates", {}),
                "unconfirmed": unconfirmed,
                "organism": organism,
            },
            "summary": None,
            "json_format": None,
            "validation_report": validation["validation_report"],
            "resource": {"id": None, "type": "annotation"},
        }

    def _annotation_result(self, query, validation, organism):
        """A fully validated annotation, described for the user."""
        result = {
            "success": True,
            "summary": self._describe_annotation_result(query, validation["updated_json"]),
            "json_format": validation["updated_json"],
            "organism": organism,
            "validation_report": validation["validation_report"],
            "resource": {"id": None, "type": "annotation"},
        }
        logger.info("JSON query extraction successful")
        logger.info(f"JSON query structure: {json.dumps(result, indent=2)}")
        return result

    def _handle_biological_query(self, query, user_id):
        """Turn a natural-language query into a validated annotation JSON.

        Stops early and asks the user when a value could only be guessed;
        otherwise returns the finished annotation. This builds the annotation
        JSON only -- nothing here runs a query against the database.
        """
        try:
            organism, schema, schema_handler, neo4j = self._organism_context(query)

            emit_to_user(user=user_id,
                         message="Extracting relevant information from your query...")
            try:
                relevant_information = self._extract_relevant_information(
                    query, enhanced_schema=schema
                )
                logger.info("Relevant information extraction successful")

                emit_to_user(user=user_id, message="Validating Constructed Json Format...")
                initial_json = self._convert_to_annotation_json(
                    relevant_information, query, enhanced_schema=schema
                )
                logger.info("Initial JSON conversion successful")

                validation = self._validate_and_update(
                    initial_json, neo4j=neo4j, schema_handler=schema_handler
                )
                logger.info("JSON validation successful")

                unconfirmed = self._collect_unconfirmed_nodes(validation["updated_json"])
                if unconfirmed:
                    return self._confirmation_needed(validation, unconfirmed, organism)

                return self._annotation_result(query, validation, organism)

            except Exception as e:
                logger.error(f"Failed to extract JSON query: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to process query: {str(e)}",
                    "pipeline_status": {"json_extraction": "failed"},
                }

        except Exception as e:
            error_msg = f"Unexpected error in biological query pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "pipeline_status": {
                    "json_extraction": "unknown",
                    "cypher_conversion": "unknown",
                    "database_execution": "unknown",
                    "summarization": "unknown",
                },
            }


