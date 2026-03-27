from collections import Counter, defaultdict
import re
import traceback
import json
import tiktoken
import logging
import os
import requests
from dotenv import load_dotenv
from app.prompts.summarizer_prompts import (
    GRAPH_BIOLOGICAL_INSIGHT_PROMPT,
    GRAPH_ID_BIOLOGICAL_QA_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_PROMPT_BASED_ON_USER_QUERY,
    SUMMARY_PROMPT_CHUNKING,
    SUMMARY_PROMPT_CHUNKING_USER_QUERY,
)
from app.storage.redis import redis_manager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class Graph_Summarizer:
    """
    Handles graph-related operations like processing nodes, edges, generating responses ...
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.llm = llm
        self.description = []
        if self.llm.__class__.__name__ == "GeminiModel":
            self.max_token = 2000
        elif self.llm.__class__.__name__ == "OpenAIModel":
            self.max_token = 100000
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.kg_service_url = os.getenv('ANNOTATION_SERVICE_URL')

    def clean_and_format_response(self, desc):
        desc = desc.strip()
        desc = re.sub(r"\n\s*\n", "\n", desc)
        desc = re.sub(r"^\s*[\*\-]\s*", "", desc, flags=re.MULTILINE)
        lines = desc.split("\n")

        formatted_lines = []
        for line in lines:
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", line)
            for sentence in sentences:
                formatted_lines.append(sentence + "\n")
        formatted_desc = " ".join(formatted_lines).strip()
        return formatted_desc

    def group_edges_by_source(self, edges):
        """Group edges by source_node."""
        grouped_edges = defaultdict(list)
        for edge in edges:
            source_node_id = edge["source"].split(" ")[-1]  # Extract ID
            grouped_edges[source_node_id].append(edge)
        return grouped_edges

    def generate_node_description(self, node):
        """Generate a description for a node with available attributes."""
        desc_parts = []

        for key, value in node.items():
            # Attempt to parse JSON-like strings into lists
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, list):
                        # Limit to top 3 items
                        top_items = parsed_value[:3]
                        if top_items:
                            desc_parts.append(
                                f"{key.capitalize()}: {', '.join(top_items)}"
                            )
                        continue  # Move to the next attribute after processing
                except json.JSONDecodeError:
                    pass  # If not a JSON string, treat it as a regular string

            # For non-list attributes, simply add them to the description
            desc_parts.append(f"{key.capitalize()}: {value}")
        return " | ".join(desc_parts)

    def generate_grouped_descriptions(self, edges, nodes, batch_size=50):
        grouped_edges = self.group_edges_by_source(edges)
        descriptions = []

        # Process each source node and its related target nodes
        for source_node_id, related_edges in grouped_edges.items():
            source_node = nodes.get(source_node_id, {})
            source_desc = self.generate_node_description(source_node)

            # Collect descriptions for all target nodes linked to this source node
            target_descriptions = []
            for edge in related_edges:
                target_node_id = edge["target"].split(" ")[-1]
                target_node = nodes.get(target_node_id, {})
                target_desc = self.generate_node_description(target_node)

                # Add the relationship and target node description
                label = edge["label"]
                target_descriptions.append(
                    f"{label} -> Target Node ({edge['target']}): {target_desc}"
                )

            # Combine the source node description with all target node descriptions
            source_and_targets = (
                f"Source Node ({source_node_id}): {source_desc}\n"
                + "\n".join(target_descriptions)
            )
            descriptions.append(source_and_targets)

            # If batch processing is required, we can break or yield after each batch
            # if len(descriptions) >= batch_size:
            #   break   Process the next batch in another iteration

        return descriptions

    def nodes_description(self, nodes):
        nodes_descriptions = []
        for source_node_id in nodes:
            source_node = nodes.get(source_node_id, {})
            source_desc = self.generate_node_description(source_node)
            nodes_descriptions.append(source_desc)
        return nodes_descriptions

    def num_tokens_from_string(self, encoding_name: str):
        """Calculates the number of tokens in each description and groups them into batches under a token limit."""
        encoding = tiktoken.get_encoding(encoding_name)
        accumulated_tokens = 0
        grouped_batched_descriptions = []
        self.current_batch = []
        for i, desc in enumerate(self.description):
            desc_tokens = len(encoding.encode(desc))
            if accumulated_tokens + desc_tokens <= self.max_token:
                self.current_batch.append(desc)
                accumulated_tokens += desc_tokens
            else:
                grouped_batched_descriptions.append(self.current_batch)
                self.current_batch = [desc]
                accumulated_tokens = desc_tokens
        if self.current_batch:
            grouped_batched_descriptions.append(self.current_batch)
        return grouped_batched_descriptions

    @staticmethod
    def _norm_text(value):
        if value is None:
            return "unknown"
        text = str(value).strip().lower()
        return text if text else "unknown"

    @staticmethod
    def _node_data(node):
        if isinstance(node, dict):
            payload = node.get("data", node)
            return payload if isinstance(payload, dict) else {}
        return {}

    @staticmethod
    def _edge_data(edge):
        if isinstance(edge, dict):
            payload = edge.get("data", edge)
            return payload if isinstance(payload, dict) else {}
        return {}

    @staticmethod
    def _category_name(raw_type):
        aliases = {
            "gene": "gene",
            "transcript": "transcript",
            "mrna": "transcript",
            "rna": "transcript",
            "protein": "protein",
            "polypeptide": "protein",
            "variant": "variant",
            "snp": "variant",
            "exon": "exon",
        }
        return aliases.get(Graph_Summarizer._norm_text(raw_type), Graph_Summarizer._norm_text(raw_type))

    @staticmethod
    def _sample_ids(ids, max_ids=3):
        out = []
        seen = set()
        for item in ids:
            sid = str(item).strip()
            if sid and sid not in seen:
                out.append(sid)
                seen.add(sid)
            if len(out) >= max_ids:
                break
        return out

    @staticmethod
    def _top_items(counter_like, top_n=5):
        return sorted(counter_like.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _normalize_graph_payload(self, raw_graph):
        if raw_graph is None:
            return {"nodes": [], "edges": []}

        if isinstance(raw_graph, str):
            try:
                raw_graph = json.loads(raw_graph)
            except json.JSONDecodeError:
                raise ValueError("graph must be a dict or a valid JSON string")

        if not isinstance(raw_graph, dict):
            raise ValueError("graph must be a dict")

        payload = raw_graph.get("graph", raw_graph)
        if not isinstance(payload, dict):
            payload = raw_graph

        nodes = payload.get("nodes", []) or []
        edges = payload.get("edges", []) or []
        return {"nodes": nodes, "edges": edges}

    def parse_graph_by_categories(self, raw_graph):
        payload = self._normalize_graph_payload(raw_graph)
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])

        node_category_by_id = {}
        node_ids_by_category = defaultdict(list)
        subtype_breakdown = defaultdict(Counter)

        for node in nodes:
            d = self._node_data(node)
            nid = str(d.get("id") or "").strip()
            if not nid:
                continue

            category = self._category_name(d.get("type") or d.get("label") or "unknown")
            node_category_by_id[nid] = category
            node_ids_by_category[category].append(nid)

            for key, value in d.items():
                key_norm = self._norm_text(key)
                if key_norm.endswith("_type") or "biotype" in key_norm:
                    subtype_breakdown[f"{category}.{key_norm}"][self._norm_text(value)] += 1

        relationship_counts = Counter()
        motif_counts = Counter()
        outgoing_by_category = defaultdict(Counter)

        for edge in edges:
            d = self._edge_data(edge)
            src = str(d.get("source") or "").strip()
            tgt = str(d.get("target") or "").strip()
            rel = self._norm_text(d.get("label") or d.get("type") or d.get("relationship") or "related_to")

            src_cat = node_category_by_id.get(src, "unknown")
            tgt_cat = node_category_by_id.get(tgt, "unknown")

            relationship_counts[rel] += 1
            motif_counts[(src_cat, rel, tgt_cat)] += 1
            outgoing_by_category[src_cat][rel] += 1

        category_profiles = {}
        for category in sorted(node_ids_by_category.keys()):
            category_profiles[category] = {
                "count": len(node_ids_by_category[category]),
                "sample_ids": self._sample_ids(node_ids_by_category[category], max_ids=3),
                "top_outgoing_relationships": self._top_items(outgoing_by_category[category], top_n=5),
            }

        top_motifs = [
            {
                "source_category": src,
                "relationship": rel,
                "target_category": tgt,
                "support": support,
            }
            for (src, rel, tgt), support in motif_counts.most_common(12)
        ]

        return {
            "categories": sorted(node_ids_by_category.keys()),
            "category_profiles": category_profiles,
            "top_relationship_labels": [name for name, _ in relationship_counts.most_common(10)],
            "top_motifs": top_motifs,
            "subtype_breakdown": {
                key: self._top_items(value, top_n=8) for key, value in subtype_breakdown.items()
            },
            "exploration_starts": self._sample_ids(
                node_ids_by_category.get("transcript", [])
                or node_ids_by_category.get("gene", [])
                or sum(
                    [
                        ids
                        for _, ids in sorted(
                            node_ids_by_category.items(),
                            key=lambda x: len(x[1]),
                            reverse=True,
                        )
                    ],
                    [],
                ),
                max_ids=3,
            ),
        }

    def generate_biological_insight_from_graph(self, graph, user_query=None):
        parsed_graph = self.parse_graph_by_categories(graph)
        prompt = GRAPH_BIOLOGICAL_INSIGHT_PROMPT.format(
            user_query=user_query or "Explain this graph biologically for a researcher.",
            parsed_graph_json=json.dumps(parsed_graph, indent=2),
        )
        text = self.llm.generate(prompt)
        return {
            "text": text,
            "summary": text,
            "parsed_graph": parsed_graph,
            "method": "direct_graph_biological_insight",
        }

    def graph_description(self, graph, limited_nodes=100):
        if not graph:
            self.descriptions = "no graph is returned"
            return self.descriptions

        limited_node_ids = set()
        if isinstance(graph, dict) and "nodes" in graph:
            if len(graph["nodes"]):
                for i in range(min(limited_nodes, len(graph["nodes"]))):
                    limited_node_ids.add(graph["nodes"][i]["data"]["id"])

                limited_nodes_data = [
                    node
                    for node in graph["nodes"]
                    if node["data"]["id"] in limited_node_ids
                ]
                limited_edges_data = []
                for edge in graph["edges"]:
                    if (
                        edge["data"]["source"] in limited_node_ids
                        and edge["data"]["target"] in limited_node_ids
                    ):
                        limited_edges_data.append(edge)

                limited_graph = {
                    "nodes": limited_nodes_data,
                    "edges": limited_edges_data,
                }
                nodes = {
                    node["data"]["id"]: node["data"] for node in limited_graph["nodes"]
                }

                if len(limited_graph["edges"]) > 0:
                    edges = [
                        {
                            "source": edge["data"]["source"],
                            "target": edge["data"]["target"],
                            "label": edge["data"]["label"],
                        }
                        for edge in limited_graph["edges"]
                    ]

                self.description = self.generate_grouped_descriptions(
                    edges, nodes, batch_size=10
                )
                self.descriptions = self.num_tokens_from_string("cl100k_base")
            else:
                self.descriptions = []

            return self.descriptions

    # def annotate_by_id(self,graph_id, token, query=None):
    #     try:
    #         cached_graph = redis_manager.get_graph_by_id(graph_id)
    #         if cached_graph and cached_graph.get("graph_summary"):
    #             logger.info(f"Cache hit for graph_id={graph_id} {cached_graph}")
    #             summary = cached_graph["graph_summary"]
    #             return summary

    #         params = {"source": "ai-assistant"}
    #         logger.info("querying the graph without user question...")
    #         response = requests.get(
    #             self.kg_service_url + "/annotation/" + graph_id,
    #             headers={"Authorization": token},
    #         )
            
    #         response.raise_for_status()
    #         json_response = response.json()
            
    #         if isinstance(json_response, str):
    #             try:
    #                 json_response = json.loads(json_response)
    #             except json.JSONDecodeError:
    #                 json_response = {"answer": json_response}

    #         if not isinstance(json_response, dict):
    #             json_response = {"answer": str(json_response)}
            
    #         response = {}
    #         if json_response.get("summary"):
    #             response["summary"] = json_response.get("summary")
    #         else:
    #             response["text"] = "Graph is too big, No summaries provided to answer your question"
    #         summary = json_response.get("summary")
    #         if summary:
    #             redis_manager.create_graph(graph_id=graph_id, graph_summary=summary)
    #             if query:
    #                 response_data = self.llm.generate(f"Please answer the question based on the summary: {summary}.\nQuestion: {query}\nAnswer:")
    #                 return {"summary": summary, "text": response_data}
    #             else:
    #                 return {"summary": summary}
    #         else:
    #             return {"text": "Graph is too big, No summaries provided to answer your question"}
    #     except Exception as e:
    #         logger.error(f"Error in annotate_by_id: {e}")
    #         return {"text": "Graph is too big, No summaries provided to answer your question"}
    def annotate_by_id(self, graph_id, token, query=None):
        try:
            cached_graph = redis_manager.get_graph_by_id(graph_id)
            if cached_graph and cached_graph.get("graph_summary"):
                summary = json.loads(cached_graph["graph_summary"])  # Convert JSON string back to dict
                logger.info(f'Cache hit for graph_id={graph_id} is graph summary of {summary}')
                return {"summary": cached_graph["graph_summary"], "text": None}

            logger.info("Querying the graph without user question...")
            http_response = requests.get(
                f"{self.kg_service_url}/annotation/{graph_id}",
                headers={"Authorization": token},
            )
            http_response.raise_for_status()

            json_response = http_response.json()
            if isinstance(json_response, str):
                try:
                    json_response = json.loads(json_response)
                except json.JSONDecodeError:
                    json_response = {"summary": None, "answer": json_response}

            if not isinstance(json_response, dict):
                json_response = {"summary": None, "answer": str(json_response)}

            summary = json_response.get("summary")
            node_count = json_response.get("node_count", 0)
            edge_count = json_response.get("edge_count", 0)
            node_count_by_label = json_response.get("node_count_by_label", {})
            edge_count_by_label = json_response.get("edge_count_by_label", {})
            
            if summary:
                enhanced_summary = {
                    "summary": summary,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "node_count_by_label": node_count_by_label,
                    "edge_count_by_label": edge_count_by_label
                }
                
                redis_manager.create_graph(graph_id=graph_id, graph_summary=json.dumps(enhanced_summary))                
                text = None
                if query:
                    text = self.llm.generate(
                        GRAPH_ID_BIOLOGICAL_QA_PROMPT.format(
                            graph_summary=summary,
                            node_count_by_label=json.dumps(node_count_by_label),
                            edge_count_by_label=json.dumps(edge_count_by_label),
                            user_query=query,
                        )
                    )
                return {"summary": enhanced_summary, "text": text}
            else:
                return {"summary": None, "text": "Graph is too big, No summaries provided"}

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error in annotate_by_id: {e}")
            return {"summary": None, "text": f"Failed to contact graph service: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error in annotate_by_id: {e}")
            return {"summary": None, "text": f"Unexpected error: {e}"}
    
    def summary(self, graph=None, user_query=None, graph_id=None, token=None):

        try:
            # send the query and the annotation id for the annotation endpoint for the answer
            if graph_id:
                result = self.annotate_by_id(
                    graph_id=graph_id, query=user_query, token=token
                )
                logger.info(f"summary of the graph {graph_id} is {result}")
                return result

            # Parallel path for direct graph payloads (without graph_id)
            if graph:
                result = self.generate_biological_insight_from_graph(
                    graph=graph,
                    user_query=user_query,
                )
                logger.info("Generated biological insight summary from direct graph payload")
                return result

            if graph:
                graph = self.graph_description(graph)

            prev_summery = []
            for i, batch in enumerate(self.descriptions):
                if prev_summery:
                    if user_query:
                        prompt = SUMMARY_PROMPT_CHUNKING_USER_QUERY.format(
                            description=batch,
                            user_query=user_query,
                            prev_summery=prev_summery,
                        )
                    else:
                        prompt = SUMMARY_PROMPT_CHUNKING.format(
                            description=batch, prev_summery=prev_summery
                        )
                else:
                    if user_query:
                        prompt = SUMMARY_PROMPT_BASED_ON_USER_QUERY.format(
                            description=batch, user_query=user_query
                        )
                        print("prompt", prompt)
                    else:
                        prompt = SUMMARY_PROMPT.format(description=batch)
                        print("prompt", prompt)

                response = self.llm.generate(prompt)
                prev_summery = [response]
                return {"text": prev_summery}
                # cleaned_desc = self.clean_and_format_response(prev_summery)
                # return cleaned_desc
        except Exception:
            traceback.print_exc()
            return {
                "summary": None,
                "text": "Failed to summarize the graph.",
            }
