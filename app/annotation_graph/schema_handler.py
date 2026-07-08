from collections import defaultdict
import json
import logging
from biocypher import BioCypher
from flask import current_app, jsonify
import yaml

logger = logging.getLogger(__name__)

class SchemaHandler:
    def __init__(self, schema_config_path, biocypher_config_path, enhanced_schema_path):
        try:
            self.bcy = BioCypher(schema_config_path=schema_config_path, biocypher_config_path=biocypher_config_path)
            self.graph_file = 'graph.pkl'
            self.enhanced_schema = open(enhanced_schema_path, 'r').read()
            self.schema = self.bcy._get_ontology_mapping()._extend_schema()
            self.processed_schema = self.process_schema(self.schema) 
            self.parent_nodes = self.get_parent_nodes()
            self.adj_list = self.get_adjacency_list()
            self.schema_graph = self.build_graph(self.adj_list)
        except Exception as e:
            logger.error(f"Unable to initialize Schema Handler: {e}")

    def _register_schema_entry(self, result, value, source, target, label):
        key_label = f'{source}-{label}-{target}' if source and target else label
        result[key_label] = {**value, "key": key_label}

    def process_schema(self, schema):
        result = {}
        for _, value in schema.items():
            label = value.get("output_label") or value.get("input_label")
            source = value.get("source")
            target = value.get("target")
            is_a = value.get("is_a")
            if isinstance(is_a, list):
                value = {**value, "is_a": is_a[0] if is_a else None}
            if isinstance(label, list):
                for i_label in label:
                    self._register_schema_entry(result, value, source, target, i_label)
            else:
                self._register_schema_entry(result, value, source, target, label)
        return result
    
    def get_parent_nodes(self):
        parent_nodes = set()
        for _, attributes in self.processed_schema.items():
            if 'represented_as' in attributes and attributes['represented_as'] == 'node' \
                    and 'is_a' in attributes and attributes['is_a'] not in parent_nodes:
                parent_nodes.add(attributes['is_a'])

        return list(parent_nodes)

    def get_parent_edges(self):
        parent_edges = set()
        for _, attributes in self.processed_schema.items():
            if 'represented_as' in attributes and attributes['represented_as'] == 'edge' \
                    and 'is_a' in attributes and attributes['is_a'] not in parent_edges:
                parent_edges.add(attributes['is_a'])

        return list(parent_edges)

    def get_nodes(self):
        nodes = {}
        for key, value in self.processed_schema.items():
            if value['represented_as'] == 'node':
                if key in self.parent_nodes:
                    continue
                parent = value['is_a']
                curr_node = {
                    'type': key,
                    'is_a': parent,
                    'label': value['input_label'],
                    'properties': value.get('properties', {})
                }
                if parent not in nodes:
                    nodes[parent] = []
                nodes[parent].append(curr_node)

        return [{'child_nodes': nodes[key], 'parent_node': key} for key in nodes]

    def get_edges(self):
        edges = {}
        for key, value in self.processed_schema.items():
            if value['represented_as'] == 'edge':
                if key in self.parent_edges:
                    continue
                label = value.get('output_label', value['input_label'])
                edge = {
                    'type': key,
                    'label': label,
                    'is_a': value['is_a'],
                    'source': value.get('source', ''),
                    'target': value.get('target', ''),
                    'properties': value.get('properties', {})
                }
                parent = value['is_a']
                if parent not in edges:
                    edges[parent] = []
                edges[parent].append(edge)
        return [{'child_edges': edges[key], 'parent_edge': key} for key in edges]

    def get_relations_for_node(self, node):
        relations = []
        node_label = node.replace('_', ' ')
        for key, value in self.processed_schema.items():
            if value['represented_as'] == 'edge':
                if 'source' in value and 'target' in value:
                    if value['source'] == node_label or value['target'] == node_label:
                        label = value.get('output_label', value['input_label'])
                        relation = {
                            'type': key,
                            'label': label,
                            'source': value.get('source', ''),
                            'target': value.get('target', '')
                        }
                        relations.append(relation)
        return relations

    @staticmethod
    def get_schema(schema_path):
        with open(schema_path, 'r') as file:
            prime_service = yaml.safe_load(file)

        schema = {}

        for key in prime_service.keys():
            if type(prime_service[key]) == str:
                continue
        
            if any(keys in prime_service[key].keys() for keys in ('source', 'target')):
                schema[key] = {
                    'source': prime_service[key]['source'],
                    'target': prime_service[key]['target']
                }

        return schema  
    
    def _process_edge_entry(self, adj_list, k, v):
        if "." in k:
            return
        if v.get("represented_as") != "edge" or "source" not in v or "target" not in v:
            return
        source = v.get("source")
        target = v.get("target")
        label = v.get("output_label") or v.get("input_label")
        if not source or not target or not label:
            return
        sources = [source] if isinstance(source, str) else source
        targets = [target] if isinstance(target, str) else target
        for s in sources:
            s = s.replace(" ", "_")
            if s in self.parent_nodes:
                continue
            adj_list.setdefault(s, {}).setdefault(label, [])
            for t in targets:
                t = t.replace(" ", "_")
                if t not in self.parent_nodes and t not in adj_list[s][label]:
                    adj_list[s][label].append(t)

    def get_adjacency_list(self):
        adj_list = {}
        for k, v in self.schema.items():
            self._process_edge_entry(adj_list, k, v)
        return adj_list
    
    def build_graph(self, schema_relationship):
        import os
        import pickle
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'rb') as f:
                graph = pickle.load(f)
            logger.debug("Retrieved graph from storage.")
            return graph

        # If no stored graph, build a new one
        graph = defaultdict(list)
        for node, relationships in schema_relationship.items():
            for rel, target in relationships.items():
                if isinstance(target, list):
                    for t in target:
                        graph[node].append((t, rel))
                else:
                    graph[node].append((target, rel))
        # store the graph in file
        return graph
        
                