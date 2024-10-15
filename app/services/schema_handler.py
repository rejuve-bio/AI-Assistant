from collections import defaultdict
import json
from biocypher import BioCypher
from flask import jsonify
import yaml

class SchemaHandler:
    def __init__(self, schema_config_path, biocypher_config_path):
        self.bcy = BioCypher(schema_config_path=schema_config_path, biocypher_config_path=biocypher_config_path)
        self.schema = self.bcy._get_ontology_mapping()._extend_schema()
        self.processed_schema = self.process_schema(self.schema) 
        self.parent_nodes = self.get_parent_nodes()
        self.parent_edges = self.get_parent_edges()
        self.adj_list = self.get_adjacency_list()
        self.schema_graph = self.build_graph(self.adj_list)

    def process_schema(self, schema):
        process_schema = {}
        for _, value in schema.items():
            input_label = value.get("input_label")
            output_label = value.get("output_label")
            source = value.get("source")
            target = value.get("target")

            label = output_label if output_label else input_label
            if isinstance(label, list):
                for i_label in label:
                    key_label = f'{source}-{i_label}-{target}' if source and target else i_label
                    process_schema[key_label] = {**value, "key": key_label}
            else:
                key_label = f'{source}-{label}-{target}' if source and target else label
                process_schema[key_label] = {**value, "key": key_label}

        return process_schema
    
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
                currNode = {
                    'type': key,
                    'is_a': parent,
                    'label': value['input_label'],
                    'properties': value.get('properties', {})
                }
                if parent not in nodes:
                    nodes[parent] = []
                nodes[parent].append(currNode)

        return [{'child_nodes': nodes[key], 'parent_node': key} for key in nodes]

    def get_edges(self):
        edges = {}
        for key, value in self.processed_schema.items():
            if value['represented_as'] == 'edge':
                if key in self.parent_edges:
                    continue
                label = value.get('output_lable', value['input_label'])
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
                        label = value.get('output_lable', value['input_label'])
                        relation = {
                            'type': key,
                            'label': label,
                            'source': value.get('source', ''),
                            'target': value.get('target', '')
                        }
                        relations.append(relation)
        return relations

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
    
    def get_adjacency_list(self):
        adj_list = {}
        for k, v in self.schema.items():
            if "." in k:
                continue
            if v.get("represented_as") == "edge" and \
                "source" in v and "target" in v:
                source = v.get("source")
                target = v.get("target")
                label = v.get("input_label")
                output_label = v.get("output_label")
                if output_label:
                    label = output_label
                if not source or not target or not label:
                    continue
                if isinstance(source, str):
                    source = [source]
                if isinstance(target, str):
                    target = [target]
                for s in source:
                    s = s.replace(" ", "_")
                    if s in self.parent_nodes:
                        continue
                    if s not in adj_list:
                        adj_list[s] = {}
                    if label not in adj_list[s]:
                        adj_list[s][label] = []
                    for t in target:
                        t = t.replace(" ", "_")
                        if t in self.parent_nodes:
                            continue
                        if t not in adj_list[s][label]:
                            adj_list[s][label].append(t)

        return adj_list
    
    def build_graph(self, schema_relationship):
        # schema_relationship = generate_schema_relationship()

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

                



