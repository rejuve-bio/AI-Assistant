import os
import re
from pprint import pprint
from rapidfuzz import process
from bioblend.galaxy import GalaxyInstance
from dotenv import load_dotenv
import json
import time
from datetime import datetime, timedelta

from sys import path
path.append('app')

from llm_handle.llm_models import GeminiModel
from galaxy_prompts import TOOL_PROMPT, WORKFLOW_PROMPT, DATASET_PROMPT

class GalaxyInformer:
    def __init__(self, entity_type):
        load_dotenv()
        self.entity_type = entity_type.lower()
        self.timestamp_file = f'app/galaxy/cache_files/timestamps/{self.entity_type}_timestamp.txt'
        self.gi = GalaxyInstance(url=os.getenv("GALAXY_URL"), key=os.getenv("GALAXY_API"))
        self.llm = GeminiModel(api_key=os.getenv("GEMINI_API_KEY"), model_provider='gemini', model_name=os.getenv("ADVANCED_LLM_VERSION") )
        self._entity_config = {
            'dataset': {
                'get_method': self._get_datasets,
                'search_fields': ['name', 'full_path'],
                'summary_prompt': DATASET_PROMPT
            },
            'tool': {
                'get_method': self._get_tools,
                'search_fields': ['name'],
                'summary_prompt': TOOL_PROMPT
            },
            'workflow': {
                'get_method': self._get_workflows,
                'search_fields': ['name'],
                'summary_prompt': WORKFLOW_PROMPT
            }
        }

    def extract_filename(self, path):
        """Extracts filename from full path (dataset specific)"""
        match = re.search(r'([^/]+)$', path)
        return match.group(1) if match else path

    def _get_datasets(self):
        """Combined dataset retrieval from libraries and histories"""
        dataset_list = []
        
        # Library datasets
        libraries = self.gi.libraries.get_libraries()
        for library in libraries:
            library_details = self.gi.libraries.show_library(library['id'], contents=True)
            for lib in library_details:
                if lib['type'] == 'file':
                    dataset_list.append({
                        "id": lib['id'],
                        "name": self.extract_filename(lib['name']),
                        "full_path": lib['name'],
                        "type": lib["type"],
                        "source": "library"
                    })
        
        # History datasets
        for data in self.gi.datasets.get_datasets():
            if data['type'] == 'file':
                dataset_list.append({
                    "id": data['id'],
                    "name": data['name'],
                    "full_path": data['url'],
                    "type": data['type'],
                    "source": "history"
                })
        
        return dataset_list

    def _get_tools(self):
        return [{
            'description': tool['description'],
            'id': tool['id'],
            'name': tool['name']
        } for tool in self.gi.tools.get_tools()]

    def _get_workflows(self):
        return [{
            'model_class': wf['model_class'],
            'owner': wf['owner'],
            'id': wf['id'],
            'name': wf['name']
        } for wf in self.gi.workflows.get_workflows(published=True)]

    def get_entities(self):
        """Get all entities based on configured type"""
        return self._entity_config[self.entity_type]['get_method']()
    
    # Save the current timestamp to a file
    def save_timestamp(self):
        with open(self.timestamp_file, "w") as f:
            f.write(str(time.time()))

    # Read the timestamp from file and compare with current time
    def within_time(self):
        try:
            with open(self.timestamp_file, "r") as f:
                saved_time = float(f.read().strip())
        except FileNotFoundError:
            return False  # If file doesn't exist, assume False
        
        saved_datetime = datetime.fromtimestamp(saved_time)
        current_datetime = datetime.now()
        
        if current_datetime - saved_datetime > timedelta(hours=10):
            return False
        return True
        
    def search_entities(self, query, threshold=85):
        """Unified fuzzy search with priority fields"""
        if self.timestamp_file:
            if self.within_time():
                    try:
                        with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'r') as f:
                            entities = json.load(f)
                    except FileNotFoundError:
                        self.save_timestamp()
                        entities=self.get_entities()
                        with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'w') as f:
                            json.dump(entities, f, indent=4)
            else:
                self.save_timestamp()
                entities = self.get_entities()
                with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'w') as f:
                    json.dump(entities, f, indent=4)

        config = self._entity_config[self.entity_type]
        matches = []


        # Priority search on configured fields
        for entity in entities:
            for field in config['search_fields']:
                if field in entity and isinstance(entity[field], str):
                    score = process.extractOne(query, [entity[field]])
                    if score and score[1] >= threshold:
                        matches.append((entity, score[1]))
                        break  # Only match once per entity

        if matches:
            return sorted(matches, key=lambda x: x[1], reverse=True)[0]

        # Fallback search across all fields
        for entity in entities:
            for key, value in entity.items():
                if key not in config['search_fields'] and isinstance(value, str):
                    score = process.extractOne(query, [value])
                    if score and score[1] >= threshold:
                        matches.append((entity, score[1]))

                    pprint(score)

        return sorted(matches, key=lambda x: x[1], reverse=True)[0] if matches else None

    def get_entity_info(self, search_query, entity_id=None):
        """Unified info retrieval with LLM summary"""
        
        if not os.path.exists(f'app/galaxy/cache_files'):
             os.makedirs('app/galaxy/cache_files')
        if not os.path.exists('app/galaxy/cache_files/timestamps'):
             os.makedirs('app/galaxy/cache_files/timestamps')
        if entity_id:
            entity = next((e for e in self.get_entities() if e['id'] == entity_id), None)
        else:
            result = self.search_entities(search_query)
            entity = result[0] if result else None

        if not entity:
            return f"No {self.entity_type} found"

        # Get detailed information
        detail_methods = {
            'dataset': lambda id: self.gi.datasets.show_dataset(id),
            'tool': lambda id: self.gi.tools.show_tool(id, io_details=True),
            'workflow': lambda id: self.gi.workflows.show_workflow(id)
        }
        
        details = detail_methods[self.entity_type](entity['id'])       
        response = self.llm.generate(prompt= self._entity_config[self.entity_type]['summary_prompt'].format(input= details))
        print(type(response))
        if isinstance(response, str):
            try:
                response= json.loads(response)
            except json.JSONDecodeError:
                print("Failed to decode JSON response")
                try:
                    return dict(response)
                except:
                    return response
        return response 