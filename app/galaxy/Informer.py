import os
import re
from rapidfuzz import process # Rapid fuzz for some reason is not wworking properly?????
from bioblend.galaxy import GalaxyInstance
from dotenv import load_dotenv
import json
import time
from datetime import datetime, timedelta
import numpy as np
import logging

from sys import path
path.append('.')

from app.storage.qdrant import Qdrant
from app.rag.rag import RAG
from app.llm_handle.llm_models import GeminiModel, gemini_embedding_model
from app.galaxy.galaxy_prompts import TOOL_PROMPT, WORKFLOW_PROMPT, DATASET_PROMPT, SELECTION_PROMPT
from app.prompts.rag_prompts import RETRIEVE_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GalaxyInformer:
    def __init__(self, entity_type):
        logger.info(f'initializing the galaxy informer with {entity_type} type')
        load_dotenv()
        self.entity_type = entity_type.lower()
        self.timestamp_file = f'app/galaxy/cache_files/timestamps/{self.entity_type}_timestamp.txt'
        self.gi = GalaxyInstance(url=os.getenv("GALAXY_URL"), key=os.getenv("GALAXY_API"))
        self.llm = GeminiModel(api_key=os.getenv("GEMINI_API_KEY"), model_provider='gemini', model_name=os.getenv("ADVANCED_LLM_VERSION") )
        self.client = Qdrant()
        self.rag = RAG(client=self.client, llm= self.llm)
        self._entity_config = {
            'dataset': {
                'get_method': self._get_datasets,
                'search_fields': ['name'],
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
        logger.info(f'Extracting file name from path.')
        match = re.search(r'([^/]+)$', path)
        return match.group(1) if match else path

    def _get_datasets(self):
        """Combined dataset retrieval from libraries and histories"""
        dataset_list = []
        logger.info(f'Gathering datsets from library')
        # Library datasets
        libraries = self.gi.libraries.get_libraries()
        for library in libraries:
            library_details = self.gi.libraries.show_library(library['id'], contents=True)
            for lib in library_details:
                if lib['type'] == 'file':
                    dataset_list.append({
                        "dataset_id": lib['id'],
                        "name": self.extract_filename(lib['name']),
                        "full_path": lib['name'],
                        "type": lib["type"],
                        "source": "library",
                        "content": f"a library dataset with id '{lib['id']}' named '{self.extract_filename(lib['name'])}' with type '{lib['type']}' and located at '{lib['name']}'"
                    })
        logger.info('Gathering datsets from the history')
        # History datasets
        for data in self.gi.datasets.get_datasets():
            if data['type'] == 'file':
                dataset_list.append({
                    "dataset_id": data['id'],
                    "name": data['name'],
                    "full_path": data['url'],
                    "type": data['type'],
                    "source": "history",
                    "content": f"a library dataset with id '{data['id']}' named '{data['name']}' with type '{data['type']}' and located at '{data['url']}'"
                })
        
        return dataset_list

    def _get_tools(self):
        return [{
            'description': tool['description'],
            'tool_id': tool['id'],
            'name': tool['name'],
            'content': f"tool with id '{tool['id']}' named '{tool['name']}'' with description ''{tool['description']}'"
        } for tool in self.gi.tools.get_tools()]

    def _get_workflows(self):
        return [{
            'description': str(wf.get('annotations', '') or wf.get('description', '') or ''),
            'model_class': wf['model_class'],
            'owner': wf['owner'],
            'workflow_id': wf['id'],
            'name': wf['name'],
            'content': f"A workflow with id '{wf['id']}' owned by '{wf['owner']}' named '{wf['name']}' with description '{str(wf.get('annotations', '') or wf.get('description', '') or '')}'"
        } for wf in self.gi.workflows.get_workflows(published=True)]

    def get_entities(self):
        """Get all entities based on configured type"""
        return self._entity_config[self.entity_type]['get_method']()
    
    # Save the current timestamp to a file
    def save_timestamp(self):
        logger.info('Saving current timestamp')
        with open(self.timestamp_file, "w") as f:
            f.write(str(time.time()))

    # Read the timestamp from file and compare with current time
    def within_time(self):
        try:
            with open(self.timestamp_file, "r") as f:
                saved_time = float(f.read().strip())
        except FileNotFoundError:
            return False  # If file doesn't exist, assume False
        logger.info('checking if the collected data has expired')
        saved_datetime = datetime.fromtimestamp(saved_time)
        current_datetime = datetime.now()
        
        if current_datetime - saved_datetime > timedelta(hours=10):
            return False
        return True
    
    def semantic_search(self, query, collection, user_id):
        """Retrive data from the qdrant collection based on the search query"""

        logger.info('Semantic search for the query')
        if isinstance(query, str):
            query=[query]
        embeddings = gemini_embedding_model(query)
        embed = np.array(embeddings)
        query = embed.reshape(-1, 768).tolist()
        # 768 is the embedding size of a gemini model and for this case we are using a gemini model to do so
        results = self.client.retrieve_data(query=query, collection=collection, user_id=user_id, galaxy=self.entity_type)
        
        return {k: results[k] for k in sorted(results.keys())[:5]}
    
    def fuzzy_search(self, query, entities, config, threshold, matches):
        """"Fuzzy search with priority fields"""
        # Priority search on configured fields
        logger.info('Fuzzy search for the query by name')
        for entity in entities:
            for field in config['search_fields']:
                if field in entity and isinstance(entity[field], str):
                    score = process.extractOne(query, [entity[field]])
                    if score and score[1] >= threshold:
                        matches.append((entity, score[1]))
                        break  # Only match once per entity

        if matches:
            return sorted(matches, key=lambda x: x[1], reverse=True)[:5]

        # Fallback search across all fields
        logger.info('No matches found in the priority fields, searching in all fields as a fallback')
        for entity in entities:
            for key, value in entity.items():
                if key not in config['search_fields'] and isinstance(value, str):
                    score = process.extractOne(query, [value])
                    if score and score[1] >= threshold:
                        matches.append((entity, score[1]))

        return sorted(matches, key=lambda x: x[1], reverse=True)[:5] if matches else None
    
    def retrive_informer_data(self):
        if self.timestamp_file:
            if self.within_time():
                    try:
                        with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'r') as f:
                            entities = json.load(f)
                    except FileNotFoundError:
                        logger.info('Cache file not found, fetching new data')
                        self.save_timestamp()
                        entities=self.get_entities()
                        # save to qdrant for later retreival of the data based of semnantic search
                        self.client.delete_collection(collection_name= f'Galaxy_{self.entity_type}')
                        self.rag.save_doc_to_rag(data=entities, collection_name=f'Galaxy_{self.entity_type}')
                        with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'w') as f:
                            json.dump(entities, f, indent=4)
            else:
                logger.info('Cache expired, fetching new data')
                self.save_timestamp()
                entities = self.get_entities()
                # save to qdrant for later retreival of the data based of semnantic search
                self.client.delete_collection(collection_name= f'Galaxy_{self.entity_type}')
                self.rag.save_doc_to_rag(data=entities, collection_name=f'Galaxy_{self.entity_type}')
                with open(f'app/galaxy/cache_files/{self.entity_type}.json', 'w') as f:
                    json.dump(entities, f, indent=4)
                    
            return entities
    
    def search_entities(self, query, threshold=85):
        """Unified fuzzy search with priority fields"""
        entities=self.retrive_informer_data()
        fuzzy_result= self.fuzzy_search(query=query, entities=entities, threshold=threshold, config=self._entity_config[self.entity_type], matches=[])
        semantic_result= self.semantic_search(query=query, collection=f'Galaxy_{self.entity_type}', user_id='1234')
        prompt = SELECTION_PROMPT.format(input = query, tuple_items = fuzzy_result , dict_items= semantic_result)
        result= self.llm.generate(prompt = prompt)
        logger.info('Retrieving search results')
        return result

    def invocation_check(self,search_query):
        """Check if the query is asking about invocations of a workflow"""

        prompt= """Determine if in the input the query is asking about information on invocations of a workflow?
          If so Determine how much of information is required from the query and catagorize it as 'general' or 'specific'.
          Finally if return  the classification 'general' or 'specific' if the user is indeed asking about invocation otherwise return 'not_invocation' if the user is not asking about invocations at all.
         **Note**: Respond with only one of these options: 'general' , 'specific' or 'not_invocation' and nothing else. 
         **Input query**: {input}
        """
        response= self.llm.generate(prompt=prompt.format(input=search_query))
        logger.info(f'Checked invocation, state: {response}')
        if response in ['general', 'specific', 'not_invocation']:
            return response

    def get_entity_info(self, search_query, entity_id=None):
        """Unified info retrieval with LLM summary"""
        
        if not os.path.exists(f'app/galaxy/cache_files'):
             os.makedirs('app/galaxy/cache_files')
        if not os.path.exists('app/galaxy/cache_files/timestamps'):
             os.makedirs('app/galaxy/cache_files/timestamps')

        if entity_id:
            logger.info(f'Direct ID inputted, Retrieving information')
            entity = next((e for e in self.get_entities() if e['id'] == entity_id), None)
        else:
            entity = self.search_entities(search_query)
            logger.info(f'Search results: {entity}')
           
        invocation_info={}
        if self.entity_type == 'workflow' :
            invocation_check = self.invocation_check(search_query)
            if invocation_check != 'not_invocation':
                logger.info('gathering invocation information')
                for i, (key, item) in enumerate(entity.items()):
                    print(item['workflow_id'])
                    invocations = self.gi.invocations.get_invocations(workflow_id=item['workflow_id'])
                    print(f'general invocation: {invocations}')
                    if invocations:
                        print('yes invocation found')
                        
                        if str(invocation_check) == 'specific':
                            specific_invocations=[]
                            for invocation in invocations:
                                invocation=self.gi.invocations.show_invocation(invocation('id'))
                                invocation_steps=self.gi.invocations.get_invocation_step_jobs_summary(invocation['id']) # ?? wy is you not working
                                print(f'invocation steps: {invocation_steps if invocation_steps else "No steps found"}')
                                specific_invocations.append(invocation)
                            invocation_info[str(i)] = {'invocation' : specific_invocations, 'steps': invocation_steps}
                        elif str(invocation_check) == 'general':
                            invocation_info[str(i)] = invocations
                        else:
                            logger.info('No invocations found')
        
        # Get detailed information
        detail_methods = {
            'dataset': lambda id: self.gi.datasets.show_dataset(id),
            'tool': lambda id: self.gi.tools.show_tool(id, io_details=True),
            'workflow': lambda id: self.gi.workflows.show_workflow(id)
        }

        response_dict = {}
        print(invocation_info)
        if invocation_info:
            from pprint import pprint
            
            response_dict['invocation_info'] = invocation_info

        logger.info('Structuring the repsonse')
        for i, (key, item) in enumerate(entity.items()):
            details = detail_methods[self.entity_type](item[f'{self.entity_type}_id'])
            prompt_text = self._entity_config[self.entity_type]['summary_prompt'].format(input=details)
            responses_found = self.llm.generate(prompt=prompt_text)

            if isinstance(responses_found, str):
                try:
                    responses_found = json.loads(responses_found)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON returned: {responses_found}")

            response_dict[str(i)] = responses_found

        logger.info(f'Generating response for the query')
        response_text= self.llm.generate(prompt=RETRIEVE_PROMPT.format(query=search_query, retrieved_content=response_dict))
        response={
            'query': search_query,
            'retrieved_content': response_dict,
            'response': response_text
        }

        return response 

if __name__ == "__main__":
    informer= GalaxyInformer('tool')
    information=informer.get_entity_info('what tools are there in my instance tha convert bed files to gff files. can you tell me in detail')
    print(information)
