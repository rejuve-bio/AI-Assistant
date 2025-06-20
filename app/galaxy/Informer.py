import os
import re
import ast
from rapidfuzz import process, fuzz 
from bioblend.galaxy import GalaxyInstance
from dotenv import load_dotenv
import json
import numpy as np
import logging
import redis

from sys import path
path.append('.')

from app.storage.qdrant import Qdrant
from app.rag.rag import RAG
from app.llm_handle.llm_models import GeminiModel, gemini_embedding_model
from app.prompts.galaxy_prompts import TOOL_PROMPT, WORKFLOW_PROMPT, DATASET_PROMPT, SELECTION_PROMPT
from app.prompts.rag_prompts import RETRIEVE_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GalaxyInformer:
    def __init__(self, entity_type):
        logger.info(f'initializing the galaxy informer with {entity_type} type')
        load_dotenv()
        self.entity_type = entity_type.lower()
        self.gi = GalaxyInstance(url=os.getenv("GALAXY_URL"), key=os.getenv("GALAXY_API"))
        self.llm = GeminiModel(api_key=os.getenv("GEMINI_API_KEY"), model_provider='gemini', model_name=os.getenv("ADVANCED_LLM_VERSION"))
        self.client = Qdrant()
        self.rag = RAG(client=self.client, llm=self.llm)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)  # Initialize Redis client
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
        
        return {k: results[k] for k in sorted(results.keys())[:10]}

    def parse_list(self, list_str):
        """Extract and safely parse a Python list from a string that may include markdown code block markers."""
        try:
            # Remove markdown code block markers (``` or ```python)
            cleaned_str = re.sub(r"^```(?:python)?\s*|\s*```$", "", list_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
            parsed = ast.literal_eval(cleaned_str.strip())
            if isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Parsed object is not a list")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing list: {e}")
            return []

    def extract_search_query(self, query):
        prompt = f"""
        Extract the main keywords from the following query for a fuzzy search in a Galaxy platform(tool/workflow/dataset/invocation) database. 
        Return a Python list of a combination of keywords that can potentially be used to get search results for to the inputed query.
        
        Input query: "{query}"
        
        Output (Python list of keywords): []
        """
        keywords = self.llm.generate(prompt=prompt)
        logger.info("Extracting keywords from input query for fuzzy search")
        
        try:
            if isinstance(keywords, list):
                return keywords
            elif isinstance(keywords, str):
                keywords=self.parse_list(list_str=keywords)
                return keywords
        except Exception as e:
            logger.error(f"Failed to parse keywords list: {e}")

    def fuzzy_search(self, query, entities, config, threshold):
        """Fuzzy search with priority fields, improved for efficiency and clarity."""
        logger.info('Fuzzy search for the query by priority fields')
        
        # Prepare priority candidates
        priority_candidates = []
        entity_map = {}  # Map: field string -> entity
        
        for entity in entities:
            for field in config['search_fields']:
                if field in entity and isinstance(entity[field], str):
                    field_value = entity[field]
                    priority_candidates.append(field_value)
                    entity_map[field_value] = entity
        
        # Run fuzzy match across all priority fields at once
        results = process.extract(query, priority_candidates, scorer=fuzz.WRatio, limit=10)
        
        # Filter by threshold and collect matches
        priority_matches = []
        for match_str, score, _ in results:
            if score >= threshold:
                priority_matches.append((entity_map[match_str], score))
        
        if priority_matches:
            logger.info(f'Found {len(priority_matches)} matches in priority fields for search query: {query}')
            return sorted(priority_matches, key=lambda x: x[1], reverse=True)[:5]
        
        # No good priority matches found → fallback to all other fields
        logger.info(f'No matches found in the priority fields for search query: {query}, searching in all fields as a fallback')
        
        fallback_candidates = []
        fallback_map = {}
        
        for entity in entities:
            for key, value in entity.items():
                if key not in config['search_fields'] and isinstance(value, str):
                    fallback_candidates.append(value)
                    fallback_map[value] = entity
        
        results = process.extract(query, fallback_candidates, scorer=fuzz.WRatio, limit=10)
        
        fallback_matches = []
        for match_str, score, _ in results:
            if score >= threshold:
                fallback_matches.append((fallback_map[match_str], score))
        
        if fallback_matches:
            logger.info(f'Found {len(fallback_matches)} matches in fallback fields for search query: {query}')
            return sorted(fallback_matches, key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f'No matches found in either priority or fallback fields for search query: {query}')
        return []  # Always return a list, even if empty

    def retrive_informer_data(self):
        entities_str = self.redis_client.get(f"{self.entity_type}_entities")
        if entities_str is not None:
            try:
                entities = json.loads(entities_str)
                logger.info(f'Retrieved cached {self.entity_type} entities from Redis')
                return entities
            except json.JSONDecodeError:
                logger.error(f'Failed to parse cached {self.entity_type} entities from Redis')

        # Fetch new data
        logger.info(f'No valid cache found for {self.entity_type}, fetching new data')
        entities = self.get_entities()
        # Save to Redis with TTL of 10 hours (36000 seconds)
        try:
            self.redis_client.setex(f"{self.entity_type}_entities", 36000, json.dumps(entities))
            logger.info(f'Saved {self.entity_type} entities to Redis with 10-hour TTL')
        except redis.RedisError as e:
            logger.error(f'Failed to save {self.entity_type} entities to Redis: {e}')
        # Save to Qdrant
        try:
            self.client.delete_collection(collection_name=f'Galaxy_{self.entity_type}')
            self.rag.save_doc_to_rag(data=entities, collection_name=f'Galaxy_{self.entity_type}')
            logger.info(f'Saved {self.entity_type} entities to Qdrant')
        except Exception as e:
            logger.error(f'Failed to save {self.entity_type} entities to Qdrant: {e}')
        return entities

    def search_entities(self, query, user_id, threshold=85):
        """Unified fuzzy and semantic search"""
        entities=self.retrive_informer_data()
        fuzzy_inputs=self.extract_search_query(query=query)
        fuzzy_results=[]
        # Fuzzy search using key words
        if isinstance(fuzzy_inputs, list):
            logger.info(f"queries for fuzzy search: {fuzzy_inputs}")
            
            for fuzzy_query in fuzzy_inputs:
                fuzzy_result= self.fuzzy_search(query=fuzzy_query, entities=entities, threshold=threshold, config=self._entity_config[self.entity_type])
                fuzzy_results.extend(fuzzy_result)
        else:
            logger.info(f"queries for fuzzy search: {query}")
            fuzzy_results= self.fuzzy_search(query=query, entities=entities, threshold=threshold, config=self._entity_config[self.entity_type])

        # semantic search on the qdrant db
        semantic_result= self.semantic_search(query=query, collection=f'Galaxy_{self.entity_type}', user_id=user_id)
        
        # reranking and structuring the search result found and retrieving the top 3 findings
        prompt = SELECTION_PROMPT.format(input = query, tuple_items = fuzzy_results , dict_items= semantic_result)
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
        response= self.llm.generate(prompt=prompt.format(input=search_query)).strip().lower()
        logger.info(f'Checked invocation, state: {repr(response)}')
        if response in ['general', 'specific', 'not_invocation']:
            return response
        else:
            logger.info('failed to identify invocation')

    def get_entity_info(self, search_query, user_id, entity_id=None):
        """Unified info retrieval with LLM summary"""
        
        # Setting a default value for searching option based on input
        search_bool = False

        if entity_id:
            logger.info(f'Direct ID inputted, Retrieving information')
            retrived_entity = next((e for e in self.get_entities() if e['id'] == entity_id), None)
            if retrived_entity != None:
                # Result found
                search_bool=True
                # structuring for further information retrieval.
                entity={'0': {'name': retrived_entity['name'], f'{self.entity_type}_id' : retrived_entity['id'] }}
                logger.info(f'direct id retrival result {entity}')
            else: 
                logger.info(f'{self.entity_type} with id {entity_id} not found, searching for similar data')
                
        if entity_id is None or search_bool is False:
            entity = self.search_entities(query = search_query, user_id = user_id)
            logger.info(f'Search results: {entity}')
           
        invocation_info={}
        if self.entity_type == 'workflow' :
            # Retriving more information about workflow
            invocation_check = self.invocation_check(search_query)
            if invocation_check != 'not_invocation':
                logger.info('gathering invocation information')
                for i, (key, item) in enumerate(entity.items()):
                    # Get results for the workflows found and choosen
                    invocations = self.gi.invocations.get_invocations(workflow_id=item['workflow_id'])
                    if invocations:
                        logger.info('invocations found, retriving information')
                        
                        if str(invocation_check) == 'specific':
                            specific_invocations=[]
                            for invocation in invocations:
                                # show invocation details with the invocation id
                                invocation=self.gi.invocations.show_invocation(invocation['id'])
                                # show invocation steps summary from the galaxy instance with the invocation id
                                invocation_steps=self.gi.invocations.get_invocation_step_jobs_summary(invocation['id'])
                                # Get invocation report from the galaxy instance with the invocation id
                                invocation_report=self.gi.invocations.get_invocation_report(invocation['id'])                          
                                # Collect results
                                specific_invocations.append({'invocation' : invocation, 'report': invocation_report,'steps': invocation_steps})
                            invocation_info[str(i)] = specific_invocations
                            
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
        if invocation_info:            
            response_dict['invocation_info'] = invocation_info

        logger.info('Structuring the repsonse')
        for i, (key, item) in enumerate(entity.items()):
            details = detail_methods[self.entity_type](item[f'{self.entity_type}_id'])
            # prompt_text = self._entity_config[self.entity_type]['summary_prompt'].format(input=details)
            # responses_found = self.llm.generate(prompt=prompt_text)

            # if isinstance(responses_found, str):
            #     try:
            #         responses_found = json.loads(responses_found)
            #     except json.JSONDecodeError:
            #         raise ValueError(f"Invalid JSON returned: {responses_found}")

            response_dict[str(i)] = details

        logger.info(f'Generating response for the query')
        response_text= self.llm.generate(prompt=RETRIEVE_PROMPT.format(query=search_query, retrieved_content=response_dict) )
        response={
            'query': search_query,
            'retrieved_content': response_dict,
            'response': response_text
        }

        return response 

if __name__ == "__main__":
    # Testing with as simple query    
    informer= GalaxyInformer('workflow')
    input_query= 'Tell me about the RNA-Seq Analysis workflow, tell me about its last recent invocation in detail'
    information=informer.get_entity_info(search_query = input_query, user_id = '1234')
    print(information['response'])