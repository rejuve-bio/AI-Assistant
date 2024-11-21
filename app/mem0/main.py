import concurrent
from datetime import datetime
import json
import logging
import os
from typing import Optional
import pandas as pd
from pydantic import Field
from app.mem0.base import MemoryBase
from app.mem0.utils import get_fact_retrieval_messages, parse_messages
from app.llm_handle.llm_models import GeminiModel, LLMInterface, embedding_model
from app.mem0.mem0_prompt import get_update_memory_messages, METADATA_PROMPT
from app.rag.qdrant import Qdrant
from app.rag.query import RAG, VECTOR_COLLECTION
import google.generativeai as genai
from app.mem0.baseconfig import BaseLlmConfig
from app.mem0.gemini import GeminiLLM

logger = logging.getLogger(__name__)





class Memory():
    def __init__(self, llm):
        self.llm = llm
        self.rag = RAG(self.llm)
        self.qdrant = Qdrant()

    def add(self, messages, user_id=None, agent_id=None, metadata=None, filters=None, prompt=None):
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters)
            concurrent.futures.wait([future1])
            vector_store_result = future1.result()
            return vector_store_result
        
    custom_prompt: Optional[str] = Field(
    description="Custom prompt for the memory",
    default=None,
    )

  
    def _add_to_vector_store(self, messages, metadata, filters):
        parsed_messages = parse_messages(messages)
        print("parsed_messages ", parsed_messages)
        self.generate_metadata(parsed_messages)

        print("parse_messages", parsed_messages)

        
     

       
        system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)
        response = self.llm.generate_response(
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
        )
        print("response:", response)
        cleaned_response = response.strip().replace('```json\n', '').replace('```', '').strip()

        try:
            new_retrieved_facts = json.loads(cleaned_response)["facts"]
            print('new_retrieved_facts:', new_retrieved_facts)
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        
        new_message_embeddings = {}
        retrieved_old_memory = []
        
        for new_mem in new_retrieved_facts:
            message_embedding = embedding_model(new_mem, self.llm)
            new_message_embeddings[new_mem] = message_embedding

            messages_embeddings_df = pd.DataFrame({"dense": [message_embedding], "id": [1]})
            print(messages_embeddings_df)

          
            existing_memories = self.qdrant.retrieve_user_data(
                collection_name='mem0',
                query=message_embedding,
                user_id=123
            )

            for mem in existing_memories:
                print('mem:', mem)
                retrieved_old_memory.append({"id": mem[1], "text": "list types of proteins?"})
                print("retrieved_old_memory", retrieved_old_memory)

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)
        
        function_calling_prompt = get_update_memory_messages(retrieved_old_memory, new_retrieved_facts)

        new_memories_with_actions = self.llm.generate_response(
            messages=[{"role": "user", "content": function_calling_prompt}],
            response_format={"type": "json_object"},
        )

       
        new_memories_with_actions = json.loads(new_memories_with_actions)
        returned_memories = []
        for resp in new_memories_with_actions.get("memory", []):
            try:
                if resp["event"] == "ADD":
                    memory_id = self._create_memory(
                        data=resp["text"], existing_embeddings=new_message_embeddings, metadata=metadata
                    )
                    returned_memories.append({"id": memory_id, "memory": resp["text"], "event": resp["event"]})
                elif resp["event"] == "UPDATE":
                    self._update_memory(
                        memory_id=temp_uuid_mapping[resp["id"]],
                        data=resp["text"],
                        existing_embeddings=new_message_embeddings,
                        metadata=metadata,
                    )
                    returned_memories.append({"id": temp_uuid_mapping[resp["id"]], "memory": resp["text"], "event": resp["event"], "previous_memory": resp["old_memory"]})
                elif resp["event"] == "DELETE":
                    self._delete_memory(memory_id=temp_uuid_mapping[resp["id"]])
                    returned_memories.append({"id": temp_uuid_mapping[resp["id"]], "memory": resp["text"], "event": resp["event"]})
            except Exception as e:
                logging.error(f"Error in new_memories_with_actions: {e}")

        return returned_memories

   
    def generate_metadata(self, parsed_messages):
   
     if isinstance(parsed_messages, str):
        parsed_messages = [{"role": "user", "content": parsed_messages}]
        user_message = parsed_messages[0].get("content", "")
        system_message = METADATA_PROMPT + "\n" + f"Conversation: {user_message}"
        response = self.llm.generate_response(
        messages=[{"role": "system", "content": system_message}],
        response_format={"type": "json_object"},
        )

     try:
            metadata = json.loads(response)
     except json.JSONDecodeError:
            logging.error("Failed to parse metadata response")
            metadata = {}
            metadata["date"] = str(datetime.now())
            print("metadata generated", metadata)

            return metadata
config = BaseLlmConfig(model="gemini-1.5-flash-latest", temperature=0.7, max_tokens=100, top_p=1.0)
llm = GeminiLLM(config)
messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": " give me some examples of proteins?"},
    ]


test=Memory(llm)._add_to_vector_store( messages=messages, metadata=None, filters=None)

