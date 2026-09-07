import uuid
import json
import openai
from app.prompts.memory_prompt import FACT_RETRIEVAL_PROMPT, get_update_memory_messages
from ..llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from app.storage.qdrant import Qdrant
import traceback


class MemoryManager:
    def __init__(self, llm, qdrant_client):
        """
        Initializes the MemoryManager with the necessary components.
        :param llm: The language model instance.
        :param qdrant_client: The Qdrant client instance (already carries its own embedding_model).
        """
        self.llm = llm
        self.client = qdrant_client
        self.embedding_model = qdrant_client.embedding_model

    def get_fact_retrieval_message(self, messages):
        """
        Constructs the fact retrieval prompt.
        :param messages: The input messages.
        :return: A tuple containing the system and user prompts.
        """
        return FACT_RETRIEVAL_PROMPT, f"Input: {messages}"

    def qdrant_client_retrieved_user_similar_preferences(self, user_id, embedding):
        """
        Retrieves similar user preferences from Qdrant.
        :param user_id: The user ID.
        :param embedding: The embedding vector.
        :return: Retrieved contents from Qdrant.
        """
        return self.client._retrieve_memory(user_id, embedding)

    def _build_memory_entry(self, resp, user_id, new_message_embeddings, temp_uuid_mapping):
        data = resp["text"]
        event = resp["event"]
        if event == "ADD":
            memory_id = self.client._create_memory_update_memory(
                user_id=user_id,
                data=data,
                embedding=new_message_embeddings[data],
            )
            return {"id": memory_id, "memory": data, "event": event}
        if event == "UPDATE":
            self.client._create_memory_update_memory(
                user_id=user_id,
                memory_id=temp_uuid_mapping[resp["id"]],
                data=data,
                embedding=new_message_embeddings[data],
            )
            return {
                "id": temp_uuid_mapping[resp["id"]],
                "memory": data,
                "event": event,
                "previous_memory": resp["old_memory"],
            }
        if event == "NONE":
            logger.debug("NOOP for Memory.")
        return None

    def add_memory(self, messages, user_id):
        try:
            """
            Adds memory for a user.
            :param messages: Messages from the user.
            :param user_id: The user ID.
            :return: A list of returned memories with their details.
            """
            if not user_id:
                return "userid is an obligatory to save memory"
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            else:
                messages = []

            system_prompt, user_prompt = self.get_fact_retrieval_message(messages)
            response = self.llm.generate(user_prompt, system_prompt)

            try:
                new_retrieved_facts = response["facts"]
            except Exception:
                new_retrieved_facts = []

            retrieved_old_memory = []
            new_message_embeddings = {}

            for fact in new_retrieved_facts:
                embedded_message = self.embedding_model(fact)
                new_message_embeddings[fact] = embedded_message
                existing_memory = self.qdrant_client_retrieved_user_similar_preferences(
                    user_id, embedded_message[0]
                )

                if existing_memory:
                    for mem in existing_memory:
                        retrieved_old_memory.append(
                            {"id": mem["id"], "text": mem["content"]}
                        )

            temp_uuid_mapping = {
                str(idx): item["id"] for idx, item in enumerate(retrieved_old_memory)
            }
            for idx, item in enumerate(retrieved_old_memory):
                retrieved_old_memory[idx]["id"] = str(idx)

            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts
            )
            new_memories_with_actions = self.llm.generate(
                prompt=function_calling_prompt
            )
            returned_memories = []

            for resp in new_memories_with_actions["memory"]:
                entry = self._build_memory_entry(
                    resp, user_id, new_message_embeddings, temp_uuid_mapping
                )
                if entry is not None:
                    returned_memories.append(entry)

            logger.debug(f"Returned memories: {returned_memories}")
            return returned_memories
        except Exception:
            logger.error("Failed to add memory", exc_info=True)
