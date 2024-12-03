
import uuid
import json
import openai
from app.prompts.memory_prompt import FACT_RETRIEVAL_PROMPT,get_update_memory_messages
import traceback

class MemoryManager:
    def __init__(self, llm, client,embedding_model):
        """
        Initializes the MemoryManager with the necessary components.
        :param llm: The language model instance.
        :param embedding_model: The embedding model instance.
        :param client: The Qdrant client instance.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.client = client

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

            metadata = {}
            system_prompt, user_prompt = self.get_fact_retrieval_message(messages)
            response = self.llm.generate(user_prompt,system_prompt)

            try:
                new_retrieved_facts = response["facts"]
            except Exception:
                new_retrieved_facts = []

            retrieved_old_memory = []
            new_message_embeddings = {}

            for fact in new_retrieved_facts:
                embedded_message = self.embedding_model(fact)
                new_message_embeddings[fact] = embedded_message
                existing_memory = self.qdrant_client_retrieved_user_similar_preferences(user_id, embedded_message[0])

                if existing_memory:
                    for mem in existing_memory:
                        retrieved_old_memory.append({"id": mem["id"], "text": mem["content"]})

            temp_uuid_mapping = {str(idx): item["id"] for idx, item in enumerate(retrieved_old_memory)}
            for idx, item in enumerate(retrieved_old_memory):
                retrieved_old_memory[idx]["id"] = str(idx)

            function_calling_prompt = get_update_memory_messages(retrieved_old_memory, new_retrieved_facts)
            new_memories_with_actions = self.llm.generate(prompt=function_calling_prompt)
            returned_memories = []

            for resp in new_memories_with_actions["memory"]:
                data = resp["text"]
                if resp["event"] == "ADD":
                    memory_id = self.client._create_memory_update_memory(
                        user_id=user_id, data=data, embedding=new_message_embeddings[data], metadata=metadata
                    )
                    returned_memories.append({"id": memory_id, "memory": data, "event": resp["event"]})

                elif resp["event"] == "UPDATE":
                    self.client._create_memory_update_memory(
                        user_id=user_id,
                        memory_id=temp_uuid_mapping[resp["id"]],
                        data=data,
                        embedding=new_message_embeddings[data],
                        metadata=metadata,
                    )
                    returned_memories.append(
                        {
                            "id": temp_uuid_mapping[resp["id"]],
                            "memory": data,
                            "event": resp["event"],
                            "previous_memory": resp["old_memory"],
                        }
                    )

                elif resp["event"] == "NONE":
                    print("NOOP for Memory.")

            print("returned memories are ",returned_memories)
            return returned_memories
        except:
            traceback.print_exc()



