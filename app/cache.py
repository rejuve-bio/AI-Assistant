import json
import os

class ConversationCache:
    def __init__(self, file_path='conversation_cache.json'):
        self.file_path = file_path
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cache from JSON file, if it exists."""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        return {}

    def save_cache(self):
        """Save the cache to the JSON file."""
        with open(self.file_path, 'w') as file:
            json.dump(self.cache, file, indent=4)

    def add_conversation(self, user_id, user_query, assistant_message):
        """Add a conversation to the cache and save it to the file."""
        if user_id not in self.cache:
            self.cache[user_id] = {
                "user": [], 
                "assistant": []
            }
   
        self.cache[user_id]["user"].append(user_query)
        self.cache[user_id]["assistant"].append(assistant_message)
        self.save_cache()  # Save to JSON file after updating

    def get_conversation(self, user_id):
        """Retrieve the conversation for a specific user."""
        if user_id in self.cache:
            return self.cache[user_id]
        return None

    def clear_user_conversation(self, user_id):
        """Clear the conversation for a specific user and save changes to the file."""
        if user_id in self.cache:
            del self.cache[user_id]
            self.save_cache()  # Save to JSON file after clearing




conversation_cache = ConversationCache()
# Add sample data for user1
conversation_cache.add_conversation(
    user_id='user1', 
    user_query='Hello, what can you do?', 
    assistant_message='I can help you with various tasks like answering questions and providing information.'
)

conversation_cache.add_conversation(
    user_id='user1', 
    user_query='What is the weather today?', 
    assistant_message='I cannot check the weather at the moment.'
)

# Retrieve the conversation for user1
conversation_data = conversation_cache.get_conversation('user1')
# print("Retrieved Conversation:", conversation_data)
