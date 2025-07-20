import json
from datetime import datetime
import logging
import os
import uuid


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class History:
    def __init__(self, filename="history.json"):
        self.filename = filename
        self.history = self._load_history()

    def _load_history(self):
        try:
            logger.info(f"Loading history from file: {self.filename}")
            with open(self.filename, "r", encoding="utf-8") as file:
                data = json.load(file)
                logger.info(f"Successfully loaded history with {len(data)} users")
                return data
        except FileNotFoundError:
            logger.warning(f"History file not found: {self.filename}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading history: {e}")
            return {}

    def _save_history(self):
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)

    def create_history(self, user_id, user_message, assistant_answer):
        # Generate a unique query ID for this conversation entry
        query_id = str(uuid.uuid4())

        entry = {
            "query_id": query_id,
            "user": user_message,
            "assistant answer": assistant_answer,
            "time": datetime.now().isoformat(),
        }
        user_id_str = str(user_id)
        self.history = self._load_history()

        if user_id_str not in self.history:
            self.history[user_id_str] = []

        # Append new entry
        self.history[user_id_str].append(entry)

        self.history[user_id_str].sort(key=lambda x: x["time"])
        self.history[user_id_str] = self.history[user_id_str][-3:]
        self._save_history()

        # Return the query_id so it can be used by the calling function
        return query_id

    def retrieve_user_history(self, user_id):
        user_id_str = str(user_id)
        # Reload history from file to get latest data
        current_history = self._load_history()
        logger.info(
            f"Retrieved history for user {user_id_str}: {len(current_history.get(user_id_str, []))} entries"
        )
        return {user_id_str: current_history.get(user_id_str, [])}

    def get_entry_by_query_id(self, user_id, query_id):
        # Retrieve a specific conversation entry by query_id
        user_id_str = str(user_id)
        current_history = self._load_history()
        user_history = current_history.get(user_id_str, [])

        # Find the entry with matching query_id
        for entry in user_history:
            if entry.get("query_id") == query_id:
                return entry

        return None
