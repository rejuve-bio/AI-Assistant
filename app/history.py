import json
from datetime import datetime
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class History:
    def __init__(self, filename="history.json"):
        self.filename = filename
        self.history = self._load_history()
    
    def _load_history(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_history(self):
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)
    
    def create_history(self, user_id, user_message, assistant_answer):
        entry = {
            "user": user_message,
            "assistant answer": assistant_answer,
            "time": datetime.now().isoformat()
        }
        user_id_str = str(user_id)
        self.history = self._load_history()
       
        if user_id_str not in self.history:
            self.history[user_id_str] = []
                
        # Append new entry
        self.history[user_id_str].append(entry)
        
        self.history[user_id_str].sort(key=lambda x: x["time"])
        self._save_history()
    
    def retrieve_user_history(self, user_id):
        user_id_str = str(user_id)
        return {user_id_str: self.history.get(user_id_str, [])}