import os
import json
from datetime import datetime
from pymongo import MongoClient
import uuid
import logging

logging.getLogger("pymongo").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



class MongoManager:
    """Unified MongoDB manager for user information, content files, and history."""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.user_info_collection = None
        self.content_files_collection = None
        self.faq_collection = None
        self.hypothesis_collection = None
        self._connect()
        self._create_indexes()

    def _connect(self):
        """Initialize MongoDB connection"""
        try:
            mongo_uri = os.getenv("MONGO_URL")
            database_name = os.getenv("MONGO_DATABASE", "ai_assistant")

            self.client = MongoClient(mongo_uri)
            self.db = self.client[database_name]
            self.user_info_collection = self.db["user_information"]
            self.content_files_collection = self.db["user_content_files"]
            self.faq_collection = self.db["faq_questions"]
            self.hypothesis_collection = self.db["hypothesis_requests"]

            logger.info(f"MongoDB connection established to {database_name}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def _create_indexes(self):
        """Create necessary indexes for performance"""
        try:
            # User information indexes
            self.user_info_collection.create_index("user_id")
            self.user_info_collection.create_index("time")
            self.user_info_collection.create_index("question_id")

            # Content files indexes
            self.content_files_collection.create_index("user_id")
            self.content_files_collection.create_index("content_id", unique=True)
            self.content_files_collection.create_index("content_type")
            self.content_files_collection.create_index("upload_time")

            # FAQ indexes
            self.faq_collection.create_index("question_id", unique=True)
            self.faq_collection.create_index("display_order")

            # Hypothesis indexes
            self.hypothesis_collection.create_index("id", unique=True)
            self.hypothesis_collection.create_index("status")

            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    # ==================== CONVERSATION HISTORY METHODS ====================
    
    def create_history(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_answer: str, 
        graph_id_referenced: str = None,
        content_ids: list = None,
        urls: list = None,
        agents_used: list = None,

    ):
        """
        Create a conversation history entry with both question and answer.
        This is the main method for saving conversations.
        """
        try:
            # Clean old records (keep only 3 most recent)
            self._clean_old_user_records(user_id)

            user_info = {
                "user_id": user_id,
                "question_id": str(uuid.uuid4()),
                "user_question": user_message,
                "assistant_answer": assistant_answer,
                "graph_id_referenced": graph_id_referenced,
                "content_ids":content_ids,
                "urls":urls,
                "agents_used": agents_used,
                "memory": None,
                "context": None,
                "time": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.user_info_collection.insert_one(user_info)
            logger.info(f"Created history with question_id: {user_info['question_id']}")
            
            return user_info.get("question_id")

        except Exception as e:
            logger.error(f"Error creating history: {e}")
            return None

    def retrieve_user_history(self, user_id: str, limit: int = 5):
        """Retrieve user conversation history"""
        try:
            cursor = (
                self.user_info_collection.find({"user_id": user_id})
                .sort("time", -1)  # Most recent first
                .limit(limit)
            )
            records = list(cursor)
            
            history = []
            for record in records:
                history.append({
                    "query_id": record.get("question_id"),
                    "user": record.get("user_question"),
                    "assistant_answer": record.get("assistant_answer"),
                    "graph_id_referenced": record.get("graph_id_referenced"),
                    "time": (
                        record.get("time").isoformat() if record.get("time") else None
                    ),
                })
            
            return {str(user_id): history}

        except Exception as e:
            logger.error(f"Error retrieving user history: {e}")
            return {str(user_id): []}

    def get_entry_by_query_id(self, user_id: str, query_id: str):
        """Get a specific history entry by query ID"""
        try:
            record = self.user_info_collection.find_one(
                {"user_id": user_id, "question_id": query_id}
            )

            if record:
                return {
                    "query_id": record.get("question_id"),
                    "user": record.get("user_question"),
                    "assistant_answer": record.get("assistant_answer"),
                    "graph_id_referenced": record.get("graph_id_referenced"),
                    "time": record.get("time").isoformat() if record.get("time") else None,
                }
            return None

        except Exception as e:
            logger.error(f"Error getting entry by query ID: {e}")
            return None

    def clear_user_history(self, user_id: str):
        """Clear all history for a user"""
        try:
            result = self.user_info_collection.delete_many({"user_id": user_id})
            logger.info(f"Cleared {result.deleted_count} history records for user {user_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error clearing user history: {e}")
            return 0

    def _clean_old_user_records(self, user_id: str):
        """Keep only 3 most recent records per user"""
        try:
            all_records = list(
                self.user_info_collection.find({"user_id": user_id}).sort("time", -1)
            )

            if len(all_records) > 3:
                records_to_delete = all_records[3:]
                for record in records_to_delete:
                    self.user_info_collection.delete_one({"_id": record["_id"]})
                logger.info(
                    f"Cleaned {len(records_to_delete)} old records for user {user_id}"
                )

        except Exception as e:
            logger.error(f"Error cleaning old user records: {e}")

    def get_context_and_memory(self, user_id: str):
        try:
            cursor = (
                self.user_info_collection.find({"user_id": user_id})
                .sort("time", -1)
                .limit(3)
            )
            records = list(cursor)
            records.reverse()

            result = []
            for record in records:
                # Parse memory if exists
                memory = None
                if record.get("memory"):
                    try:
                        memory_data = json.loads(record["memory"])
                        if "content" in memory_data:
                            content = memory_data["content"]
                            memory = json.loads(content) if isinstance(content, str) else content
                        else:
                            memory = memory_data
                    except (json.JSONDecodeError, TypeError):
                        memory = None

                if memory in [None, []]:
                    memory = ""

                result.append({
                    "question": record.get("user_question", ""),
                    "context": {
                        "answer": record.get("assistant_answer", ""),
                        "agents_used": record.get("agents_used", []),
                        "memory": memory,
                    },
                })

            return result

        except Exception as e:
            logger.error(f"Error getting context and memory: {e}")
            return []
    def add_content_file(
        self,
        user_id: str,
        content_id: str,
        content_type: str = "pdf",
        filename: str = None,
        num_pages: int = None,
        url: str = None,
        title: str = None,
        author: str = None,
        publish_date: datetime = None,
        file_size: float = None,
        upload_time: datetime = None,
        summary: str = None,
        keywords: str = None,
        topics: str = None,
        suggested_questions: str = None,
    ):
        """Add a content file record"""
        try:
            content_file = {
                "user_id": user_id,
                "content_id": content_id,
                "content_type": content_type,
                "filename": filename,
                "num_pages": num_pages,
                "url": url,
                "title": title,
                "author": author,
                "publish_date": publish_date,
                "file_size": file_size,
                "upload_time": upload_time or datetime.utcnow(),
                "summary": summary,
                "keywords": keywords,
                "topics": topics,
                "suggested_questions": suggested_questions,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.content_files_collection.insert_one(content_file)
            content_file["_id"] = result.inserted_id
            logger.info(f"Added content file: {content_id}")
            return content_file

        except Exception as e:
            logger.error(f"Error adding content file: {e}")
            raise

    def get_user_content_files(self, user_id: str, content_type: str = None):
        """Get content files for a user"""
        try:
            query = {"user_id": user_id}
            if content_type:
                query["content_type"] = content_type

            cursor = self.content_files_collection.find(query).sort("upload_time", -1)
            return list(cursor)

        except Exception as e:
            logger.error(f"Error retrieving content files: {e}")
            return []

    def get_content_file_by_id(self, user_id: str, content_id: str):
        """Get a specific content file by ID"""
        try:
            return self.content_files_collection.find_one(
                {"user_id": user_id, "content_id": content_id}
            )
        except Exception as e:
            logger.error(f"Error retrieving content file by ID: {e}")
            return None

    def get_content_count(self, user_id: str, content_type: str = None):
        """Count content files for a user"""
        try:
            query = {"user_id": user_id}
            if content_type:
                query["content_type"] = content_type

            return self.content_files_collection.count_documents(query)

        except Exception as e:
            logger.error(f"Error counting content files: {e}")
            return 0

    def update_content_file(
        self,
        user_id: str,
        content_id: str,
        summary: str = None,
        keywords: str = None,
        topics: str = None,
        suggested_questions: str = None,
    ):
        """Update a content file record"""
        try:
            update_data = {"updated_at": datetime.utcnow()}
            if summary is not None:
                update_data["summary"] = summary
            if keywords is not None:
                update_data["keywords"] = keywords
            if topics is not None:
                update_data["topics"] = topics
            if suggested_questions is not None:
                update_data["suggested_questions"] = suggested_questions

            result = self.content_files_collection.update_one(
                {"user_id": user_id, "content_id": content_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                return self.get_content_file_by_id(user_id, content_id)
            return None

        except Exception as e:
            logger.error(f"Error updating content file: {e}")
            raise

    def delete_content_file(self, user_id: str, content_id: str):
        """Delete a content file record"""
        try:
            result = self.content_files_collection.delete_one(
                {"user_id": user_id, "content_id": content_id}
            )
            deleted = result.deleted_count > 0
            if deleted:
                logger.info(f"Deleted content file: {content_id}")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting content file: {e}")
            return False


    # ==================== FAQ METHODS ====================

    def get_all_faq_questions(self, context=None):
        """Get all FAQ questions ordered by display_order"""
        try:
            query = {"context":context}
            cursor = self.faq_collection.find(query).sort("display_order", 1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving FAQ questions: {e}")
            return []

    def get_faq_by_id(self, question_id: str):
        """Get a specific FAQ question and answer"""
        try:
            return self.faq_collection.find_one({"question_id": question_id})
        except Exception as e:
            logger.error(f"Error retrieving FAQ by ID: {e}")
            return None

    def seed_faq_questions(self, questions: list):
        """
        Sync FAQ questions from JSON to MongoDB.
        - Adds new questions
        - Updates existing questions (if content changed)
        - Removes questions not present in the input list
        """
        try:
            # 1. Get all current question IDs from the input list
            input_ids = [q["question_id"] for q in questions]
            
            # 2. Update or Insert (Upsert) each question from the input
            for q in questions:
                self.faq_collection.replace_one(
                    {"question_id": q["question_id"]},
                    q,
                    upsert=True
                )
            
            # 3. Delete questions that are in DB but NOT in the input list
            result = self.faq_collection.delete_many(
                {"question_id": {"$nin": input_ids}}
            )
            
            logger.info(f"FAQ Sync Complete: Seeded/Updated {len(questions)} questions. Removed {result.deleted_count} old questions.")
            
        except Exception as e:
            logger.error(f"Error seeding FAQ questions: {e}")


    # ==================== HYPOTHESIS METHODS ====================

    # Note: The methods below are currently "Dead Code" and not used by the AI Assistant.
    # They were designed for local job tracking in MongoDB, but the current implementation 
    # uses a direct API flow with the hypothesis server.

    def create_hypothesis_request(self, data: dict):
        """Create a new hypothesis request"""
        try:
            self.hypothesis_collection.insert_one(data)
            logger.info(f"Created hypothesis request: {data.get('id')}")
            return True
        except Exception as e:
            logger.error(f"Error creating hypothesis request: {e}")
            raise

    def get_hypothesis_request(self, hypothesis_id: str):
        """Get a hypothesis request by ID"""
        try:
            return self.hypothesis_collection.find_one({"id": hypothesis_id})
        except Exception as e:
            logger.error(f"Error retrieving hypothesis request: {e}")
            return None

    def update_hypothesis_status(self, hypothesis_id: str, status: str, enrich_id: str = None):
        """Update the status and enrich_id of a hypothesis request"""
        try:
            update_data = {"status": status}
            if enrich_id:
                update_data["enrich_id"] = enrich_id
            
            result = self.hypothesis_collection.update_one(
                {"id": hypothesis_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating hypothesis status: {e}")
            return False


# Global instance
mongo_db_manager = MongoManager()