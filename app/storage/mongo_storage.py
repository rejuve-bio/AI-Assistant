import os
import json
from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from langgraph.checkpoint.mongodb import MongoDBSaver
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
        self.threads_collection = None
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
            self.threads_collection = self.db["conversation_threads"]

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
            self.content_files_collection.create_index("content_hash")

            # FAQ indexes
            self.faq_collection.create_index("question_id", unique=True)
            self.faq_collection.create_index("display_order")

            # Hypothesis indexes
            self.hypothesis_collection.create_index("id", unique=True)
            self.hypothesis_collection.create_index("status")

            # Conversation thread indexes
            self.threads_collection.create_index("thread_id", unique=True)
            self.threads_collection.create_index("user_id")

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
        pending_json: dict = None,
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
                "pending_json": pending_json,
                "memory": None,
                "context": None,
                "time": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            self.user_info_collection.insert_one(user_info)
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

    def get_latest_pending_json(self, user_id: str):
        """Return pending_json from the most recent history entry if it has one, else None."""
        try:
            latest = self.user_info_collection.find_one(
                {"user_id": user_id},
                sort=[("time", -1)]
            )
            if latest and latest.get("pending_json"):
                return latest["pending_json"]
            return None
        except Exception as e:
            logger.error(f"Error getting latest pending JSON: {e}")
            return None

    def _parse_memory_record(self, record):
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
        return memory

    @staticmethod
    def _time_ago(timestamp) -> str:
        """Human-readable elapsed time since a stored UTC timestamp, e.g. '12 minutes ago'."""
        if not timestamp:
            return "unknown time ago"
        seconds = (datetime.utcnow() - timestamp).total_seconds()
        if seconds < 60:
            return "less than a minute ago"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"

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
                memory = self._parse_memory_record(record)
                result.append({
                    "question": record.get("user_question", ""),
                    "asked": self._time_ago(record.get("time")),
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

    def get_content_summaries(self, user_id, content_ids=None):
        """Summaries for a user's uploaded content (PDF and web), optionally
        narrowed to specific content_ids."""
        content_summaries = []
        all_content = self.get_user_content_files(user_id)

        if content_ids:
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
            filtered_content = all_content

        for content in filtered_content:
            if content.get("content_type") == "pdf":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "pdf",
                        "filename": content.get("filename"),
                        "summary": content.get("summary") or "",
                    }
                )
            elif content.get("content_type") == "web":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "web",
                        "url": content.get("url"),
                        "title": content.get("title"),
                        "summary": content.get("summary") or "",
                    }
                )

        return content_summaries

    

    def get_thread(self, thread_id: str, user_id: str = None) -> dict:
        """Returns the thread document, or an empty-shaped dict if it doesn't
        exist yet (never returns None — callers can always read its fields).
        """
        query = {"thread_id": thread_id}
        if user_id:
            query["user_id"] = user_id
        try:
            doc = self.threads_collection.find_one(query)
            if doc:
                return doc
        except Exception as e:
            logger.error(f"Error getting thread {thread_id}: {e}")
        return {"thread_id": thread_id, "messages": [], "tool_calls": [], "content_ids": [], "running_summary": "", "message_count": 0, "summary_watermark": 0}

    def append_proposal_summary(self, thread_id: str, user_id: str, line: str) -> None:
        """Appends one short, deterministic line describing a proposal
        (kind, title, real ids, source/type) to the thread's running
        summary. This -- not the full proposal objects -- is what _seed
        feeds the orchestrator for awareness of earlier proposals in the
        thread, so "summarize the hypothesis from before" has something
        accurate and compact to work from without re-exposing raw state
        that goes stale. Distinct field from the legacy `running_summary`
        string used by the old /query path -- this one is a growing list of
        lines, owned entirely by the new orchestrator."""
        if not line:
            return
        try:
            self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {
                    "$push": {"proposal_summary_lines": line},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$setOnInsert": {
                        "thread_id": thread_id, "user_id": user_id,
                        "messages": [], "tool_calls": [], "running_summary": "",
                    },
                },
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Error appending proposal summary for thread {thread_id}: {e}")

    def get_proposal_summary(self, thread_id: str, user_id: str) -> list:
        return self.get_thread(thread_id, user_id).get("proposal_summary_lines", [])

    def add_thread_content_ids(self, thread_id: str, user_id: str, content_ids: list) -> None:

        if not content_ids:
            return
        try:
            self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {
                    "$addToSet": {"content_ids": {"$each": content_ids}},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$setOnInsert": {
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "messages": [],
                        "tool_calls": [],
                        "running_summary": "",
                    },
                },
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Error recording content_ids for thread {thread_id}: {e}")

    def get_thread_content_ids(self, thread_id: str, user_id: str) -> list:
        return self.get_thread(thread_id, user_id).get("content_ids", [])

    def append_message(
        self,
        thread_id: str,
        user_id: str,
        role: str,
        text: str,
        json_format: dict = None,
        agents: list = None,
        graph_id: str = None,
        resource_type: str = None,
        client_message_id: str = None,
        context_part: dict = None,
        run_part: dict = None,
        files_part: list = None,
        quick_replies: list = None,
        proposals: list = None,
    ) -> dict:
        try:
            counter_doc = self.threads_collection.find_one_and_update(
                {"thread_id": thread_id, "user_id": user_id},
                {
                    "$inc": {"next_seq": 1},
                    "$setOnInsert": {
                        "thread_id": thread_id, "user_id": user_id,
                        "messages": [], "tool_calls": [], "running_summary": "",
                    },
                },
                upsert=True,
                return_document=True,
            )
            seq = counter_doc.get("next_seq", 1)

            entry = {"seq": seq, "role": role, "text": text, "created_at": datetime.utcnow()}
            if json_format:
                entry["json_format"] = json_format
            if agents:
                entry["agents"] = agents
            if graph_id:
                entry["graph_id"] = graph_id
            if resource_type:
                entry["resource_type"] = resource_type
            elif graph_id:
                entry["resource_type"] = "unknown"
            if client_message_id:
                entry["client_message_id"] = client_message_id
            if context_part:
                entry["context_part"] = context_part
            if run_part:
                entry["run_part"] = run_part
            if files_part:
                entry["files_part"] = files_part
            if quick_replies:
                entry["quick_replies"] = quick_replies
            if proposals:
                entry["proposals"] = proposals
            doc = self.threads_collection.find_one_and_update(
                {"thread_id": thread_id, "user_id": user_id},
                {
                    "$push": {"messages": entry},
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": datetime.utcnow()},
                },
                upsert=True,
                return_document=True,
            )
            return doc or self.get_thread(thread_id, user_id)
        except DuplicateKeyError:
            logger.warning(
                f"Refusing to append to thread {thread_id}: owned by another user"
            )
            return self.get_thread(thread_id, user_id)
        except Exception as e:
            logger.error(f"Error appending message to thread {thread_id}: {e}")
            return self.get_thread(thread_id, user_id)

    def find_message_by_client_id(self, thread_id: str, user_id: str, client_message_id: str) -> Optional[dict]:
        """Looks up a previously-stored message by its client-supplied
        idempotency key, scoped to this thread. Used to detect a retried
        send so it can be answered from the stored result instead of
        reprocessed -- see the client_message_id convention in the
        Threads API v1 proposal."""
        if not client_message_id:
            return None
        try:
            doc = self.threads_collection.find_one(
                {"thread_id": thread_id, "user_id": user_id, "messages.client_message_id": client_message_id},
                {"messages.$": 1},
            )
            if doc and doc.get("messages"):
                return doc["messages"][0]
        except Exception as e:
            logger.error(f"Error looking up client_message_id {client_message_id} in thread {thread_id}: {e}")
        return None

    @staticmethod
    def _iso(dt) -> Optional[str]:
        """One canonical timestamp shape everywhere in the v1 API --
        second precision, UTC, literal 'Z' -- matching the doc's own
        example ("2026-10-08T09:00:04Z"). Source datetimes are a mix of
        naive (stored via datetime.utcnow()) and tz-aware (ObjectId's
        generation_time), which without this produced three different
        serializations across the same response."""
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _thread_title(doc: dict) -> str:
        first_text = next((m.get("text") for m in doc.get("messages") or [] if m.get("text")), "")
        return doc.get("custom_title") or first_text[:80] or "Untitled thread"

    def get_thread_summary(self, thread_id: str, user_id: str) -> Optional[dict]:
        """One thread's summary row, or None if it doesn't belong to this
        user -- the data backing a future GET /v1/threads/{id} read."""
        doc = self.threads_collection.find_one({"thread_id": thread_id, "user_id": user_id})
        if not doc:
            return None
        title = self._thread_title(doc)
        created_at = doc["_id"].generation_time if doc.get("_id") else None
        return {
            "id": doc["thread_id"],
            "title": title,
            "context": doc.get("context"),
            "created_at": self._iso(created_at),
            "updated_at": self._iso(doc.get("updated_at")),
            "message_count": doc.get("message_count", 0),
        }

    def thread_owner_conflict(self, thread_id: str, user_id: str) -> bool:
        """True if thread_id already exists but belongs to a different user --
        the 404 case: 'another user's thread counts as missing'."""
        doc = self.threads_collection.find_one({"thread_id": thread_id}, {"user_id": 1})
        return bool(doc and doc.get("user_id") != user_id)

    def set_thread_context(self, thread_id: str, user_id: str, species: str, sources: list) -> dict:
        """Sets a thread's species/sources, matching the Threads API v1
        proposal's PUT .../context. Returns {"changed": bool, "context": {...}}
        so the caller can tell a real change (201 + divider message) from a
        no-op (204)."""
        existing = self.get_thread_summary(thread_id, user_id) or {}
        new_context = {"species": species, "sources": sources}
        if existing.get("context") == new_context:
            return {"changed": False, "context": new_context}
        self.threads_collection.update_one(
            {"thread_id": thread_id, "user_id": user_id},
            {
                "$set": {"context": new_context, "updated_at": datetime.utcnow()},
                "$setOnInsert": {
                    "thread_id": thread_id, "user_id": user_id,
                    "messages": [], "tool_calls": [], "running_summary": "",
                },
            },
            upsert=True,
        )
        return {"changed": True, "context": new_context}

    def set_thread_title(self, thread_id: str, user_id: str, title: Optional[str]) -> None:
        """PATCH .../threads/{id}: an empty title clears the override, so the
        summary falls back to the first message's text again -- 'clearing a
        name sends the first message's text back, so it's never empty'."""
        if title:
            self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {"$set": {"custom_title": title, "updated_at": datetime.utcnow()}},
            )
        else:
            self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {"$unset": {"custom_title": ""}, "$set": {"updated_at": datetime.utcnow()}},
            )

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        """DELETE .../threads/{id}: removes only the thread record -- never
        touches the annotations/hypotheses it referenced, since those live
        in their own services entirely."""
        result = self.threads_collection.delete_one({"thread_id": thread_id, "user_id": user_id})
        return result.deleted_count > 0

    def delete_message(self, thread_id: str, user_id: str, seq: int) -> bool:
        """DELETE .../threads/{id}/messages/{message_id}: removes one message
        by its stable seq, matched directly -- not by array position, so
        deleting one message never changes any other message's id.

        Existence is checked separately before the update: the update's own
        $set (updated_at) always matches and modifies the document even when
        $pull finds nothing, so modified_count alone can't tell a real
        delete from a no-op on an already-gone or bogus seq.
        """
        exists = self.threads_collection.find_one(
            {"thread_id": thread_id, "user_id": user_id, "messages.seq": seq},
            {"_id": 1},
        )
        if not exists:
            return False
        self.threads_collection.update_one(
            {"thread_id": thread_id, "user_id": user_id},
            {"$pull": {"messages": {"seq": seq}}, "$inc": {"message_count": -1}, "$set": {"updated_at": datetime.utcnow()}},
        )
        return True

    def list_threads(self, user_id: str) -> list:
        """Thread summaries for a user, latest activity first -- the data
        backing a future GET /v1/threads read."""
        try:
            cursor = self.threads_collection.find(
                {"user_id": user_id},
                {"thread_id": 1, "messages": {"$slice": 5}, "message_count": 1, "updated_at": 1, "_id": 1, "context": 1, "custom_title": 1},
            ).sort("updated_at", -1)
            summaries = []
            for doc in cursor:
                title = self._thread_title(doc)
                created_at = doc["_id"].generation_time if doc.get("_id") else None
                summaries.append({
                    "id": doc["thread_id"],
                    "title": title,
                    "context": doc.get("context"),
                    "created_at": self._iso(created_at),
                    "updated_at": self._iso(doc.get("updated_at")),
                    "message_count": doc.get("message_count", 0),
                })
            return summaries
        except Exception as e:
            logger.error(f"Error listing threads for user {user_id}: {e}")
            return []

    def record_tool_call(self, thread_id: str, user_id: str, agent: str, resource_id: str, label: str = "") -> None:
        if not resource_id:
            return
        try:
            entry = {"agent": agent, "id": resource_id, "label": label, "created_at": datetime.utcnow()}
            result = self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {"$push": {"tool_calls": entry}, "$set": {"updated_at": datetime.utcnow()}},
            )
            if result.matched_count == 0:
                logger.warning(
                    f"No thread {thread_id} for this user — tool call not recorded"
                )
        except Exception as e:
            logger.error(f"Error recording tool call for thread {thread_id}: {e}")

    def extend_running_summary(self, thread_id: str, user_id: str, new_summary: str, through_count: int) -> None:
        """Extends the thread's running summary to cover its first
        through_count messages. Unlike the old compact_thread this replaced,
        the stored messages themselves are never touched or deleted -- every
        one stays retrievable via GET /v1/threads/{id}/messages for the life
        of the thread. This is purely an extra, lightweight aid for the
        LLM's context on a long thread, not a replacement for full storage."""
        try:
            self.threads_collection.update_one(
                {"thread_id": thread_id, "user_id": user_id},
                {"$set": {
                    "running_summary": new_summary,
                    "summary_watermark": through_count,
                    "updated_at": datetime.utcnow(),
                }},
            )
        except Exception as e:
            logger.error(f"Error extending running summary for thread {thread_id}: {e}")

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
        upload_time: datetime = None,
        summary: str = None,
        keywords: str = None,
        topics: str = None,
        metadata: dict = None,
        content_hash: str = None,
    ):
        """Add a content file record"""
        try:
            meta = metadata or {}
            content_file = {
                "user_id": user_id,
                "content_id": content_id,
                "content_type": content_type,
                "filename": filename,
                "content_hash": content_hash,
                "num_pages": num_pages,
                "url": url,
                "title": title,
                "author": author,
                "publish_date": meta.get("publish_date"),
                "file_size": meta.get("file_size"),
                "upload_time": upload_time or datetime.utcnow(),
                "summary": summary,
                "keywords": keywords,
                "topics": topics,
                "suggested_questions": meta.get("suggested_questions"),
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


def create_checkpointer(mongo_client):

    db_name = os.getenv("MONGO_CHECKPOINT_DATABASE", "ai_assistant_checkpoints")
    checkpointer = MongoDBSaver(mongo_client, db_name=db_name)
    logger.info(f"LangGraph checkpointer initialized (Mongo db={db_name})")
    return checkpointer
