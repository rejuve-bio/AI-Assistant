from datetime import datetime
from app.storage.sql_redis_storage import db_manager, UserInformation
import uuid


class HistoryManager:
    def create_history(self, user_id, user_message, assistant_answer):
        # Generate a unique query_id
        query_id = str(uuid.uuid4())
        now = datetime.now()
        # Create a new UserInformation record
        db_manager.create_user_information(
            user_id=user_id,
            user_question=user_message,
            memory=None,
            context=None,
        )
        # Update the assistant_answer and question_id for the latest record
        db = db_manager.get_session()
        try:
            record = (
                db.query(UserInformation)
                .filter_by(user_id=user_id)
                .order_by(UserInformation.time.desc())
                .first()
            )
            if record:
                record.assistant_answer = assistant_answer
                record.question_id = query_id
                db.commit()
        finally:
            db.close()
        return query_id

    def retrieve_user_history(self, user_id, limit=5):
        records = db_manager.get_user_information(user_id, limit=limit)
        user_id_str = str(user_id)
        history = []
        for record in records:
            history.append(
                {
                    "query_id": record.question_id,
                    "user": record.user_question,
                    "assistant answer": record.assistant_answer,
                    "time": record.time.isoformat() if record.time else None,
                }
            )
        return {user_id_str: history}

    def get_entry_by_query_id(self, user_id, query_id):
        db = db_manager.get_session()
        try:
            record = (
                db.query(UserInformation)
                .filter_by(user_id=user_id, question_id=query_id)
                .first()
            )
            if record:
                return {
                    "query_id": record.question_id,
                    "user": record.user_question,
                    "assistant answer": record.assistant_answer,
                    "time": record.time.isoformat() if record.time else None,
                }
            return None
        finally:
            db.close()

    def clear_user_history(self, user_id):
        db = db_manager.get_session()
        try:
            records = db.query(UserInformation).filter_by(user_id=user_id).all()
            for record in records:
                db.delete(record)
            db.commit()
        finally:
            db.close()
