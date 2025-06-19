import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid
import redis
import json
from app.storage.memory_layer import MemoryManager


# SQLite database configuration
DATABASE_DIR = os.getenv('DATABASE_DIR', './data')
DATABASE_FILE = os.getenv('DATABASE_FILE', 'assistant.db')

# Ensure data directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, DATABASE_FILE)}"

# Create database engine with SQLite optimizations
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Allow multiple threads
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserInformation(Base):
    __tablename__ = "user_information"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)  # Your user identifier
    question_id = Column(String, default=lambda: str(uuid.uuid4()))
    user_question = Column(Text, nullable=False)
    time = Column(DateTime, default=datetime.utcnow)
    memory = Column(Text)  # JSON string for memory data
    context = Column(Text)  # JSON string for context data

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    def __init__(self):
        self.SessionLocal = SessionLocal
        # Create tables on initialization
        create_tables()
    
    def get_session(self):
        return self.SessionLocal()
    
    def create_user_information(self, user_id: str, user_question: str, 
                                memory: dict = None, context: dict = None):
        """Create a new user information record and keep only the 3 most recent messages per user."""
        db = self.get_session()
        try:
            # 1. Create the new record first
            user_info = UserInformation(
                user_id=user_id,
                user_question=user_question,
                memory=json.dumps(memory) if memory else None,
                context=json.dumps(context) if context else None
            )
            db.add(user_info)
            db.flush()  # This assigns the ID without committing
            
            # Store the values we need before cleanup
            result_data = {
                'id': user_info.id,
                'user_id': user_info.user_id,
                'question_id': user_info.question_id,
                'user_question': user_info.user_question,
                'time': user_info.time,
                'memory': user_info.memory,
                'context': user_info.context
            }

            # 2. Retrieve all messages for this user, ordered by descending time
            messages = db.query(UserInformation).filter(
                UserInformation.user_id == user_id
            ).order_by(UserInformation.time.desc()).all()

            # 3. If more than 3 messages, delete the oldest ones
            if len(messages) > 3:
                for msg in messages[3:]:
                    db.delete(msg)
            
            db.commit()
            
            # Create a detached object with the stored data
            result_obj = UserInformation()
            for key, value in result_data.items():
                setattr(result_obj, key, value)
            
            return result_obj
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_user_information(self, user_id: str, limit: int = 10):
        """Retrieve user information records"""
        db = self.get_session()
        try:
            return db.query(UserInformation).filter(
                UserInformation.user_id == user_id
            ).order_by(UserInformation.time.asc()).limit(limit).all()
        finally:
            db.close()
    
    def get_context_and_memory(self, user_id: str):
        """Extract user questions and memory for a user"""
        user_information = self.get_user_information(user_id)
        
        questions = []
        memories = []
        
        for record in user_information:
            # Get user question
            questions.append(record.user_question)
            
            # Extract and parse memory JSON
            if record.memory:
                try:
                    memory_data = json.loads(record.memory)
                    # Handle the nested JSON structure
                    if 'content' in memory_data:
                        content = memory_data['content']
                        if isinstance(content, str):
                            # Parse the inner JSON string
                            memory_content = json.loads(content)
                        else:
                            memory_content = content
                        memories.append(memory_content)
                    else:
                        memories.append(memory_data)
                except json.JSONDecodeError:
                    memories.append(None)
            else:
                memories.append(None)
        
        # Filter out empty lists and None values from memories
        filtered_memories = []
        for mem in memories:
            if mem is not None and mem != []:
                filtered_memories.append(mem)
        
        # If all memories are empty/None, return empty string
        if not filtered_memories:
            filtered_memories = [""]
        
        return {
            'questions': questions,
            'memories': filtered_memories
        }
    
    def update_user_information(self, question_id: str, memory: dict = None, context: dict = None):
        '''
        Update user information by question_id
        '''
        db = self.get_session()
        try:
            user_info = db.query(UserInformation).filter(
                UserInformation.question_id == question_id
            ).first()
            
            if user_info:
                if memory is not None:
                    user_info.memory = json.dumps(memory)
                if context is not None:
                    user_info.context = json.dumps(context)
                
                db.commit()
                return user_info
            return None
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    async def save_user_information(self,advanced_llm,query,user_id,context=None):
        try:
            memory_manager = MemoryManager(advanced_llm)
            memory = memory_manager.add_memory(query, user_id)
            memory_value = memory[0]['memory'] if memory and len(memory) > 0 else None
            user_info = self.create_user_information(
                user_id=user_id,
                user_question=query,
                memory=memory_value,
                context=context)
            print(f"Saved user information with question_id: {user_info.question_id}, {user_info.user_question} {user_info.memory} {user_info.context}")
            return user_info
        except Exception as e:
            print(f"Error saving user information: {e}")
            return None
       
# Initialize database manager
db_manager = DatabaseManager()

REDIS_URL=os.getenv('REDIS_URL')
class RedisGraphManager:
    """Redis storage for graphs with automatic 24-hour expiration."""
    def __init__(self, url=REDIS_URL):
        self.redis = redis.Redis.from_url(url, decode_responses=True)

    def create_graph(self, graph_id=None, graph_summary=None, context=None):
        """Create a new graph that expires in 24 hours."""
        graph_id = graph_id or str(uuid.uuid4())  
        key = f"graph:{graph_id}"

        data = {
            "graph_id": graph_id,
            "graph_summary": graph_summary or "",
            "context": context or ""
        }
        self.redis.hset(key, mapping=data)
        self.redis.expire(key, 86400)  # 24 hours in seconds
        return {"graph_id": graph_id}

    def get_graph_by_id(self, graph_id):
        """Retrieve a graph by its ID if it has not yet expired."""
        key = f"graph:{graph_id}"

        if not self.redis.exists(key):
            return None

        data = self.redis.hgetall(key)
        if not data:
            return None
        return {"graph_id": graph_id, **data}