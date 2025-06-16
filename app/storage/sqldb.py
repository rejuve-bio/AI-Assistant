import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid

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

class GraphInformation(Base):
    __tablename__ = "graph_information"
    
    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    graph_summary = Column(Text)
    context = Column(Text)  # JSON string for context data
    created_at = Column(DateTime, default=datetime.utcnow)

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
        """Create a new user information record"""
        with self.get_session() as db:
            user_info = UserInformation(
                user_id=user_id,
                user_question=user_question,
                memory=json.dumps(memory) if memory else None,
                context=json.dumps(context) if context else None
            )
            db.add(user_info)
            db.commit()
            db.refresh(user_info)
            return user_info
    
    def create_graph_information(self, graph_id: str = None, graph_summary: str = None,
                                context: dict = None):
        """Create a new graph information record"""
        with self.get_session() as db:
            graph_info = GraphInformation(
                graph_id=graph_id or str(uuid.uuid4()),
                graph_summary=graph_summary,
                context=json.dumps(context) if context else None
            )
            db.add(graph_info)
            db.commit()
            db.refresh(graph_info)
            return graph_info
    
    def get_user_information(self, user_id: str, limit: int = 10):
        """Retrieve user information records"""
        with self.get_session() as db:
            return db.query(UserInformation).filter(
                UserInformation.user_id == user_id
            ).order_by(UserInformation.time.desc()).limit(limit).all()
    
    def get_graph_information(self, limit: int = 10):
        """Retrieve graph information records"""
        with self.get_session() as db:
            return db.query(GraphInformation).order_by(
                GraphInformation.created_at.desc()
            ).limit(limit).all()
    
    def get_graph_by_id(self, graph_id: str):
        """Retrieve a specific graph by ID"""
        with self.get_session() as db:
            return db.query(GraphInformation).filter(
                GraphInformation.graph_id == graph_id
            ).first()
    
    def get_graphs_by_type(self, limit: int = 10):
        """Retrieve graphs ordered by creation date"""
        with self.get_session() as db:
            return db.query(GraphInformation).order_by(
                GraphInformation.created_at.desc()
            ).limit(limit).all()
    
    def update_graph_information(self, graph_id: str, graph_summary: str = None,
                                context: dict = None):
        """Update graph information"""
        with self.get_session() as db:
            graph_info = db.query(GraphInformation).filter(
                GraphInformation.graph_id == graph_id
            ).first()
            if graph_info:
                if graph_summary is not None:
                    graph_info.graph_summary = graph_summary
                if context is not None:
                    graph_info.context = json.dumps(context)
                db.commit()
                return graph_info
            return None
    
    def update_user_information(self, question_id: str, memory: dict = None, context: dict = None):
        """Update memory or context for a specific question"""
        with self.get_session() as db:
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
    
    def get_user_memory(self, user_id: str):
        """Retrieve user's memory data"""
        with self.get_session() as db:
            latest_info = db.query(UserInformation).filter(
                UserInformation.user_id == user_id,
                UserInformation.memory.isnot(None)
            ).order_by(UserInformation.time.desc()).first()
            
            if latest_info and latest_info.memory:
                return json.loads(latest_info.memory)
            return {}
    
    def get_user_context(self, user_id: str):
        """Retrieve user's latest context"""
        with self.get_session() as db:
            latest_info = db.query(UserInformation).filter(
                UserInformation.user_id == user_id,
                UserInformation.context.isnot(None)
            ).order_by(UserInformation.time.desc()).first()
            
            if latest_info and latest_info.context:
                return json.loads(latest_info.context)
            return {}
    
    def delete_graph(self, graph_id: str):
        """Delete a graph by ID"""
        with self.get_session() as db:
            graph_info = db.query(GraphInformation).filter(
                GraphInformation.graph_id == graph_id
            ).first()
            if graph_info:
                db.delete(graph_info)
                db.commit()
                return True
            return False
    
    def get_database_stats(self):
        """Get database statistics"""
        with self.get_session() as db:
            user_info_count = db.query(UserInformation).count()
            graph_info_count = db.query(GraphInformation).count()
            unique_users = db.query(UserInformation.user_id).distinct().count()
            
            return {
                "total_questions": user_info_count,
                "total_graphs": graph_info_count,
                "unique_users": unique_users,
                "database_file": DATABASE_URL
            }

# Initialize database manager
db_manager = DatabaseManager()