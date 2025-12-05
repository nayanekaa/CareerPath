"""SQLAlchemy models for storing psychometric scores and ratings."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class PsychometricScore(Base):
    """Store psychometric test responses and computed profiles."""
    __tablename__ = "psychometric_scores"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    test_responses = Column(JSON, nullable=False)  # Dict of question_id -> response
    archetype = Column(String(255), nullable=True)
    learning_style = Column(String(255), nullable=True)
    work_style = Column(String(255), nullable=True)
    traits = Column(JSON, nullable=True)  # List of trait strings
    description = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RoadmapRating(Base):
    """Store grader ratings for generated career roadmaps."""
    __tablename__ = "roadmap_ratings"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    roadmap_id = Column(String(255), nullable=False)  # ID of the generated roadmap
    grader_id = Column(String(255), nullable=False)  # ID of the person grading
    
    # Rating dimensions (1-5 scale)
    usefulness = Column(Integer, nullable=True)  # How useful is this roadmap?
    clarity = Column(Integer, nullable=True)  # How clear are the instructions?
    factuality = Column(Integer, nullable=True)  # How accurate is the information?
    actionability = Column(Integer, nullable=True)  # How actionable are the steps?
    
    # Overall rating and comments
    overall_rating = Column(Integer, nullable=True)  # 1-5 overall score
    comments = Column(String(2000), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database initialization
DATABASE_URL = "sqlite:///./backend/careerpath.db"


def get_engine():
    """Get SQLAlchemy engine."""
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def get_session_maker():
    """Get SessionLocal factory."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
