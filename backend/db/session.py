# backend/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.core.config import DATABASE_URL

# Create a SQLAlchemy engine instance.
# The engine is the starting point for any SQLAlchemy application.
# It's the 'home base' for the actual database and its DBAPI.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a SessionLocal class. Each instance of SessionLocal will be a database session.
# The session is the primary interface for all database operations.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
