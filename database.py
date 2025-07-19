"""
Database configuration and initialization
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncio
import logging

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Async database setup
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create async session
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Flask app for synchronous operations
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "trading-system-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize Flask-SQLAlchemy
db.init_app(app)

async def init_database():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from models import *
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("✅ Database tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

async def get_db_session():
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_sync_db_session():
    """Get synchronous database session for Flask operations"""
    with app.app_context():
        return db.session

# Database utility functions
async def execute_query(query: str, params: dict = None):
    """Execute a raw SQL query"""
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(query, params or {})
            await session.commit()
            return result
        except Exception as e:
            await session.rollback()
            logger.error(f"Query execution failed: {e}")
            raise

async def check_database_connection():
    """Check if database connection is working"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
