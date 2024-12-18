from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import redis
import logging
from typing import Dict, Any

# Base declarativa para modelos de SQLAlchemy
Base = declarative_base()
metadata = MetaData()
