from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from core.config_manager import ConfigManager
import logging

logger = logging.getLogger(__name__)

# Database Configuration
config_manager = ConfigManager()

# Deberíamos tener ya las variables de entorno cargadas desde el main
db_config = config_manager.get_config()
DATABASE_URL = db_config.get('POSTGRES_URL')

print("DATABASE_URL:", DATABASE_URL)  # Debug: Imprimir la URL
engine = None  # Inicializar engine como None

SessionLocal = None # y la sesión también

def initialize_database():
    global engine
    global SessionLocal
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800
    )

    SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


# Declarative base for models
Base = declarative_base()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
