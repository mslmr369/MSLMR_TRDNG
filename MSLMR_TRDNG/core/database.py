from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import redis
import logging
from typing import Dict, Any

# Base declarativa para modelos de SQLAlchemy
Base = declarative_base()
metadata = MetaData()

class DatabaseManager:
    """
    Gestiona conexiones a múltiples bases de datos
    """
    _instance = None
    
    def __new__(cls, config=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config=None):
        if self._initialized:
            return
        
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Conexión PostgreSQL
        try:
            self.postgres_engine = create_engine(
                self.config.POSTGRES_URL,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800
            )
            
            # Crear sesión de base de datos
            self.SessionLocal = scoped_session(
                sessionmaker(
                    bind=self.postgres_engine, 
                    autocommit=False, 
                    autoflush=False
                )
            )
        except Exception as e:
            self.logger.error(f"Error conectando a PostgreSQL: {e}")
            raise
        
        # Conexión Redis
        try:
            self.redis_client = redis.from_url(
                self.config.REDIS_URL,
                decode_responses=True
            )
            self.logger.info("Conexiones de base de datos inicializadas")
        except Exception as e:
            self.logger.error(f"Error conectando a Redis: {e}")
            raise
        
        self._initialized = True
    
    def get_postgres_session(self):
        """Obtiene una sesión de base de datos PostgreSQL"""
        return self.SessionLocal()
    
    def get_redis_client(self):
        """Obtiene cliente Redis"""
        return self.redis_client
    
    def create_tables(self):
        """Crea todas las tablas definidas"""
        Base.metadata.create_all(bind=self.postgres_engine)
        self.logger.info("Tablas de base de datos creadas")
    
    def drop_tables(self):
        """Elimina todas las tablas (usar con precaución)"""
        Base.metadata.drop_all(bind=self.postgres_engine)
        self.logger.warning("Todas las tablas han sido eliminadas")

def initialize_databases(config):
    """
    Función de inicialización de bases de datos
    """
    try:
        # Inicializar gestor de bases de datos
        db_manager = DatabaseManager(config)
        
        # Crear tablas
        db_manager.create_tables()
        
        return db_manager
    except Exception as e:
        logging.error(f"Error inicializando bases de datos: {e}")
        raise
