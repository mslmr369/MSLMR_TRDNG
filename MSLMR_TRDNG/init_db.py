# init_db.py
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateTable
from core.config_manager import ConfigManager
from core.database import Base
from core.models.postgres_models import Price, Indicator, Signal, Trade, Backtest, Alert

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_database(config):
    """
    Initializes the database, creating all necessary tables.
    """

    db_url = config.get('POSTGRES_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set.")
        return

    engine = create_engine(db_url)

    try:
        # Create all tables defined in your models
        Base.metadata.create_all(bind=engine)
        
        # It's important to create also the individual statements, just in case there is any issue creating all at once:
        # for model in [Price, Indicator, Signal, Trade, Backtest, Alert]:
        #     try:
        #         # Generate and execute CREATE TABLE statement for each model
        #         create_table_stmt = str(CreateTable(model.__table__).compile(engine))
        #         engine.execute(create_table_stmt)
        #         logger.info(f"Table '{model.__tablename__}' created successfully.")
        #     except SQLAlchemyError as e:
        #         logger.error(f"Error creating table '{model.__tablename__}': {e}")
        logger.info("Database initialization successful.")

    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")

if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.get_config()

    init_database(config)
