import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.logging_system import LoggingMixin
from core.database import Base  # Assuming you have SQLAlchemy models defined in a separate file

class DatabaseInteractor(LoggingMixin):
    """
    Handles interactions with the database.
    """

    def __init__(self, db_url: str):
        """
        Initializes the database interactor.

        :param db_url: Database connection URL.
        """
        try:
            self.engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800
            )
            self.SessionLocal = scoped_session(sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            ))
        except SQLAlchemyError as e:
            self.logger.error(f"Database connection error: {e}")
            raise

    def create_tables(self):
        """
        Creates all tables defined in the models.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Tables created successfully.")
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def get_session(self):
        """
        Provides a new database session.

        :return: A new SQLAlchemy session.
        """
        return self.SessionLocal()

    def store_data(self, data: List[Any]):
        """
        Stores a list of data objects into the database.

        :param data: List of data objects to store.
        """
        session = self.get_session()
        try:
            for item in data:
                session.merge(item)  # Use merge to avoid duplicates
            session.commit()
            self.logger.info("Data stored successfully.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error storing data: {e}")
            raise
        finally:
            session.close()

    def retrieve_data(
        self,
        model,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves data from the database based on model, symbol, and timeframe.

        :param model: The SQLAlchemy model to query.
        :param symbol: Symbol of the asset.
        :param timeframe: Timeframe of the data.
        :param start_date: Optional start date for the query.
        :param end_date: Optional end date for the query.
        :return: DataFrame with the retrieved data or None if an error occurred.
        """
        session = self.get_session()
        try:
            query = session.query(model).filter_by(symbol=symbol, timeframe=timeframe)
            if start_date:
                query = query.filter(model.timestamp >= start_date)
            if end_date:
                query = query.filter(model.timestamp <= end_date)
            query = query.order_by(model.timestamp)
            data = query.all()
            if not data:
                return None

            # Dynamically construct DataFrame based on model attributes
            columns = {column.name: [] for column in model.__table__.columns}
            for row in data:
                for column_name in columns.keys():
                    columns[column_name].append(getattr(row, column_name))

            df = pd.DataFrame(columns)
            if not df.empty:
                df.set_index('timestamp', inplace=True)

            return df

        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving data: {e}")
            return None
        finally:
            session.close()
