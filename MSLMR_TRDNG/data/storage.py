# Correctly using DatabaseInteractor and removing session management
import os
import json
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, List, Optional
import logging
from datetime import datetime
from core.database_interactor import DatabaseInteractor
from core.database import Base
from data.models import MarketDataModel, IndicatorDataModel

class DataStorageManager:
    """
    Gestor de almacenamiento de datos de mercado
    Soporta múltiples backends: PostgreSQL, CSV, JSON
    """
    def __init__(
        self, 
        database_url: Optional[str] = None,
        storage_path: str = './data/storage'
    ):
        """
        Inicializa gestor de almacenamiento
        
        :param database_url: URL de conexión a base de datos
        :param storage_path: Ruta para almacenamiento de archivos
        """
        # Configurar logger
        self.logger = logging.getLogger(__name__)
        
        # Configurar almacenamiento de archivos
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Configurar base de datos si se proporciona URL
        self.db_interactor = None
        if database_url:
            try:
                self.db_interactor = DatabaseInteractor(database_url)
                self.db_interactor.create_tables()
                self.logger.info("Database tables created successfully.")
            except Exception as e:
                self.logger.error(f"Error setting up database: {e}")
                self.db_interactor = None

    def store_market_data(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]],
        storage_type: str = 'database'
    ):
        """
        Almacena datos de mercado
        
        :param data: Datos de mercado
        :param storage_type: Tipo de almacenamiento ('database', 'csv', 'json')
        """
        if storage_type == 'database' and self.db_interactor:
            self._store_to_database(data)
        elif storage_type == 'csv':
            self._store_to_csv(data)
        elif storage_type == 'json':
            self._store_to_json(data)
        else:
            self.logger.warning(f"Tipo de almacenamiento no soportado: {storage_type}")

    def _store_to_database(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Almacena datos en base de datos PostgreSQL
        """
        if not self.db_interactor:
            self.logger.error("DatabaseInteractor is not initialized.")
            return

        try:
            for symbol, timeframe_data in data.items():
                for timeframe, df in timeframe_data.items():
                    for _, row in df.iterrows():
                        # Guardar datos de mercado
                        market_data = MarketDataModel(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=row.name,
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume']
                        )
                        self.db_interactor.store_data([market_data])

                        # Guardar indicadores si existen
                        if 'rsi' in row:
                            indicator_data = IndicatorDataModel(
                                symbol=symbol,
                                timeframe=timeframe,
                                timestamp=row.name,
                                rsi=row['rsi'],
                                macd_line=row.get('macd_line'),
                                macd_signal=row.get('signal_line'),
                                macd_histogram=row.get('histogram')
                            )
                            self.db_interactor.store_data([indicator_data])

            self.logger.info("Datos almacenados exitosamente en base de datos")
        except Exception as e:
            self.logger.error(f"Error almacenando datos en base de datos: {e}")

    def _store_to_csv(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Almacena datos en archivos CSV
        """
        for symbol, timeframe_data in data.items():
            for timeframe, df in timeframe_data.items():
                filename = os.path.join(
                    self.storage_path, 
                    f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                df.to_csv(filename)
                self.logger.info(f"Datos almacenados en {filename}")

    def _store_to_json(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Almacena datos en archivos JSON
        """
        for symbol, timeframe_data in data.items():
            for timeframe, df in timeframe_data.items():
                filename = os.path.join(
                    self.storage_path, 
                    f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
                )
                df.to_json(filename)
                self.logger.info(f"Datos almacenados en {filename}")

    def retrieve_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'database'
    ) -> Optional[pd.DataFrame]:
        """
        Recupera datos de mercado
        
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        :param start_date: Fecha de inicio
        :param end_date: Fecha de fin
        :param source: Fuente de datos
        :return: DataFrame con datos de mercado
        """
        if source == 'database' and self.db_interactor:
            return self._retrieve_from_database(
                symbol, timeframe, start_date, end_date
            )
        elif source == 'csv':
            return self._retrieve_from_csv(
                symbol, timeframe, start_date, end_date
            )
        elif source == 'json':
            return self._retrieve_from_json(
                symbol, timeframe, start_date, end_date
            )
        else:
            self.logger.warning(f"Fuente de datos no soportada: {source}")
            return None

    def _retrieve_from_database(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Recupera datos desde base de datos PostgreSQL
        """
        if not self.db_interactor:
            self.logger.error("DatabaseInteractor is not initialized.")
            return None

        try:
            df = self.db_interactor.retrieve_data(
                MarketDataModel, symbol, timeframe, start_date, end_date
            )
            return df

        except Exception as e:
            self.logger.error(f"Error retrieving data from database: {e}")
            return None

    def _retrieve_from_csv(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Recupera datos desde archivos CSV
        """
        try:
            filename = os.path.join(
                self.storage_path, 
                f"{symbol}_{timeframe}*.csv"
            )
            files = sorted(glob.glob(filename))
            
            if not files:
                self.logger.warning(f"No se encontraron archivos para {symbol}")
                return None

            # Leer último archivo
            df = pd.read_csv(files[-1], index_col='timestamp', parse_dates=True)
            
            # Filtrar por fechas
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df

        except Exception as e:
            self.logger.error(f"Error recuperando datos CSV: {e}")
            return None

    def _retrieve_from_json(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Recupera datos desde archivos JSON
        """
        try:
            filename = os.path.join(
                self.storage_path, 
                f"{symbol}_{timeframe}*.json"
            )
            files = sorted(glob.glob(filename))
            
            if not files:
                self.logger.warning(f"No se encontraron archivos para {symbol}")
                return None

            # Leer último archivo
            df = pd.read_json(files[-1], orient='index', convert_axes=True)
            df.index = pd.to_datetime(df.index)
            
            # Filtrar por fechas
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df

        except Exception as e:
            self.logger.error(f"Error recuperando datos JSON: {e}")
            return None
