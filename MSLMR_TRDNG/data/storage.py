import os
import json
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Base para modelos SQLAlchemy
Base = declarative_base()

class MarketDataModel(Base):
    """
    Modelo de almacenamiento de datos de mercado
    """
    __tablename__ = 'market_data'

    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    timeframe = sa.Column(sa.String(10), nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    open = sa.Column(sa.Float, nullable=False)
    high = sa.Column(sa.Float, nullable=False)
    low = sa.Column(sa.Float, nullable=False)
    close = sa.Column(sa.Float, nullable=False)
    volume = sa.Column(sa.Float, nullable=False)
    
    # Índices para mejorar rendimiento
    __table_args__ = (
        sa.Index('idx_symbol_timeframe', 'symbol', 'timeframe'),
        sa.Index('idx_timestamp', 'timestamp')
    )

class IndicatorDataModel(Base):
    """
    Modelo de almacenamiento de indicadores
    """
    __tablename__ = 'indicators'

    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    timeframe = sa.Column(sa.String(10), nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    
    # Indicadores genéricos
    rsi = sa.Column(sa.Float)
    macd_line = sa.Column(sa.Float)
    macd_signal = sa.Column(sa.Float)
    macd_histogram = sa.Column(sa.Float)
    
    # Índices
    __table_args__ = (
        sa.Index('idx_symbol_timeframe', 'symbol', 'timeframe'),
        sa.Index('idx_timestamp', 'timestamp')
    )

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
        if database_url:
            try:
                self.engine = sa.create_engine(database_url)
                Base.metadata.create_all(self.engine)
                self.Session = sessionmaker(bind=self.engine)
            except Exception as e:
                self.logger.error(f"Error configurando base de datos: {e}")
                self.engine = None
                self.Session = None

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
        if storage_type == 'database' and self.Session:
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
        if not self.Session:
            self.logger.error("No se ha configurado conexión a base de datos")
            return

        session = self.Session()
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
                        session.merge(market_data)

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
                            session.merge(indicator_data)

            session.commit()
            self.logger.info("Datos almacenados exitosamente en base de datos")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error almacenando datos en base de datos: {e}")
        finally:
            session.close()

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
        if source == 'database' and self.Session:
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
        if not self.Session:
            self.logger.error("No se ha configurado conexión a base de datos")
            return None

        session = self.Session()
        try:
            # Consulta base de datos
            query = session.query(MarketDataModel).filter_by(
                symbol=symbol, timeframe=timeframe
            )
            
            # Filtrar por fechas si se proporcionan
            if start_date:
                query = query.filter(MarketDataModel.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketDataModel.timestamp <= end_date)
            
            # Convertir a DataFrame
            data = query.all()
            if not data:
                return None

            df = pd.DataFrame([
                {
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                } for row in data
            ])
            
            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Error recuperando datos: {e}")
            return None
        finally:
            session.close()

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

# Ejemplo de uso
def main():
    # Configurar gestor de almacenamiento
    storage_manager = DataStorageManager(
        database_url='postgresql://user:pass@localhost/trading_db',
        storage_path='./data/market_data'
    )

    # Simular datos de mercado
    from data.ingestion import MultiSymbolDataIngestion
    data_ingestion = MultiSymbolDataIngestion()
    market_data = data_ingestion.concurrent_data_fetch()

    # Almacenar datos en base de datos
    storage_manager.store_market_data(market_data, storage_type='database')

    # Recuperar datos
    retrieved_data = storage_manager.retrieve_market_data(
        symbol='BTC/USDT', 
        timeframe='1h',
        start_date=datetime.now() - pd.Timedelta(days=7)
    )

    print(retrieved_data)

if __name__ == "__main__":
    main()
