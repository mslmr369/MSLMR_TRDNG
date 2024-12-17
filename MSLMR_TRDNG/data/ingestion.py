import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from core.cache import CacheFactory
from core.logging_system import log_method
import logging

class DataIngestionManager:
    def __init__(
        self, 
        exchange_id: str = 'binance', 
        symbols: List[str] = ['BTC/USDT'],
        timeframes: List[str] = ['1h', '4h', '1d']
    ):
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Inicializar exchange
        self.exchange = getattr(ccxt, exchange_id)()
        
        # Inicializar caché
        self.cache_factory = CacheFactory()
        self.data_cache = self.cache_factory.get_cache('market_data')
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    @log_method()
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos de un símbolo
        
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        :param limit: Número máximo de candeleros
        :return: DataFrame con datos históricos
        """
        # Buscar en caché primero
        cache_key = f"{symbol}_{timeframe}_historical"
        cached_data = self.data_cache._client.get(cache_key)
        
        if cached_data:
            return pd.read_json(cached_data)
        
        try:
            # Obtener datos del exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Almacenar en caché
            self.data_cache._client.set(
                cache_key, 
                df.to_json(), 
                ex=3600  # 1 hora de caché
            )
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise

class MultiSymbolDataIngestion:
    def __init__(
        self, 
        exchange_id: str = 'binance',
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ):
        self.exchange_id = exchange_id
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.timeframes = timeframes or ['1h', '4h', '1d']
        
        self.ingestion_manager = DataIngestionManager(
            exchange_id=exchange_id,
            symbols=self.symbols,
            timeframes=self.timeframes
        )
        
        self.logger = logging.getLogger(__name__)
    
    def fetch_multi_symbol_data(
        self, 
        limit: int = 1000
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Obtiene datos para múltiples símbolos y timeframes
        
        :param limit: Número máximo de candeleros por símbolo/timeframe
        :return: Diccionario anidado de DataFrames
        """
        all_data = {}
        
        for symbol in self.symbols:
            symbol_data = {}
            
            for timeframe in self.timeframes:
                try:
                    df = self.ingestion_manager.fetch_historical_data(
                        symbol, timeframe, limit
                    )
                    symbol_data[timeframe] = df
                except Exception as e:
                    self.logger.warning(
                        f"Error fetching data for {symbol} on {timeframe}: {e}"
                    )
            
            all_data[symbol] = symbol_data
        
        return all_data

    def concurrent_data_fetch(
        self, 
        limit: int = 1000
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Obtiene datos en paralelo para múltiples símbolos y timeframes
        
        :param limit: Número máximo de candeleros por símbolo/timeframe
        :return: Diccionario anidado de DataFrames
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_data = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Preparar futures
            futures = {
                executor.submit(
                    self.ingestion_manager.fetch_historical_data, 
                    symbol, 
                    timeframe, 
                    limit
                ): (symbol, timeframe)
                for symbol in self.symbols
                for timeframe in self.timeframes
            }
            
            # Procesar resultados
            for future in as_completed(futures):
                symbol, timeframe = futures[future]
                
                try:
                    df = future.result()
                    
                    # Estructura de datos anidada
                    if symbol not in all_data:
                        all_data[symbol] = {}
                    all_data[symbol][timeframe] = df
                
                except Exception as e:
                    self.logger.warning(
                        f"Error en fetch concurrente para {symbol} {timeframe}: {e}"
                    )
        
        return all_data
