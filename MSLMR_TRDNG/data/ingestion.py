import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from core.cache import CacheDecorator, CacheFactory
from core.logging_system import log_method, LoggingMixin
import logging
import asyncio

class DataIngestionManager(LoggingMixin):
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

    @CacheDecorator(cache_client=CacheFactory().get_cache('market_data'), key_prefix='historical_data')
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        # ... (Rest of the code remains the same) ...

class AsyncDataIngestionManager(DataIngestionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exchange = getattr(ccxt.async_support, self.exchange_id)()

    async def async_fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos de un símbolo de forma asíncrona

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
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

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

    async def concurrent_data_fetch(
        self,
        limit: int = 1000
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Obtiene datos en paralelo para múltiples símbolos y timeframes

        :param limit: Número máximo de candeleros por símbolo/timeframe
        :return: Diccionario anidado de DataFrames
        """
        all_data = {}

        async def fetch_and_store(symbol, timeframe):
            try:
                df = await self.async_fetch_historical_data(
                    symbol, timeframe, limit
                )
                if symbol not in all_data:
                    all_data[symbol] = {}
                all_data[symbol][timeframe] = df
            except Exception as e:
                self.logger.warning(
                    f"Error fetching data for {symbol} on {timeframe}: {e}"
                )

        tasks = [
            fetch_and_store(symbol, timeframe)
            for symbol in self.symbols
            for timeframe in self.timeframes
        ]

        await asyncio.gather(*tasks)
        return all_data
