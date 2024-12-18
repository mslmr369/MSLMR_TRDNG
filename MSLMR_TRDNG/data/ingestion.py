import ccxt.async_support as ccxt
import pandas as pd
from typing import Dict, List, Optional
from core.cache import CacheDecorator, CacheFactory
from core.logging_system import log_method, LoggingMixin
import asyncio
import pickle

class AsyncDataIngestionManager(LoggingMixin):
    def __init__(self, exchange_id: str, symbols: List[str], timeframes: List[str]):
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.timeframes = timeframes
        self.cache_factory = CacheFactory()
        self.data_cache = self.cache_factory.get_cache('market_data')

        # Asynchronously initialize the exchange
        self.exchange = getattr(ccxt, self.exchange_id)()

    async def async_fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        cache_key = f"{self.exchange_id}:{symbol}:{timeframe}:{limit}"
        cached_data = self.data_cache._client.get(cache_key)

        if cached_data:
            self.logger.info(f"Cache hit for {cache_key}")
            self.data_cache.increment_hit_counter(hit=True)
            df = pd.DataFrame(pickle.loads(cached_data))  # Unpickle the data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df

        self.data_cache.increment_hit_counter(hit=False)

        try:
            self.logger.info(f"Fetching OHLCV data for {symbol} on {timeframe} from exchange")
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert the DataFrame to bytes using pickle
            data_to_cache = pickle.dumps(df.to_dict(orient='records'))

            # Cache the result with an expiration time of 1 hour (3600 seconds)
            self.data_cache._client.setex(cache_key, 3600, data_to_cache)

            return df

        except ccxt.NetworkError as e:
            self.logger.error(f"Network error fetching {symbol} {timeframe}: {e}")
            raise
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error fetching {symbol} {timeframe}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {symbol} {timeframe}: {e}")
            raise

    async def close_connections(self):
        await self.exchange.close()

    async def concurrent_data_fetch(
        self,
        limit: int = 1000
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        all_data = {}

        async def fetch_and_store(symbol, timeframe):
            try:
                df = await self.async_fetch_ohlcv(symbol, timeframe, limit)
                if symbol not in all_data:
                    all_data[symbol] = {}
                all_data[symbol][timeframe] = df
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol} on {timeframe}: {e}")

        tasks = [fetch_and_store(symbol, timeframe) for symbol in self.symbols for timeframe in self.timeframes]
        await asyncio.gather(*tasks)

        return all_data
