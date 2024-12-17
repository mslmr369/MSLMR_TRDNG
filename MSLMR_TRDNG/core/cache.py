import redis
import pickle
import hashlib
import time
from functools import wraps
from typing import Any, Callable, Optional

class DistributedCache:
    """
    Sistema de caché distribuida con Redis
    Soporta serialización, expiración y estrategias de invalidación
    """
    def __init__(
        self, 
        redis_url: str = 'redis://localhost:6379/0', 
        default_timeout: int = 3600
    ):
        """
        Inicializa el cliente de Redis
        
        :param redis_url: URL de conexión a Redis
        :param default_timeout: Tiempo de expiración por defecto (segundos)
        """
        self._client = redis.from_url(redis_url, decode_responses=False)
        self._default_timeout = default_timeout

    def _generate_key(self, func: Callable, *args, **kwargs) -> str:
        """
        Genera una clave única basada en la función y sus argumentos
        
        :param func: Función a cachear
        :param args: Argumentos posicionales
        :param kwargs: Argumentos de palabra clave
        :return: Clave de caché hash
        """
        key_parts = [
            func.__module__,
            func.__name__,
            str(args),
            str(sorted(kwargs.items()))
        ]
        return hashlib.md5('_'.join(map(str, key_parts)).encode()).hexdigest()

    def cached(
        self, 
        timeout: Optional[int] = None, 
        key_prefix: str = 'trading_cache:'
    ):
        """
        Decorador para cachear resultados de funciones
        
        :param timeout: Tiempo de expiración personalizado
        :param key_prefix: Prefijo para la clave de caché
        :return: Decorador
        """
        timeout = timeout or self._default_timeout

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave única
                cache_key = f"{key_prefix}{self._generate_key(func, *args, **kwargs)}"
                
                try:
                    # Intentar obtener de caché
                    cached_result = self._client.get(cache_key)
                    
                    if cached_result:
                        return pickle.loads(cached_result)
                    
                    # Ejecutar función si no está en caché
                    result = func(*args, **kwargs)
                    
                    # Almacenar en caché
                    serialized_result = pickle.dumps(result)
                    self._client.setex(cache_key, timeout, serialized_result)
                    
                    return result
                
                except Exception as e:
                    # Fallback: ejecutar función sin caché
                    print(f"Cache error: {e}")
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator

    def invalidate(self, pattern: str = 'trading_cache:*'):
        """
        Invalidar caché por patrón
        
        :param pattern: Patrón de claves a invalidar
        """
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)

    def get_cache_stats(self) -> dict:
        """
        Obtiene estadísticas del caché
        
        :return: Diccionario con métricas de caché
        """
        info = self._client.info()
        return {
            'total_keys': info.get('db0', {}).get('keys', 0),
            'memory_used': info.get('used_memory_human', '0B'),
            'hit_ratio': self._calculate_hit_ratio()
        }

    def _calculate_hit_ratio(self) -> float:
        """
        Calcula el ratio de aciertos del caché
        
        :return: Ratio de aciertos
        """
        try:
            hits = self._client.get('cache_hits') or 0
            misses = self._client.get('cache_misses') or 0
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except:
            return 0.0

    def increment_hit_counter(self, hit: bool = True):
        """
        Incrementa contadores de aciertos/fallos
        
        :param hit: Si es un acierto de caché
        """
        try:
            if hit:
                self._client.incr('cache_hits')
            else:
                self._client.incr('cache_misses')
        except:
            pass

class CacheManager:
    """
    Gestor centralizado de caché con múltiples estrategias
    """
    def __init__(
        self, 
        redis_url: str = 'redis://localhost:6379/0',
        default_timeout: int = 3600
    ):
        self._distributed_cache = DistributedCache(redis_url, default_timeout)
        self._strategies = {}

    def register_cache_strategy(
        self, 
        name: str, 
        strategy: Callable[[Any], str]
    ):
        """
        Registra estrategia personalizada de generación de claves
        
        :param name: Nombre de la estrategia
        :param strategy: Función para generar clave
        """
        self._strategies[name] = strategy

    def cache_with_strategy(
        self, 
        strategy_name: str, 
        timeout: Optional[int] = None
    ):
        """
        Decorador para usar estrategia de caché personalizada
        
        :param strategy_name: Nombre de la estrategia
        :param timeout: Tiempo de expiración
        """
        def decorator(func: Callable):
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                raise ValueError(f"Estrategia {strategy_name} no encontrada")
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = strategy(*args, **kwargs)
                
                # Lógica de caché similar a DistributedCache
                cached_result = self._distributed_cache._client.get(cache_key)
                
                if cached_result:
                    return pickle.loads(cached_result)
                
                result = func(*args, **kwargs)
                
                serialized_result = pickle.dumps(result)
                self._distributed_cache._client.setex(
                    cache_key, 
                    timeout or self._distributed_cache._default_timeout, 
                    serialized_result
                )
                
                return result
            
            return wrapper
        return decorator

# Ejemplo de uso de estrategias personalizadas
def trading_data_strategy(symbol: str, timeframe: str) -> str:
    """
    Estrategia de caché personalizada para datos de trading
    """
    return f"trading_data:{symbol}:{timeframe}"

# Añadir al final del archivo core/cache.py

class CacheDecorator:
    """
    Decorador de caché más versátil con opciones adicionales
    """
    def __init__(
        self,
        cache_client: DistributedCache,
        timeout: Optional[int] = None,
        key_prefix: str = 'trading_cache:',
        fallback_strategy: Optional[Callable] = None
    ):
        self.cache = cache_client
        self.timeout = timeout or 3600
        self.key_prefix = key_prefix
        self.fallback_strategy = fallback_strategy

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generar clave de caché
            cache_key = f"{self.key_prefix}{self.cache._generate_key(func, *args, **kwargs)}"

            try:
                # Intentar obtener de caché
                cached_result = self.cache._client.get(cache_key)

                if cached_result:
                    self.cache.increment_hit_counter(hit=True)
                    return pickle.loads(cached_result)

                # Caché miss
                self.cache.increment_hit_counter(hit=False)

                # Ejecutar función
                result = func(*args, **kwargs)

                # Aplicar estrategia de fallback si existe
                if self.fallback_strategy:
                    result = self.fallback_strategy(result)

                # Almacenar en caché
                serialized_result = pickle.dumps(result)
                self.cache._client.setex(cache_key, self.timeout, serialized_result)

                return result

            except Exception as e:
                # Logging o manejo de errores
                print(f"Cache error for {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper

class CacheFactory:
    """
    Fábrica para crear instancias de caché con configuraciones predefinidas
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._caches = {}
        return cls._instance

    def get_cache(
        self,
        name: str = 'default',
        redis_url: str = 'redis://localhost:6379/0',
        default_timeout: int = 3600
    ) -> DistributedCache:
        """
        Obtiene o crea una instancia de caché

        :param name: Nombre de la instancia de caché
        :param redis_url: URL de conexión de Redis
        :param default_timeout: Tiempo de expiración por defecto
        :return: Instancia de DistributedCache
        """
        if name not in self._caches:
            self._caches[name] = DistributedCache(redis_url, default_timeout)
        return self._caches[name]

    def create_decorator(
        self,
        cache_name: str = 'default',
        timeout: Optional[int] = None,
        key_prefix: str = 'trading_cache:',
        fallback_strategy: Optional[Callable] = None
    ) -> CacheDecorator:
        """
        Crea un decorador de caché

        :param cache_name: Nombre de la instancia de caché
        :param timeout: Tiempo de expiración
        :param key_prefix: Prefijo para la clave de caché
        :param fallback_strategy: Estrategia de respaldo
        :return: Instancia de CacheDecorator
        """
        cache = self.get_cache(cache_name)
        return CacheDecorator(
            cache_client=cache,
            timeout=timeout,
            key_prefix=key_prefix,
            fallback_strategy=fallback_strategy
        )

# Ejemplo de uso avanzado
def main():
    # Inicializar fábrica de caché
    cache_factory = CacheFactory()

    # Obtener caché por nombre
    trading_cache = cache_factory.get_cache('trading')

    # Crear decorador de caché
    @cache_factory.create_decorator(
        cache_name='trading',
        timeout=1800,  # 30 minutos
        key_prefix='trading_data:',
        fallback_strategy=lambda x: x  # Estrategia de respaldo opcional
    )
    def fetch_trading_data(symbol: str, timeframe: str):
        # Simular obtención de datos
        print(f"Fetching data for {symbol} on {timeframe}")
        return {"symbol": symbol, "data": [1, 2, 3, 4, 5]}

    # Primer llamado: busca en caché
    print(fetch_trading_data('BTC/USDT', '1h'))

    # Segundo llamado: debe usar caché
    print(fetch_trading_data('BTC/USDT', '1h'))

    # Obtener estadísticas de caché
    print(trading_cache.get_cache_stats())

if __name__ == "__main__":
    main()
