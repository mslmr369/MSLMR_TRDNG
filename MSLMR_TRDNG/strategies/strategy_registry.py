from typing import Dict, Type, Any, Optional
from .base import BaseStrategy
from core.logging_system import LoggingMixin

class StrategyRegistry(LoggingMixin):
    """
    Registro centralizado de estrategias de trading
    """
    _instance = None
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_strategy(
        cls, 
        name: str, 
        strategy_class: Type[BaseStrategy]
    ):
        """
        Registra una estrategia en el sistema
        
        :param name: Nombre de la estrategia
        :param strategy_class: Clase de la estrategia
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError("La estrategia debe heredar de BaseStrategy")
        
        cls._strategies[name] = strategy_class
        cls._instance.logger.info(f"Estrategia registrada: {name}")
    
    @classmethod
    def get_strategy(
        cls, 
        name: str
    ) -> Optional[Type[BaseStrategy]]:
        """
        Obtiene una estrategia registrada
        
        :param name: Nombre de la estrategia
        :return: Clase de la estrategia o None
        """
        strategy = cls._strategies.get(name)
        if not strategy:
            cls._instance.logger.warning(f"Estrategia no encontrada: {name}")
        return strategy
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Lista todas las estrategias registradas
        
        :return: Lista de nombres de estrategias
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def create_strategy(
        cls, 
        name: str, 
        **kwargs
    ) -> Optional[BaseStrategy]:
        """
        Crea una instancia de estrategia
        
        :param name: Nombre de la estrategia
        :param kwargs: Parámetros para inicializar la estrategia
        :return: Instancia de estrategia
        """
        strategy_class = cls.get_strategy(name)
        if strategy_class:
            return strategy_class(**kwargs)
        return None
    
    @classmethod
    def remove_strategy(cls, name: str):
        """
        Elimina una estrategia del registro
        
        :param name: Nombre de la estrategia a eliminar
        """
        if name in cls._strategies:
            del cls._strategies[name]
            cls._instance.logger.info(f"Estrategia eliminada: {name}")
