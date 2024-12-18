from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, List, Any
from strategies.risk_management import RiskManager

class BaseStrategy(ABC):
    """
    Clase base abstracta para estrategias de trading
    Define la interfaz común para todas las estrategias
    """

    def __init__(
        self,
        name: str = 'BaseStrategy',
        description: str = 'Estrategia base de trading',
        risk_manager: Optional[RiskManager] = None  # Add RiskManager here
    ):
        """
        Inicializa una estrategia base

        :param name: Nombre de la estrategia
        :param description: Descripción de la estrategia
        :param risk_manager: Instancia de RiskManager
        """
        self.name = name
        self.description = description
        self.parameters = {}
        self.risk_manager = risk_manager or RiskManager() # Use default if not provided

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Genera señales de trading basadas en datos

        :param data: DataFrame con datos históricos
        :return: Diccionario con señal de trading o None
        """
        pass

    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        Devuelve los indicadores requeridos por la estrategia

        :return: Lista de indicadores necesarios
        """
        pass

    def set_parameters(self, **kwargs):
        """
        Establece parámetros de la estrategia

        :param kwargs: Parámetros a establecer
        """
        for key, value in kwargs.items():
            self.parameters[key] = value

    def get_parameters(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros actuales de la estrategia

        :return: Diccionario de parámetros
        """
        return self.parameters

    def validate_parameters(self) -> bool:
        """
        Valida los parámetros de la estrategia

        :return: Booleano indicando si los parámetros son válidos
        """
        # Implementación base, puede ser sobrescrita
        return True

    def reset(self):
        """
        Reinicia la estrategia a su estado inicial
        """
        self.parameters.clear()
