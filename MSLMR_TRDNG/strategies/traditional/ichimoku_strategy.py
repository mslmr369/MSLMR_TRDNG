import pandas as pd
from typing import Dict, Optional
from strategies.base import BaseStrategy
from ta.trend import IchimokuIndicator

class IchimokuStrategy(BaseStrategy):
    """
    Estrategia de Trading basada en el indicador Ichimoku Kinkō Hyō
    """

    def __init__(
        self,
        window1: int = 9,
        window2: int = 26,
        window3: int = 52
    ):
        """
        Inicializa los parámetros de la estrategia Ichimoku

        :param window1: Período para la línea de conversión (Tenkan-sen)
        :param window2: Período para la línea base (Kijun-sen)
        :param window3: Período para la línea de retraso (Chikou Span) y el tramo principal B (Senkou Span B)
        """
        super().__init__(name='Ichimoku_Strategy', description='Estrategia basada en el indicador Ichimoku')
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3

    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Genera señales de trading basadas en el indicador Ichimoku

        :param data: DataFrame con datos de precios
        :return: Diccionario con señal de trading
        """
        indicator = IchimokuIndicator(
            high=data['high'],
            low=data['low'],
            window1=self.window1,
            window2=self.window2,
            window3=self.window3
        )

        ichi = indicator.ichimoku()

        # Señal de compra: Tenkan-sen cruza Kijun-sen de abajo hacia arriba
        if ichi['isa_9'][-2] < ichi['isb_26'][-2] and ichi['isa_9'][-1] > ichi['isb_26'][-1]:
            return {
                'type': 'buy',
                'price': data['close'].iloc[-1]
            }

        # Señal de venta: Tenkan-sen cruza Kijun-sen de arriba hacia abajo
        elif ichi['isa_9'][-2] > ichi['isb_26'][-2] and ichi['isa_9'][-1] < ichi['isb_26'][-1]:
            return {
                'type': 'sell',
                'price': data['close'].iloc[-1]
            }

        # Sin señal
        return None

    def get_required_indicators(self) -> List[str]:
        """
        Devuelve los indicadores requeridos por la estrategia

        :return: Lista de indicadores necesarios
        """
        return ['high', 'low']

    def validate_parameters(self) -> bool:
        """
        Valida los parámetros de la estrategia.

        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        if not all(isinstance(param, int) and param > 0 for param in [self.window1, self.window2, self.window3]):
            self.logger.error("Los parámetros window1, window2 y window3 deben ser enteros positivos.")
            return False

        if not all(self.window1 < self.window2, self.window2
