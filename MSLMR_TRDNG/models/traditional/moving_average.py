import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from strategies.base import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    """
    Estrategia de Trading basada en Medias Móviles
    Soporta múltiples tipos de medias móviles y configuraciones
    """
    
    def __init__(
        self, 
        short_window: int = 20, 
        long_window: int = 50,
        ma_type: str = 'simple'
    ):
        """
        Inicializa la estrategia de media móvil
        
        :param short_window: Ventana de media móvil corta
        :param long_window: Ventana de media móvil larga
        :param ma_type: Tipo de media móvil ('simple', 'exponencial', 'ponderada')
        """
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        
        # Validar tipo de media móvil
        valid_types = ['simple', 'exponencial', 'ponderada']
        if ma_type not in valid_types:
            raise ValueError(f"Tipo de media móvil debe ser uno de: {valid_types}")
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las medias móviles basadas en el tipo seleccionado
        
        :param data: DataFrame con datos de precios
        :return: DataFrame con medias móviles añadidas
        """
        df = data.copy()
        
        # Media móvil simple
        if self.ma_type == 'simple':
            df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
            df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        
        # Media móvil exponencial
        elif self.ma_type == 'exponencial':
            df['short_ma'] = df['close'].ewm(
                span=self.short_window, 
                adjust=False
            ).mean()
            df['long_ma'] = df['close'].ewm(
                span=self.long_window, 
                adjust=False
            ).mean()
        
        # Media móvil ponderada
        elif self.ma_type == 'ponderada':
            weights = np.arange(1, self.long_window + 1)
            df['long_ma'] = df['close'].rolling(
                window=self.long_window
            ).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
            weights_short = np.arange(1, self.short_window + 1)
            df['short_ma'] = df['close'].rolling(
                window=self.short_window
            ).apply(lambda x: np.dot(x, weights_short) / weights_short.sum(), raw=True)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Genera señales de trading basadas en cruces de medias móviles
        
        :param data: DataFrame con datos de precios
        :return: Diccionario con señal de trading
        """
        # Calcular medias móviles
        df = self.calculate_moving_averages(data)
        
        # Eliminar filas con valores nulos
        df.dropna(inplace=True)
        
        # Obtener últimos valores
        last_short_ma = df['short_ma'].iloc[-1]
        last_long_ma = df['long_ma'].iloc[-1]
        prev_short_ma = df['short_ma'].iloc[-2]
        prev_long_ma = df['long_ma'].iloc[-2]
        current_price = df['close'].iloc[-1]
        
        # Señal de compra: cruce alcista
        if (prev_short_ma <= prev_long_ma and 
            last_short_ma > last_long_ma):
            return {
                'type': 'buy',
                'price': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': current_price * 1.03
            }
        
        # Señal de venta: cruce bajista
        elif (prev_short_ma >= prev_long_ma and 
              last_short_ma < last_long_ma):
            return {
                'type': 'sell',
                'price': current_price,
                'stop_loss': current_price * 1.05,
                'take_profit': current_price * 0.97
            }
        
        # Sin señal
        return None
    
    def get_required_indicators(self) -> List[str]:
        """
        Devuelve los indicadores requeridos por la estrategia
        
        :return: Lista de indicadores necesarios
        """
        return ['close']

    def optimize_parameters(
        self, 
        historical_data: pd.DataFrame, 
        parameter_ranges: Dict[str, List[int]] = None
    ) -> Dict[str, int]:
        """
        Optimiza parámetros de la estrategia mediante búsqueda de cuadrículas
        
        :param historical_data: Datos históricos para optimización
        :param parameter_ranges: Rangos de parámetros para búsqueda
        :return: Mejores parámetros
        """
        if not parameter_ranges:
            parameter_ranges = {
                'short_window': range(10, 50, 5),
                'long_window': range(20, 100, 10)
            }
        
        best_performance = float('-inf')
        best_params = {}
        
        for short_window in parameter_ranges['short_window']:
            for long_window in parameter_ranges['long_window']:
                if short_window >= long_window:
                    continue
                
                # Probar configuración
                strategy = MovingAverageStrategy(
                    short_window=short_window, 
                    long_window=long_window
                )
                
                # Simular resultados
[O                trades = self._simulate_trades(strategy, historical_data)
                performance = self._calculate_performance(trades)
                
                # Actualizar mejores parámetros
                if performance > best_performance:
                    best_performance = performance
                    best_params = {
                        'short_window': short_window,
                        'long_window': long_window
                    }
        
        return best_params

    def _simulate_trades(self, strategy, data):
        # Implementación de simulación de trades
        pass

    def _calculate_performance(self, trades):
        # Implementación de cálculo de rendimiento
        pass
