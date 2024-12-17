import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from strategies.base import BaseStrategy

class RSIMACDStrategy(BaseStrategy):
    """
    Estrategia de Trading basada en RSI y MACD
    Combina indicadores de momento y tendencia
    """
    
    def __init__(
        self, 
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        """
        Inicializa parámetros de la estrategia RSI-MACD
        
        :param rsi_period: Período para cálculo de RSI
        :param rsi_oversold: Umbral de sobreventa
        :param rsi_overbought: Umbral de sobrecompra
        :param macd_fast: Período de media móvil rápida
        :param macd_slow: Período de media móvil lenta
        :param macd_signal: Período de señal MACD
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calcula el Índice de Fuerza Relativa (RSI)
        
        :param data: Serie de precios de cierre
        :param period: Período de cálculo
        :return: Serie con valores de RSI
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self, 
        data: pd.Series, 
        fast_period: int, 
        slow_period: int, 
        signal_period: int
    ) -> Dict[str, pd.Series]:
        """
        Calcula MACD (Moving Average Convergence Divergence)
        
        :param data: Serie de precios de cierre
        :return: Diccionario con líneas MACD
        """
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Genera señales de trading basadas en RSI y MACD
        
        :param data: DataFrame con datos de precios
        :return: Diccionario con señal de trading
        """
        # Calcular RSI y MACD
        rsi = self._calculate_rsi(data['close'], self.rsi_period)
        macd_indicators = self._calculate_macd(
            data['close'], 
            self.macd_fast, 
            self.macd_slow, 
            self.macd_signal
        )
        
        # Eliminar filas con valores nulos
        data = data.copy()
        data['rsi'] = rsi
        data.update(macd_indicators)
        data.dropna(inplace=True)
        
        # Obtener último valor
        last_rsi = data['rsi'].iloc[-1]
        last_macd_line = data['macd_line'].iloc[-1]
        last_signal_line = data['signal_line'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Señal de compra: RSI sobreventa + MACD bullish
        if (last_rsi < self.rsi_oversold and 
            last_macd_line > last_signal_line):
            return {
                'type': 'buy',
                'price': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': current_price * 1.03
            }
        
        # Señal de venta: RSI sobrecompra + MACD bearish
        elif (last_rsi > self.rsi_overbought and 
              last_macd_line < last_signal_line):
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

# Ejemplo de uso y prueba
def main():
    # Simular datos de mercado
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Inicializar estrategia
    strategy = RSIMACDStrategy()
    
    # Generar señales
    for i in range(len(df) - 14):  # Requiere datos suficientes para RSI
        signal = strategy.generate_signal(df.iloc[:i+14])
        if signal:
            print(f"Fecha: {df.index[i]}, Señal: {signal}")

if __name__ == "__main__":
    main()
