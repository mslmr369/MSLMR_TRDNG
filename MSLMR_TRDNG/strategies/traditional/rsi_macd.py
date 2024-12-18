import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from strategies.base import BaseStrategy
from strategies.risk_management import RiskManager

class RSIMACDStrategy(BaseStrategy):
    """
    Estrategia de Trading basada en RSI y MACD
    Combina indicadores de momento y tendencia
    """
    
    def __init__(
        self, 
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Inicializa parámetros de la estrategia RSI-MACD
        
        :param rsi_period: Período para cálculo de RSI
        :param rsi_oversold: Umbral de sobreventa
        :param rsi_overbought: Umbral de sobrecompra
        :param macd_fast: Período de media móvil rápida
        :param macd_slow: Período de media móvil lenta
        :param macd_signal: Período de señal MACD
        :param risk_manager: Instancia de RiskManager
        """
        # Now uses super to call the BaseStrategy init
        super().__init__(name='RSI_MACD', description='Estrategia que combina RSI y MACD')
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.risk_manager = risk_manager or RiskManager()
        # Setting the parameters in the strategy
        self.set_parameters(
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal
        )

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
            signal = {
                'type': 'buy',
                'price': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': current_price * 1.03
            }

        
        # Señal de venta: RSI sobrecompra + MACD bearish
        elif (last_rsi > self.rsi_overbought and 
              last_macd_line < last_signal_line):
            signal = {
                'type': 'sell',
                'price': current_price,
                'stop_loss': current_price * 1.05,
                'take_profit': current_price * 0.97
            }
        
        # Sin señal
        else:
            signal = None
        # Aplicar gestión de riesgo a la señal
        if signal:
            # Calcular stop loss dinámico basado en ATR
            current_price = data['close'].iloc[-1]
            stop_loss = self.risk_manager.calculate_dynamic_stop_loss(current_price, data, side=signal['type'])
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = current_price + (current_price - stop_loss) * self.risk_manager.reward_risk_ratio
            # Validate the trade using RiskManager
            if not self.risk_manager.validate_trade_risk(
                entry_price=current_price,
                stop_loss=stop_loss,
                position_size=1,  # You might want to replace this with actual position size
                account_balance=10000  # Replace with actual account balance if available
            ):
                self.logger.info(f"Trade for {self.name} on {data.index[-1]} did not pass risk validation.")
                return None

        return signal
    
    def get_required_indicators(self) -> List[str]:
        """
        Devuelve los indicadores requeridos por la estrategia
        
        :return: Lista de indicadores necesarios
        """
        return ['close']

    def validate_parameters(self) -> bool:
        """
        Valida los parámetros de la estrategia.
        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        if not isinstance(self.rsi_period, int) or self.rsi_period <= 0:
            self.logger.error("rsi_period debe ser un entero positivo.")
            return False
        
        if not isinstance(self.rsi_overbought, (int, float)) or not 0 <= self.rsi_overbought <= 100:
            self.logger.error("rsi_overbought debe ser un número entre 0 y 100.")
            return False

        if not isinstance(self.rsi_oversold, (int, float)) or not 0 <= self.rsi_oversold <= 100:
            self.logger.error("rsi_oversold debe ser un número entre 0 y 100.")
            return False
        
        if self.rsi_oversold >= self.rsi_overbought:
            self.logger.error("rsi_oversold debe ser menor que rsi_overbought.")
            return False

        if not isinstance(self.macd_fast, int) or self.macd_fast <= 0:
            self.logger.error("macd_fast debe ser un entero positivo.")
            return False

        if not isinstance(self.macd_slow, int) or self.macd_slow <= 0:
            self.logger.error("macd_slow debe ser un entero positivo.")
            return False

        if not isinstance(self.macd_signal, int) or self.macd_signal <= 0:
            self.logger.error("macd_signal debe ser un entero positivo.")
            return False

        if self.macd_fast >= self.macd_slow:
            self.logger.error("macd_fast debe ser menor que macd_slow.")
            return False

        return True
