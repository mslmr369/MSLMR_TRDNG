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
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
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

    # ... _calculate_rsi and _calculate_macd remain the same

    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        # ... (Previous code for RSI and MACD calculation) ...

        # Aplicar gestión de riesgo a la señal
        if signal:
            # Calcular stop loss dinámico basado en ATR
            current_price = data['close'].iloc[-1]
            stop_loss = self.risk_manager.calculate_dynamic_stop_loss(current_price, data, side=signal['type'])
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = current_price + (current_price - stop_loss) * self.risk_manager.stop_loss_atr_multiplier
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
