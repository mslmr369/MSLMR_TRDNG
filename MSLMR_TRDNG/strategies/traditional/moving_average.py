import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from strategies.base import BaseStrategy
from strategies.risk_management import RiskManager

class MovingAverageStrategy(BaseStrategy):
    """
    Estrategia de Trading basada en Medias Móviles
    Soporta múltiples tipos de medias móviles y configuraciones
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        ma_type: str = 'simple',
        risk_manager: Optional[RiskManager] = None  # Accept RiskManager instance
    ):
        """
        Inicializa la estrategia de media móvil

        :param short_window: Ventana de media móvil corta
        :param long_window: Ventana de media móvil larga
        :param ma_type: Tipo de media móvil ('simple', 'exponencial', 'ponderada')
        :param risk_manager: Instancia de RiskManager
        """
        super().__init__(name='Moving_Average', description='Estrategia de cruce de medias móviles')  # Call to super is needed
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.risk_manager = risk_manager or RiskManager()

        # Validar tipo de media móvil
        valid_types = ['simple', 'exponencial', 'ponderada']
        if ma_type not in valid_types:
            raise ValueError(f"Tipo de media móvil debe ser uno de: {valid_types}")

        # Set parameters for the strategy
        self.set_parameters(
            short_window=short_window,
            long_window=long_window,
            ma_type=ma_type
        )

    # ... Rest of the class definition ...

    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        # ... (Previous code for moving average calculation) ...
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
