import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from core.logging_system import LoggingMixin, log_method
from core.config_manager import ConfigManager
from strategies.risk_management import RiskManager

class PortfolioManager(LoggingMixin):
    """
    Gestiona la composición y riesgo de un portfolio de trading
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Inicializa el gestor de portfolio

        :param initial_capital: Capital inicial
        :param max_portfolio_risk: Riesgo máximo del portfolio
        :param max_single_trade_risk: Riesgo máximo por operación
        """
        config_manager = ConfigManager()
        config = config_manager.get_config()

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = config.get('MAX_PORTFOLIO_RISK', 0.1)
        self.max_single_trade_risk = config.get('MAX_SINGLE_TRADE_RISK', 0.02)
        self.risk_manager = RiskManager()  # Instantiate RiskManager

        # Seguimiento de posiciones
        self.active_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

    @log_method()
    def calculate_position_size(
        self,
        current_price: float,
        stop_loss_price: float,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """
        Calcula el tamaño de posición basado en el riesgo

        :param current_price: Precio actual
        :param stop_loss_price: Precio de stop loss
        :param risk_per_trade: Porcentaje de riesgo por trade
        :return: Tamaño de posición
        """
        risk_per_trade = risk_per_trade or self.max_single_trade_risk
        risk_amount = self.current_capital * risk_per_trade

        # Calcular riesgo por acción/contrato
        risk_per_share = abs(current_price - stop_loss_price)

        # Calcular tamaño de posición
        if risk_per_share == 0:
            self.logger.warning(f"Riesgo por acción es cero al calcular el tamaño de la posición. El precio actual es {current_price} y el precio de stop loss es {stop_loss_price}.")
            return 0  # Evitar la división por cero

        position_size = risk_amount / risk_per_share
        return np.floor(position_size)

    @log_method()
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ):
        """
        Abre una nueva posición

        :param symbol: Símbolo del activo
        :param side: Lado de la operación ('buy' o 'sell')
        :param entry_price: Precio de entrada
        :param stop_loss: Precio de stop loss
        :param take_profit: Precio de take profit
        """
        # Calcular tamaño de posición
        position_size = self.calculate_position_size(entry_price, stop_loss)

        # Verificar límites de riesgo a nivel de portfolio
        if self.risk_manager.check_max_exposure(trades_df=pd.DataFrame(self.trade_history), account_balance=self.current_capital, max_exposure=self.max_portfolio_risk):
            self.logger.warning(f"No se puede abrir posición para {symbol} porque se excede el límite de riesgo de portfolio.")
            return

        # Verificar límites de riesgo a nivel de posición
        if position_size * abs(entry_price - stop_loss) > self.current_capital * self.max_single_trade_risk:
            self.logger.warning(f"Posición para {symbol} excede el riesgo máximo por operación")
            return
        
        # Registrar posición
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'timestamp': pd.Timestamp.now()
        }

        self.active_positions[symbol] = position
        self.logger.info(f"Posición abierta: {position}")

    # ... (Rest of the PortfolioManager class remains the same) ...
