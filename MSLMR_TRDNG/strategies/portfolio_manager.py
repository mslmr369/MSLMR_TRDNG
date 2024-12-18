import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from core.logging_system import LoggingMixin, log_method
from core.config_manager import ConfigManager

class PortfolioManager(LoggingMixin):
    """
    Gestiona la composición y riesgo de un portfolio de trading
    """

    def __init__(
        self,
        initial_capital: float = 10000.0
    ):
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
        position_size = self.calculate_position_size(entry_price, stop_loss)

        if position_size * abs(entry_price - stop_loss) > self.current_capital * self.max_single_trade_risk:
            self.logger.warning(f"Posición para {symbol} excede el riesgo máximo por operación")
            return

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

    @log_method()
    def close_position(
        self,
        symbol: str,
        exit_price: float
    ):
        """
        Cierra una posición existente

        :param symbol: Símbolo del activo
        :param exit_price: Precio de salida
        """
        if symbol not in self.active_positions:
            self.logger.warning(f"No existe posición abierta para {symbol}")
            return

        position = self.active_positions[symbol]

        # Calcular profit/loss
        if position['side'] == 'buy':
            profit_loss = (exit_price - position['entry_price']) * position['size']
        else:
            profit_loss = (position['entry_price'] - exit_price) * position['size']

        # Actualizar capital
        self.current_capital += profit_loss

        # Registrar trade en historial
        trade_record = {
            **position,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'exit_timestamp': pd.Timestamp.now()
        }
        self.trade_history.append(trade_record)

        # Eliminar posición
        del self.active_positions[symbol]

        self.logger.info(f"Posición cerrada: {trade_record}")

    def get_portfolio_metrics(self) -> Dict:
        """
        Calcula métricas del portfolio

        :return: Diccionario con métricas
        """
        if not self.trade_history:
            return {}

        trades_df = pd.DataFrame(self.trade_history)

        return {
            'total_trades': len(self.trade_history),
            'total_profit': trades_df['profit_loss'].sum(),
            'win_rate': (trades_df['profit_loss'] > 0).mean(),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'current_capital': self.current_capital
        }

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """
        Calcula el máximo drawdown

        :param trades_df: DataFrame de trades
        :return: Máximo drawdown
        """
        cumulative_profit = trades_df['profit_loss'].cumsum()
        return abs((cumulative_profit.cummax() - cumulative_profit).max())

# Ejemplo de uso
def main():
    # Inicializar gestor de portfolio
    portfolio = PortfolioManager(
        initial_capital=10000.0,
        max_portfolio_risk=0.1,
        max_single_trade_risk=0.02
    )

    # Simular algunas operaciones
    portfolio.open_position(
        symbol='BTC/USDT',
        side='buy',
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000
    )

    # Cerrar posición
    portfolio.close_position(
        symbol='BTC/USDT',
        exit_price=51000
    )

    # Obtener métricas
    metrics = portfolio.get_portfolio_metrics()
    print("Métricas del Portfolio:", metrics)

if __name__ == "__main__":
    main()
