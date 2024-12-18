
import math
from typing import Dict, List, Optional
import pandas as pd
from core.logging_system import LoggingMixin
from core.config_manager import ConfigManager

class RiskManager(LoggingMixin):
    """
    Gestiona el riesgo de las estrategias de trading
    """

    def __init__(self):
        """
        Inicializa el gestor de riesgos con los valores de configuración
        """
        config_manager = ConfigManager()
        config = config_manager.get_config()
        self.max_portfolio_risk = config.get('MAX_PORTFOLIO_RISK', 0.1)
        self.max_single_trade_risk = config.get('MAX_SINGLE_TRADE_RISK', 0.02)
        self.stop_loss_atr_multiplier = config.get('STOP_LOSS_MULTIPLIER', 2.0)
        self.trailing_stop_loss_multiplier = config.get('TRAILING_STOP_LOSS_MULTIPLIER', 1.5)
        self.trailing_stop_loss_period = config.get('TRAILING_STOP_LOSS_PERIOD', 14)

    def calculate_atr(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calcula el Average True Range (ATR)

        :param data: DataFrame con datos de mercado
        :param period: Período para cálculo de ATR
        :return: Serie con valores de ATR
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Calcular True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # Calcular ATR
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        data: pd.DataFrame,
        atr_period: int = 14,
        side: str = 'buy'
    ) -> float:
        """
        Calcula stop loss dinámico basado en ATR

        :param entry_price: Precio de entrada
        :param data: DataFrame con datos de mercado
        :param atr_period: Período para cálculo de ATR
        :return: Precio de stop loss
        """
        atr = self.calculate_atr(data, atr_period)

        # Usar último valor de ATR
        current_atr = atr.iloc[-1]

        # Calcular stop loss basado en la dirección del trade
        stop_loss = entry_price - (current_atr * self.stop_loss_atr_multiplier) if side == 'buy' \
            else entry_price + (current_atr * self.stop_loss_atr_multiplier)

        return stop_loss

    def calculate_trailing_stop_loss(self, side, current_price, last_stop_loss, atr):
        """
        Calculate the trailing stop loss based on the current price,
        previous stop loss, and Average True Range (ATR).
        """
        stop_distance = atr * self.trailing_stop_loss_multiplier
        if side == 'buy':
            stop_loss = current_price - stop_distance
            if last_stop_loss is not None:
                stop_loss = max(stop_loss, last_stop_loss)
        elif side == 'sell':
            stop_loss = current_price + stop_distance
            if last_stop_loss is not None:
                stop_loss = min(stop_loss, last_stop_loss)
        else:
            stop_loss = None  # No se pudo determinar un stop loss
        return stop_loss

    def validate_trade_risk(
        self,
        entry_price: float,
        stop_loss: float,
        position_size: float,
        account_balance: float
    ) -> bool:
        """
        Valida el riesgo de una operación

        :param entry_price: Precio de entrada
        :param stop_loss: Precio de stop loss
        :param position_size: Tamaño de posición
        :param account_balance: Saldo de cuenta
        :return: Booleano indicando si el trade cumple los criterios de riesgo
        """
        # Calcular riesgo de la operación
        trade_risk = abs(entry_price - stop_loss) * position_size
        portfolio_risk = trade_risk / account_balance

        # Verificar límites de riesgo
        if portfolio_risk > self.max_single_trade_risk:
            self.logger.warning(
                f"Riesgo de operación {portfolio_risk:.2%} "
                f"excede límite máximo {self.max_single_trade_risk:.2%}"
            )
            return False

        return True

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calcula el criterio de Kelly para sizing de posición

        :param win_rate: Porcentaje de trades ganadores
        :param win_loss_ratio: Ratio de ganancias vs pérdidas
        :return: Porcentaje óptimo de capital a arriesgar
        """
        # Fórmula de Kelly: f = (p * b - q) / b
        # p: probabilidad de ganar
        # q: probabilidad de perder (1 - p)
        # b: ratio de ganancias

        win_probability = win_rate
        loss_probability = 1 - win_rate

        kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

        # Fractional Kelly para reducir volatilidad
        conservative_kelly = kelly_fraction * 0.5

        return max(0, min(conservative_kelly, self.max_single_trade_risk))

    def analyze_portfolio_concentration(
        self,
        trades: List[Dict],
        max_concentration: float = 0.3
    ) -> Dict:
        """
        Analiza la concentración del portfolio

        :param trades: Lista de trades
        :param max_concentration: Concentración máxima permitida
        :return: Informe de concentración
        """
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        # Concentración por símbolo
        symbol_exposure = trades_df.groupby('symbol')['size'].sum()
        total_exposure = symbol_exposure.sum()

        symbol_concentration = symbol_exposure / total_exposure

        # Identificar símbolos sobre concentración
        over_concentrated = symbol_concentration[symbol_concentration > max_concentration]

        report = {
            'total_exposure': total_exposure,
            'symbol_concentration': symbol_concentration.to_dict(),
            'over_concentrated_symbols': list(over_concentrated.index)
        }

        if over_concentrated:
            self.logger.warning(
                f"Símbolos sobre concentración máxima: {list(over_concentrated.index)}"
            )

        return report

    def simulate_monte_carlo_drawdown(
        self,
        trades: List[Dict],
        simulations: int = 1000
    ) -> Dict:
        """
        Simula drawdowns potenciales mediante Montecarlo

        :param trades: Lista de trades
        :param simulations: Número de simulaciones
        :return: Estadísticas de drawdown
        """
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)
        cumulative_returns = trades_df['profit_loss'].cumsum()

        drawdowns = []

        # Simulación de Montecarlo
        for _ in range(simulations):
            # Remuestreo aleatorio de trades
            resampled_returns = cumulative_returns.sample(
                n=len(cumulative_returns),
                replace=True
            )

            # Calcular drawdown
            running_max = resampled_returns.cummax()
            drawdown = (resampled_returns - running_max) / running_max

            drawdowns.append(drawdown.min())

        return {
            'max_drawdown': np.percentile(drawdowns, 95),
            'average_drawdown': np.mean(drawdowns),
            'drawdown_std': np.std(drawdowns)
        }

    def generate_risk_report(
        self,
        trades: List[Dict],
        account_balance: float
    ) -> Dict:
        """
        Genera un informe completo de riesgo

        :param trades: Lista de trades
        :param account_balance: Saldo de cuenta
        :return: Informe de riesgo
        """
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        risk_report = {
            # Métricas de riesgo básicas
            'total_trades': len(trades),
            'profitable_trades': (trades_df['profit_loss'] > 0).sum(),
            'win_rate': (trades_df['profit_loss'] > 0).mean(),

            # Análisis de riesgo
            'max_single_trade_loss': trades_df['profit_loss'].min(),
            'max_single_trade_profit': trades_df['profit_loss'].max(),
            'average_trade_profit': trades_df['profit_loss'].mean(),

            # Distribución de pérdidas
            'loss_distribution': {
                'median_loss': trades_df[trades_df['profit_loss'] < 0]['profit_loss'].median(),
                '95th_percentile_loss': np.percentile(
                    trades_df[trades_df['profit_loss'] < 0]['profit_loss'],
                    95
                )
            },

            # Concentración
            'portfolio_concentration': self.analyze_portfolio_concentration(trades),

            # Simulación de drawdown
            'monte_carlo_drawdown': self.simulate_monte_carlo_drawdown(trades)
        }

        return risk_report
