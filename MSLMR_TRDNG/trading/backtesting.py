import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from strategies.base import BaseStrategy
from strategies.risk_management import RiskManager
from strategies.portfolio_manager import PortfolioManager
from core.logging_system import LoggingMixin

@dataclass
class BacktestConfiguration:
    """Configuración de backtesting"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% por trade
    risk_free_rate: float = 0.02    # Tasa libre de riesgo
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT'])
    timeframes: List[str] = field(default_factory=lambda: ['1h'])

@dataclass
class TradeEvent:
    """Representación detallada de un evento de trade"""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    size: float = 0.0
    profit_loss: float = 0.0
    commission: float = 0.0
    duration: Optional[pd.Timedelta] = None

class BacktestEngine(LoggingMixin):
    """
    Motor de backtesting avanzado con soporte para múltiples estrategias
    """
    
    def __init__(self, config: BacktestConfiguration):
        """
        Inicializa el motor de backtesting
        
        :param config: Configuración de backtesting
        """
        self.config = config
        self.portfolio_manager = PortfolioManager(
            initial_capital=config.initial_capital
        )
        self.risk_manager = RiskManager()
        
        # Almacenamiento de resultados
        self.trades: List[TradeEvent] = []
        self.equity_curve: List[Dict] = []
    
    def load_historical_data(
        self, 
        symbol: str, 
        timeframe: str
    ) -> pd.DataFrame:
        """
        Carga datos históricos para backtesting
        
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        :return: DataFrame con datos históricos
        """
        try:
            # En un escenario real, esto vendría de una fuente de datos
            # Por ahora, generamos datos sintéticos
            dates = pd.date_range(
                start=self.config.start_date or datetime(2022, 1, 1), 
                end=self.config.end_date or datetime(2023, 1, 1), 
                freq=timeframe
            )
            
            # Simular precios
            np.random.seed(42)
            prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
            volume = np.random.random(len(dates)) * 1000000
            
            df = pd.DataFrame({
                'open': prices,
                'high': prices + np.random.random(len(dates)),
                'low': prices - np.random.random(len(dates)),
                'close': prices,
                'volume': volume
            }, index=dates)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error cargando datos históricos: {e}")
            raise
    
    def run_backtest(
        self, 
        strategy: BaseStrategy, 
        symbol: str, 
        timeframe: str
    ):
        """
        Ejecuta backtesting para una estrategia específica
        
        :param strategy: Estrategia a testear
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        """
        # Cargar datos históricos
        market_data = self.load_historical_data(symbol, timeframe)
        
        current_position: Optional[TradeEvent] = None
        
        for timestamp, row in market_data.iterrows():
            try:
                # Generar señal de la estrategia
                signal = strategy.generate_signal(market_data.loc[:timestamp])
                
                # Lógica de entrada/salida de trades
                if signal and not current_position:
                    # Abrir nueva posición
                    entry_price = row['close']
                    stop_loss = signal.get('stop_loss', entry_price * 0.95)
                    
                    # Calcular tamaño de posición basado en riesgo
                    position_size = self.portfolio_manager.calculate_position_size(
                        entry_price, stop_loss
                    )
                    
                    # Validar riesgo
                    if self.risk_manager.validate_trade_risk(
                        entry_price, stop_loss, position_size, 
                        self.portfolio_manager.current_capital
                    ):
                        current_position = TradeEvent(
                            symbol=symbol,
                            side=signal['type'],
                            entry_time=timestamp,
                            entry_price=entry_price,
                            size=position_size,
                            commission=entry_price * position_size * self.config.commission_rate
                        )
                
                elif current_position:
                    # Lógica de cierre de posición
                    exit_condition = (
                        (signal and signal['type'] != current_position.side) or
                        (row['close'] <= current_position.stop_loss if current_position.side == 'buy' else 
                         row['close'] >= current_position.stop_loss)
                    )
                    
                    if exit_condition:
                        exit_price = row['close']
                        
                        # Calcular profit/loss
                        if current_position.side == 'buy':
                            profit_loss = (exit_price - current_position.entry_price) * current_position.size
                        else:
                            profit_loss = (current_position.entry_price - exit_price) * current_position.size
                        
                        # Actualizar trade
                        current_position.exit_time = timestamp
                        current_position.exit_price = exit_price
                        current_position.profit_loss = profit_loss
                        current_position.duration = timestamp - current_position.entry_time
                        current_position.commission += (
                            exit_price * current_position.size * self.config.commission_rate
                        )
                        
                        # Registrar trade
                        self.trades.append(current_position)
                        
                        # Resetear posición
                        current_position = None
                
                # Actualizar curva de equity
                current_capital = self.portfolio_manager.current_capital + sum(
                    trade.profit_loss for trade in self.trades
                )
                
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'capital': current_capital
                })
            
            except Exception as e:
                self.logger.error(f"Error en backtesting: {e}")
    
    def run_multiple_strategies(
        self, 
        strategies: List[BaseStrategy]
    ) -> Dict:
        """
        Ejecuta backtesting para múltiples estrategias
        
        :param strategies: Lista de estrategias a testear
        :return: Resultados comparativos
        """
        results = {}
        
        for strategy in strategies:
            strategy_trades = []
            
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    self.run_backtest(strategy, symbol, timeframe)
                    strategy_trades.extend(self.trades)
            
            # Calcular métricas de la estrategia
            results[strategy.name] = self._calculate_strategy_metrics(strategy_trades)
        
        return results
    
    def _calculate_strategy_metrics(
        self, 
        trades: List[TradeEvent]
    ) -> Dict:
        """
        Calcula métricas de rendimiento de una estrategia
        
        :param trades: Lista de trades
        :return: Diccionario con métricas
        """
        if not trades:
            return {}
        
        # Conversión a DataFrame para cálculos
        trades_df = pd.DataFrame([
            {
                'profit_loss': trade.profit_loss,
                'duration': trade.duration.total_seconds() if trade.duration else 0,
                'commission': trade.commission
            } for trade in trades
        ])
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': (trades_df['profit_loss'] > 0).sum(),
            'losing_trades': (trades_df['profit_loss'] < 0).sum(),
            'win_rate': (trades_df['profit_loss'] > 0).mean(),
            'total_profit': trades_df['profit_loss'].sum(),
            'avg_profit_per_trade': trades_df['profit_loss'].mean(),
            'max_profit': trades_df['profit_loss'].max(),
            'max_loss': trades_df['profit_loss'].min(),
            'avg_trade_duration': trades_df['duration'].mean(),
            'total_commissions': trades_df['commission'].sum()
        }
        
        return metrics

# Ejemplo de uso
def main():
    from models.traditional.rsi_macd import RSIMACDStrategy
    from models.traditional.moving_average import MovingAverageStrategy
    
    # Configuración de backtesting
    config = BacktestConfiguration(
        initial_capital=10000.0,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 1, 1)
    )
    
    # Inicializar motor de backtesting
    backtest_engine = BacktestEngine(config)
    
    # Estrategias a testear
    strategies = [
        RSIMACDStrategy(),
        MovingAverageStrategy()
    ]
    
    # Ejecutar backtesting
    results = backtest_engine.run_multiple_strategies(strategies)
    
    # Imprimir resultados
    for strategy_name, metrics in results.items():
        print(f"Resultados para {strategy_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()