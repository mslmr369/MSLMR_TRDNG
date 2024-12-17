import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import ccxt
from strategies.strategy_registry import StrategyRegistry
from strategies.portfolio_manager import PortfolioManager
from strategies.risk_management import RiskManager
from core.logging_system import LoggingMixin, log_method
from models.ml.forecasting import TimeSeriesForecaster

class TradingExecutor(LoggingMixin):
    """
    Ejecutor central de estrategias de trading
    Coordina estrategias, gestión de portfolio y ejecución de trades
    """
    
    def __init__(
        self, 
        exchange_id: str = 'binance',
        symbols: List[str] = ['BTC/USDT', 'ETH/USDT'],
        timeframes: List[str] = ['1h', '4h'],
        initial_capital: float = 10000.0,
        dry_run: bool = True
    ):
        """
        Inicializa el ejecutor de trading
        
        :param exchange_id: ID del exchange
        :param symbols: Símbolos a operar
        :param timeframes: Intervalos de tiempo
        :param initial_capital: Capital inicial
        :param dry_run: Modo de prueba sin ejecución real
        """
        # Configuración de exchange
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)()
        
        # Configuración de trading
        self.symbols = symbols
        self.timeframes = timeframes
        self.dry_run = dry_run
        
        # Componentes del sistema
        self.strategy_registry = StrategyRegistry()
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        self.risk_manager = RiskManager()
        
        # Forecaster para análisis predictivo
        self.forecaster = TimeSeriesForecaster()
    
    @log_method()
    def fetch_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Obtiene datos de mercado del exchange
        
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        :param limit: Número de candeleros
        :return: DataFrame con datos de mercado
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    @log_method()
    def execute_trade(
        self, 
        symbol: str, 
        side: str, 
        amount: float, 
        price: float
    ) -> Optional[Dict]:
        """
        Ejecuta una operación en el exchange
        
        :param symbol: Símbolo del activo
        :param side: Lado de la operación ('buy' o 'sell')
        :param amount: Cantidad a operar
        :param price: Precio de la operación
        :return: Detalles de la operación
        """
        if self.dry_run:
            # Simular ejecución en modo prueba
            self.logger.info(f"Simulated {side} order for {symbol}: {amount} @ {price}")
            return {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'simulated'
            }
        
        try:
            # Ejecutar orden real
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            self.logger.info(f"Executed {side} order: {order}")
            return order
        
        except Exception as e:
            self.logger.error(f"Error executing {side} order for {symbol}: {e}")
            return None
    
    def run_strategy_for_symbol(
        self, 
        strategy_name: str, 
        symbol: str, 
        timeframe: str
    ):
        """
        Ejecuta una estrategia para un símbolo específico
        
        :param strategy_name: Nombre de la estrategia
        :param symbol: Símbolo del activo
[O        :param timeframe: Intervalo temporal
        """
        # Obtener estrategia del registro
        strategy_class = self.strategy_registry.get_strategy(strategy_name)
        if not strategy_class:
            self.logger.error(f"Estrategia no encontrada: {strategy_name}")
            return
        
        # Inicializar estrategia
        strategy = strategy_class()
        
        try:
            # Obtener datos de mercado
            market_data = self.fetch_market_data(symbol, timeframe)
            
            # Generar señal
            signal = strategy.generate_signal(market_data)
            
            if signal:
                # Análisis predictivo
                prediction = self.forecaster.predict(market_data)
                
                # Gestión de riesgo
                current_price = market_data['close'].iloc[-1]
                stop_loss = signal.get('stop_loss', current_price * 0.95)
                
                # Validar riesgo de la operación
                position_size = self.portfolio_manager.calculate_position_size(
                    current_price, stop_loss
                )
                
                risk_validated = self.risk_manager.validate_trade_risk(
                    current_price, stop_loss, position_size, 
                    self.portfolio_manager.current_capital
                )
                
                if risk_validated:
                    # Abrir posición
                    self.portfolio_manager.open_position(
                        symbol=symbol,
                        side=signal['type'],
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=signal.get('take_profit', current_price * 1.03)
                    )
                    
                    # Ejecutar trade
                    self.execute_trade(
                        symbol=symbol, 
                        side=signal['type'],
                        amount=position_size,
                        price=current_price
                    )
        
        except Exception as e:
            self.logger.error(f"Error ejecutando estrategia {strategy_name}: {e}")
    
    def run(self):
        """
        Ejecuta todas las estrategias registradas para todos los símbolos
        """
        # Obtener estrategias registradas
        strategies = self.strategy_registry.list_strategies()
        
        for strategy_name in strategies:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    self.run_strategy_for_symbol(strategy_name, symbol, timeframe)
        
        # Generar informe de portfolio
        portfolio_metrics = self.portfolio_manager.get_portfolio_metrics()
        risk_report = self.risk_manager.generate_risk_report(
            self.portfolio_manager.trade_history, 
            self.portfolio_manager.current_capital
        )
        
        self.logger.info("Portfolio Metrics: " + str(portfolio_metrics))
        self.logger.info("Risk Report: " + str(risk_report))

# Ejemplo de uso
def main():
    # Registrar estrategias
    from models.traditional.rsi_macd import RSIMACDStrategy
    from models.traditional.moving_average import MovingAverageStrategy
    
    registry = StrategyRegistry()
    registry.register_strategy('RSI_MACD', RSIMACDStrategy)
    registry.register_strategy('Moving_Average', MovingAverageStrategy)
    
    # Inicializar ejecutor
    executor = TradingExecutor(
        exchange_id='binance', 
        symbols=['BTC/USDT', 'ETH/USDT'],
        dry_run=True
    )
    
    # Ejecutar estrategias
    executor.run()

if __name__ == "__main__":
    main()
