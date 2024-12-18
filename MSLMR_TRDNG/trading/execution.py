import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import ccxt
from strategies.strategy_registry import StrategyRegistry
from strategies.portfolio_manager import PortfolioManager
from strategies.risk_management import RiskManager
from core.logging_system import LoggingMixin, log_method
from models.ml.forecasting import TimeSeriesForecaster
from data.ingestion import DataIngestionManager
from core.config_manager import ConfigManager

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
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

        # Configuración de exchange
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': self.config.get('EXCHANGE_API_KEY'),
            'secret': self.config.get('EXCHANGE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Trading de futuros
            }
        })

        # Configuración de trading
        self.symbols = symbols or self.config.get('TRADING_SYMBOLS', [])
        self.timeframes = timeframes or self.config.get('TIMEFRAMES', [])
        self.dry_run = dry_run

        # Componentes del sistema
        self.strategy_registry = StrategyRegistry()
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            max_portfolio_risk=self.config.get('MAX_PORTFOLIO_RISK', 0.1),
            max_single_trade_risk=self.config.get('MAX_SINGLE_TRADE_RISK', 0.02)
        )
        self.risk_manager = RiskManager(
            max_portfolio_risk=self.config.get('MAX_PORTFOLIO_RISK', 0.1),
            max_single_trade_risk=self.config.get('MAX_SINGLE_TRADE_RISK', 0.02),
            stop_loss_atr_multiplier=self.config.get('STOP_LOSS_MULTIPLIER', 2.0)
        )
        self.forecaster = TimeSeriesForecaster()

        # Inicializar DataIngestionManager para la obtención de datos
        self.data_ingestion_manager = DataIngestionManager(
            exchange_id=exchange_id,
            symbols=self.symbols,
            timeframes=self.timeframes
        )

    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Obtiene datos de mercado utilizando DataIngestionManager.

        :param symbol: Símbolo del activo.
        :param timeframe: Intervalo temporal.
        :param limit: Número de candeleros.
        :return: DataFrame con datos de mercado.
        """
        try:
            df = self.data_ingestion_manager.fetch_historical_data(symbol, timeframe, limit)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol} on {timeframe}: {e}")
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
        :param timeframe: Intervalo temporal
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
