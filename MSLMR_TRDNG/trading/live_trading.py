import time
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from typing import Dict, List, Optional

from strategies.strategy_registry import StrategyRegistry
from strategies.portfolio_manager import PortfolioManager
from strategies.risk_management import RiskManager
from core.logging_system import LoggingMixin, log_method
from models.ml.forecasting import TimeSeriesForecaster
from core.config_manager import ConfigManager
from data.ingestion import AsyncDataIngestionManager

class LiveTradingSystem(LoggingMixin):
    """
    Sistema de trading en vivo con gestión de riesgo y monitoreo
    """

    def __init__(
        self,
        exchange_id: str = 'binance',
        symbols: List[str] = None,
        timeframes: List[str] = None,
        dry_run: bool = False
    ):
        """
        Inicializa el sistema de trading en vivo

        :param exchange_id: ID del exchange
        :param api_key: Clave de API
        :param api_secret: Secreto de API
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
                'defaultType': 'future'
            }
        })
        self.symbols = symbols or self.config.get('TRADING_SYMBOLS', [])
        self.timeframes = timeframes or self.config.get('TIMEFRAMES', [])
        self.dry_run = dry_run
        self.initial_capital = self.config.get('INITIAL_CAPITAL', 10000.0)

        # Componentes del sistema
        self.strategy_registry = StrategyRegistry()
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.initial_capital,
            max_portfolio_risk=self.config.get('MAX_PORTFOLIO_RISK', 0.1),
            max_single_trade_risk=self.config.get('MAX_SINGLE_TRADE_RISK', 0.02)
        )
        self.risk_manager = RiskManager(
            max_portfolio_risk=self.config.get('MAX_PORTFOLIO_RISK', 0.1),
            max_single_trade_risk=self.config.get('MAX_SINGLE_TRADE_RISK', 0.02),
            stop_loss_atr_multiplier=self.config.get('STOP_LOSS_MULTIPLIER', 2.0)
        )
        self.forecaster = TimeSeriesForecaster()
        self.alert_system = AlertSystem()

        # Configuracion de alertas
        if self.config.get('TELEGRAM_TOKEN') and self.config.get('TELEGRAM_CHAT_ID'):
            self.alert_system.add_channel(TelegramAlertChannel(self.config_manager))

        if self.config.get('EMAIL_HOST') and self.config.get('EMAIL_PORT') and self.config.get('EMAIL_USER') and self.config.get('EMAIL_PASSWORD') and self.config.get('EMAIL_RECIPIENTS'):
            self.alert_system.add_channel(EmailAlertChannel(self.config_manager))

        self.data_ingestion_manager = AsyncDataIngestionManager(
            exchange_id=exchange_id,
            symbols=self.symbols,
            timeframes=self.timeframes
        )

    # ... (async fetch_live_market_data remains the same) ...

    # ... (execute_live_trade remains the same) ...

    async def run_strategy_live(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str
    ):
        """
        Ejecuta una estrategia en vivo para un símbolo específico
        """
        strategy_class = self.strategy_registry.get_strategy(strategy_name)
        if not strategy_class:
            self.logger.error(f"Estrategia no encontrada: {strategy_name}")
            return

        strategy = strategy_class(risk_manager=self.risk_manager)

        try:
            market_data = await self.fetch_live_market_data(symbol, timeframe)
            signal = strategy.generate_signal(market_data)

            if signal:
                # Análisis predictivo con forecasting
                prediction = self.forecaster.predict(market_data)

                # Gestión de riesgo
                current_price = market_data['close'].iloc[-1]
                stop_loss = signal.get('stop_loss', current_price * 0.95)

                # Calcular tamaño de posición
                position_size = self.portfolio_manager.calculate_position_size(
                    current_price, stop_loss
                )

                # Validar riesgo de la operación
                risk_validated = self.risk_manager.validate_trade_risk(
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    account_balance=self.portfolio_manager.current_capital
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

                    # Ejecutar trade en vivo
                    await self.execute_live_trade(
                        symbol=symbol,
                        side=signal['type'],
                        amount=position_size
                    )

        except Exception as e:
            self.logger.error(f"Error ejecutando estrategia {strategy_name}: {e}")
            await self.alert_system.send_alert(
                "Error de Estrategia",
                f"Fallo en {strategy_name} para {symbol}: {e}"
            )

    async def continuous_trading_loop(self):
        """
        Bucle principal de trading continuo
        """
        while True:
            try:
                # Obtener estrategias registradas
                strategies = self.strategy_registry.list_strategies()

                tasks = []
                for strategy_name in strategies:
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            task = asyncio.create_task(
                                self.run_strategy_live(strategy_name, symbol, timeframe)
                            )
                            tasks.append(task)

                await asyncio.gather(*tasks)
                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Error en bucle de trading: {e}")
                await self.alert_system.send_alert(
                    "Error Crítico",
                    f"Fallo en bucle de trading: {e}"
                )
                await asyncio.sleep(60)

    async def start(self):
        """
        Iniciar sistema de trading en vivo
        """
        try:
            self.logger.info("Iniciando sistema de trading en vivo")
            await self.continuous_trading_loop()
        except Exception as e:
            self.logger.critical(f"Error crítico al iniciar trading: {e}")
            await self.alert_system.send_alert(
                "Error de Inicio",
                f"Fallo al iniciar sistema de trading: {e}"
            )
        finally:
            await self.exchange.close()

# Example usage
async def main():
    # Registrar estrategias
    from models.traditional.rsi_macd import RSIMACDStrategy
    from models.traditional.moving_average import MovingAverageStrategy

    registry = StrategyRegistry()
    registry.register_strategy('RSI_MACD', RSIMACDStrategy)
    registry.register_strategy('Moving_Average', MovingAverageStrategy)

    # Inicializar sistema de trading
    trading_system = LiveTradingSystem(
        exchange_id='binance',
        symbols=['BTC/USDT', 'ETH/USDT'],
        dry_run=True  # Modo de prueba
    )

    # Iniciar trading
    await trading_system.start()

if __name__ == "__main__":
    asyncio.run(main())

