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
from monitoring.alerts import AlertSystem

class LiveTradingSystem(LoggingMixin):
    """
    Sistema de trading en vivo con gestión de riesgo y monitoreo
    """
    
    def __init__(
        self, 
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: List[str] = ['BTC/USDT', 'ETH/USDT'],
        timeframes: List[str] = ['1h', '4h'],
        initial_capital: float = 10000.0,
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
        # Configuración de exchange
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Trading de futuros
            }
        })
        
        # Configuración de trading
        self.symbols = symbols
        self.timeframes = timeframes
        self.dry_run = dry_run
        
        # Componentes del sistema
        self.strategy_registry = StrategyRegistry()
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        self.risk_manager = RiskManager()
        self.forecaster = TimeSeriesForecaster()
        self.alert_system = AlertSystem()
    
    @log_method()
    async def fetch_live_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Obtiene datos de mercado en vivo del exchange
        
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        :param limit: Número de candeleros
        :return: DataFrame con datos de mercado
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching live data for {symbol}: {e}")
            await self.alert_system.send_alert(
                "Fallo en obtención de datos en vivo", 
                f"Error para {symbol}: {e}"
            )
            raise
    
    @log_method()
    async def execute_live_trade(
        self, 
        symbol: str, 
        side: str, 
        amount: float
    ) -> Optional[Dict]:
        """
        Ejecuta una operación en vivo en el exchange
        
        :param symbol: Símbolo del activo
        :param side: Lado de la operación ('buy' o 'sell')
        :param amount: Cantidad a operar
        :return: Detalles de la operación
        """
        if self.dry_run:
            self.logger.info(f"Simulated {side} order for {symbol}: {amount}")
            return {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'status': 'simulated'
            }
        
        try:
            order = await self.exchange.create_market_order(symbol, side, amount)
            
            # Enviar alerta de trade ejecutado
            await self.alert_system.send_alert(
                "Trade Ejecutado", 
                f"{side.upper()} {amount} {symbol}"
            )
            
            return order
        
        except Exception as e:
            self.logger.error(f"Error executing {side} order for {symbol}: {e}")
            await self.alert_system.send_alert(
                "Error de Trade", 
                f"Fallo {side} de {symbol}: {e}"
            )
            return None
    
    async def run_strategy_live(
        self, 
        strategy_name: str, 
        symbol: str, 
        timeframe: str
    ):
        """
        Ejecuta una estrategia en vivo para un símbolo específico
        
        :param strategy_name: Nombre de la estrategia
        :param symbol: Símbolo del activo
        :param timeframe: Intervalo temporal
        """
        # Obtener estrategia del registro
        strategy_class = self.strategy_registry.get_strategy(strategy_name)
        if not strategy_class:
            self.logger.error(f"Estrategia no encontrada: {strategy_name}")
            return
        
        strategy = strategy_class()
        
        try:
            # Obtener datos de mercado en vivo
            market_data = await self.fetch_live_market_data(symbol, timeframe)
            
            # Generar señal
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
                
                # Ejecutar todas las estrategias concurrentemente
                await asyncio.gather(*tasks)
                
                # Pausa entre iteraciones
                await asyncio.sleep(300)  # 5 minutos
            
            except Exception as e:
                self.logger.error(f"Error en bucle de trading: {e}")
                await self.alert_system.send_alert(
                    "Error Crítico", 
                    f"Fallo en bucle de trading: {e}"
                )
                
                # Esperar antes de reintentar
                await asyncio.sleep(60)
    
    async def start(self):
        """
        Iniciar sistema de trading en vivo
        """
        try:
            self.logger.info("Iniciando sistema de trading en vivo")
            
            # Iniciar bucle de trading
            await self.continuous_trading_loop()
        
        except Exception as e:
            self.logger.critical(f"Error crítico al iniciar trading: {e}")
            await self.alert_system.send_alert(
                "Error de Inicio", 
                f"Fallo al iniciar sistema de trading: {e}"
            )
        
        finally:
            # Cerrar conexión con exchange
            await self.exchange.close()

# Ejemplo de uso
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
        api_key='tu_api_key',
        api_secret='tu_api_secret',
        symbols=['BTC/USDT', 'ETH/USDT'],
        dry_run=True  # Modo de prueba
    )
    
    # Iniciar trading
    await trading_system.start()

if __name__ == "__main__":
    asyncio.run(main())

