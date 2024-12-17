import os
import sys
import argparse
import asyncio
import logging
from typing import Optional, List

# Importaciones de componentes de trading
from trading.live_trading import LiveTradingSystem
from strategies.strategy_registry import StrategyRegistry

# Importaciones de estrategias
from models.traditional.rsi_macd import RSIMACDStrategy
from models.traditional.moving_average import MovingAverageStrategy

# Importaciones de monitoreo
from monitoring.alerts import AlertSystem
from monitoring.dashboards import TradingDashboard

class LiveTradingManager:
    """
    Gestor de sistema de trading en vivo
    Coordina ejecución, monitoreo y alertas
    """
    
    def __init__(
        self, 
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
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
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Configuraciones
        self.exchange_id = exchange_id
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.timeframes = timeframes or ['1h', '4h']
        
        # Componentes del sistema
        self.strategy_registry = StrategyRegistry()
        self._register_strategies()
        
        # Sistema de trading en vivo
        self.trading_system = LiveTradingSystem(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            symbols=self.symbols,
            timeframes=self.timeframes,
            initial_capital=initial_capital,
            dry_run=dry_run
        )
        
        # Sistema de alertas
        self.alert_system = self._configure_alerts()
        
        # Dashboard
        self.dashboard = TradingDashboard()
    
    def _register_strategies(self):
        """
        Registra estrategias disponibles para trading en vivo
        """
        self.strategy_registry.register_strategy('RSI_MACD', RSIMACDStrategy)
        self.strategy_registry.register_strategy('Moving_Average', MovingAverageStrategy)
    
    def _configure_alerts(self) -> AlertSystem:
        """
        Configura el sistema de alertas
        
        :return: Sistema de alertas configurado
        """
        alert_system = AlertSystem()
        
        # Añadir canales de alerta configurados
        # Email
        # alert_system.add_channel(
        #     EmailAlertChannel(
        #         smtp_host='smtp.gmail.com',
        #         smtp_port=587,
        #         sender_email='tu_email@gmail.com',
        #         sender_password='tu_contraseña',
        #         recipients=['destinatario@ejemplo.com']
        #     )
        # )
        
        # Telegram (descomentar y configurar)
        # alert_system.add_channel(
        #     TelegramAlertChannel(
        #         bot_token='tu_token_de_bot',
        #         chat_ids=['tu_chat_id']
        #     )
        # )
        
        return alert_system
    
    async def start_trading(self):
        """
        Inicia el sistema de trading en vivo
        """
        try:
            self.logger.info("Iniciando sistema de trading en vivo")
            
            # Enviar alerta de inicio
            await self.alert_system.send_alert(
                "Sistema de Trading", 
                "Iniciando trading en vivo", 
                severity='info'
            )
            
            # Iniciar dashboard en segundo plano
            dashboard_task = asyncio.create_task(
                self._start_dashboard()
            )
            
            # Iniciar trading
            trading_task = asyncio.create_task(
                self.trading_system.start()
            )
            
            # Esperar a que finalicen las tareas
            await asyncio.gather(
                dashboard_task, 
                trading_task
            )
        
        except Exception as e:
            self.logger.error(f"Error crítico en sistema de trading: {e}")
            
            # Enviar alerta de error
            await self.alert_system.send_alert(
                "Error Crítico", 
                f"Fallo en sistema de trading: {e}", 
                severity='critical'
            )
    
    async def _start_dashboard(self):
[O        """
        Inicia el dashboard en modo asíncrono
        """
        try:
            self.logger.info("Iniciando dashboard de trading")
            self.dashboard.run(debug=False)
        except Exception as e:
            self.logger.error(f"Error iniciando dashboard: {e}")
    
    def run(self):
        """
        Ejecuta el sistema de trading
        """
        try:
            asyncio.run(self.start_trading())
        except KeyboardInterrupt:
            self.logger.info("Sistema de trading detenido por el usuario")
        except Exception as e:
            self.logger.error(f"Error inesperado: {e}")


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Sistema de Trading en Vivo')
    
    parser.add_argument(
        '--exchange', 
        type=str, 
        default='binance', 
        help='Exchange a usar'
    )
    parser.add_argument(
        '--api-key', 
        type=str, 
        help='Clave de API del exchange'
    )
    parser.add_argument(
        '--api-secret', 
        type=str, 
        help='Secreto de API del exchange'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['BTC/USDT', 'ETH/USDT'], 
        help='Símbolos a operar'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Modo de prueba sin ejecución real'
    )
    
    args = parser.parse_args()
    
    # Inicializar y ejecutar sistema de trading
    trading_manager = LiveTradingManager(
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbols=args.symbols,
        dry_run=args.dry_run
    )
    
    # Ejecutar trading
    trading_manager.run()

if __name__ == '__main__':
    main()


