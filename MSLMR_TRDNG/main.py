import os
import sys
from core.config_manager import ConfigManager
from core.logging_system import setup_logging
from core.database_interactor import DatabaseInteractor
from trading.live_trading import LiveTradingSystem
import asyncio

def main():
    # Initialize ConfigManager
    config_manager = ConfigManager()
    config = config_manager.get_config()

    # Setup logging
    logger = setup_logging(config)

    try:
        logger.info(f"Iniciando {config['PROJECT_NAME']}")

        # Initialize DatabaseInteractor
        db_interactor = DatabaseInteractor(config['POSTGRES_URL'])
        db_interactor.create_tables()

        # Start trading system using asyncio.run() for async execution
        async def run_trading_system():
            trading_system = LiveTradingSystem()
            await trading_system.start()

        asyncio.run(run_trading_system())

    except Exception as e:
        logger.exception(f"Error crítico al iniciar {config['PROJECT_NAME']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
