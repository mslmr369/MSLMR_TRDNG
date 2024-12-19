from core.database import initialize_database
async def main():
    # Initialize ConfigManager
    config_manager = ConfigManager()
    config = config_manager.get_config()

    # Setup logging
    logger = setup_logging(config)

    # Initialize Database
    initialize_database()

    # Register strategies
    strategy_registry = StrategyRegistry()
    strategy_registry.register_strategy('RSI_MACD', RSIMACDStrategy)
    strategy_registry.register_strategy('Moving_Average', MovingAverageStrategy)

    # Register ML models
    # Example usage in main.py or another appropriate initialization file
    from models.ml.forecasting import TimeSeriesForecaster
    from models.ml.prediction import MarketNeuralNetwork

    model_registry = ModelRegistry()
    model_registry.register_model('TimeSeriesForecaster', TimeSeriesForecaster)
    model_registry.register_model('MarketNeuralNetwork', MarketNeuralNetwork)

    try:
        logger.info(f"Iniciando {config['PROJECT_NAME']}")

        # Initialize DatabaseInteractor
        db_interactor = DatabaseInteractor(config['POSTGRES_URL'])
        db_interactor.create_tables()

        # Start trading system
        trading_system = LiveTradingSystem(
            symbols=config['TRADING_SYMBOLS'],
            timeframes=config['TIMEFRAMES']
        )
        await trading_system.start()

    except Exception as e:
        logger.exception(f"Error crítico al iniciar {config['PROJECT_NAME']}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())