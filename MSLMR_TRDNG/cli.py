import typer
import os
from typing import Optional, List
from datetime import datetime
import asyncio

from core.config_manager import ConfigManager
from core.logging_system import setup_logging
from data.ingestion import AsyncDataIngestionManager
from strategies.strategy_registry import StrategyRegistry
from trading.live_trading import LiveTradingSystem

app = typer.Typer(help="MSLMR_TRDNG - Algorithmic Cryptocurrency Trading System")

@app.command()
def run_live(
    config: Optional[str] = typer.Option(
        "development",
        "--config",
        "-c",
        help="Configuration file to use (development/production).",
    ),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Trading symbols (e.g., BTC/USDT)."
    ),
    timeframes: Optional[List[str]] = typer.Option(
        None, "--timeframe", "-t", help="Timeframes for trading (e.g., 1h)."
    ),
    strategies: Optional[List[str]] = typer.Option(
        None,
        "--strategy",
        "-st",
        help="Strategies to run (e.g., Moving_Average).",
    ),
    models: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="ML models to use (comma-separated if multiple, e.g., model1,model2).",
    ),
    db: Optional[str] = typer.Option(
        "postgres", "--db", help="Database backend to use (postgres/mongo)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Enable dry run mode (no actual trades)."
    ),
):
    """
    Starts the live trading system.
    """
    os.environ["ENVIRONMENT"] = config

    config_manager = ConfigManager()
    config_data = config_manager.get_config()

    setup_logging(config_data)

    async def main():
        # Initialize and start trading system
        trading_system = LiveTradingSystem(
            exchange_id=config_data["EXCHANGE_ID"],
            symbols=symbols or config_data["TRADING_SYMBOLS"],
            timeframes=timeframes or config_data["TIMEFRAMES"],
            dry_run=dry_run,
        )
        await trading_system.start()

    asyncio.run(main())

@app.command()
def run_backtest(
    start_date: str = typer.Option(
        ...,
        "--start-date",
        "-sd",
        help="Start date for backtesting (YYYY-MM-DD).",
    ),
    end_date: str = typer.Option(
        ..., "--end-date", "-ed", help="End date for backtesting (YYYY-MM-DD)."
    ),
    config: Optional[str] = typer.Option(
        "development",
        "--config",
        "-c",
        help="Configuration file to use (development/production).",
    ),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Trading symbols (e.g., BTC/USDT)."
    ),
    timeframes: Optional[List[str]] = typer.Option(
        None, "--timeframe", "-t", help="Timeframes for trading (e.g., 1h)."
    ),
    strategies: Optional[List[str]] = typer.Option(
        None,
        "--strategy",
        "-st",
        help="Strategies to backtest (e.g., Moving_Average).",
    ),
    db: Optional[str] = typer.Option(
        "postgres", "--db", help="Database backend to use (postgres/mongo)."
    ),
    initial_capital: float = typer.Option(
        10000.0, "--capital", help="Initial capital for backtesting."
    ),
    output_format: str = typer.Option(
        "json", "--output", "-o", help="Output format (json, html)."
    )
):
    """
    Runs backtesting for the given strategies and period.
    """
    os.environ["ENVIRONMENT"] = config

    config_manager = ConfigManager()
    config_data = config_manager.get_config()

    setup_logging(config_data)

    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        typer.echo(
            "Invalid date format. Please use YYYY-MM-DD.", err=True
        )
        raise typer.Exit(code=1)

    # Import and register strategies dynamically
    strategy_registry = StrategyRegistry()
    from models.traditional.rsi_macd import RSIMACDStrategy
    from models.traditional.moving_average import MovingAverageStrategy

    strategy_registry.register_strategy("RSI_MACD", RSIMACDStrategy)
    strategy_registry.register_strategy("Moving_Average", MovingAverageStrategy)

    # Initialize and run backtesting
    from trading.backtesting import BacktestEngine, BacktestConfiguration
    from analytics.reporting import StrategyReporter
    from analytics.performance import PerformanceAnalyzer
    from utils.helpers import StrategyReporter
    from models.ml.model_registry import ModelRegistry
    from models.traditional.rsi_macd import RSIMACDStrategy
    from models.traditional.moving_average import MovingAverageStrategy
    
    backtest_config = BacktestConfiguration(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=symbols or config_data["TRADING_SYMBOLS"],
        timeframes=timeframes or config_data["TIMEFRAMES"],
    )
    backtest_engine = BacktestEngine(backtest_config)

    backtest_results = {}
    for strategy_name in strategies:
        strategy_class = strategy_registry.get_strategy(strategy_name)
        if strategy_class:
            strategy = strategy_class()
            for symbol in symbols or backtest_config.symbols:
                for timeframe in timeframes or backtest_config.timeframes:
                    result = backtest_engine.run_backtest(
                        strategy, symbol, timeframe
                    )
                    backtest_results[f"{strategy_name}_{symbol}_{timeframe}"] = result
        else:
            typer.echo(f"Strategy '{strategy_name}' not found.")

    # Generate a report
    reporter = StrategyReporter(strategies_data=strategies_data)
    if output_format == "html":
        reporter.create_performance_dashboard()
    elif output_format == "json":
        reporter.generate_summary_report()

    typer.echo(f"Backtesting complete. Results: {backtest_results}")

@app.command()
def list_strategies():
    """
    Lists all registered strategies.
    """
    registry = StrategyRegistry()
    strategies = registry.list_strategies()
    typer.echo("Available strategies:")
    for strategy in strategies:
        typer.echo(f"- {strategy}")

@app.command()
def list_models():
    """
    Lists all registered ML models.
    """
    registry = ModelRegistry()
    models = registry.list_models()
    typer.echo("Available ML models:")
    for model in models:
        typer.echo(f"- {model}")

@app.command()
def init_db():
    """
    Initializes the database.
    """
    from core.database_interactor import DatabaseInteractor
    from core.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    db_interactor = DatabaseInteractor(config['POSTGRES_URL'])
    db_interactor.create_tables()

# Ensure this is called when the script is run
if __name__ == "__main__":
    app()
