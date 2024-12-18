# strategies/__init__.py
# Import the base strategy class
from .base import BaseStrategy

# Import the StrategyRegistry class
from .strategy_registry import StrategyRegistry

# Import concrete strategy implementations
from .traditional.rsi_macd import RSIMACDStrategy
from .traditional.moving_average import MovingAverageStrategy

__all__ = [
    'BaseStrategy',
    'StrategyRegistry',
    'RSIMACDStrategy',
    'MovingAverageStrategy',
    # Add other concrete strategies here if they are created in the future
]
