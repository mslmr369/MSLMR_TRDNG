# strategies/traditional/__init__.py
# Import the strategy implementations in the traditional module
from .rsi_macd import RSIMACDStrategy
from .moving_average import MovingAverageStrategy

__all__ = [
    'RSIMACDStrategy',
    'MovingAverageStrategy'
]
