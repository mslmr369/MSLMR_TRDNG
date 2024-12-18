# core/models/__init__.py
# Import the model classes here, so they are recognized by Alembic

from .postgres_models import Base, Price, Indicator, Signal, Trade, Backtest, Alert
# Import other model classes from elastic_models.py and mongo_models.py when needed

__all__ = [
    'Base',
    'Price',
    'Indicator',
    'Signal',
    'Trade',
    'Backtest',
    'Alert'
]
