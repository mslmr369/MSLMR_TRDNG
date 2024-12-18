# core/exceptions.py
class TradingException(Exception):
    """Base class for all trading related exceptions."""
    pass

class InvalidStrategyError(TradingException):
    """Raised when an invalid strategy is used."""
    pass

class DataValidationError(TradingException):
    """Raised when data validation fails."""
    pass

class RiskManagementException(TradingException):
    """Base class for exceptions related to risk management."""
    pass

class PositionSizeException(RiskManagementException):
    """Raised when there's an issue calculating position size."""
    pass

class InsufficientFundsError(TradingException):
    """Raised when there are insufficient funds for a trade."""
    pass

class OrderPlacementError(TradingException):
    """Raised when there's an issue placing an order."""
    pass

class OrderCancellationError(TradingException):
    """Raised when there's an issue canceling an order."""
    pass

class ExchangeError(TradingException):
    """Raised when there's an error communicating with the exchange."""
    pass
