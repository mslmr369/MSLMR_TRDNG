import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseConfig:
    """Base configuration"""
    PROJECT_NAME: str = "MSLMR_TRDNG"
    DEBUG: bool = False
    TESTING: bool = False
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your_secret_key')
    # Database
    POSTGRES_URL: str = os.getenv('DATABASE_URL')
    REDIS_URL: str = os.getenv('REDIS_URL')

    # Trading
    EXCHANGE_API_KEY: str = os.getenv('EXCHANGE_API_KEY')
    EXCHANGE_API_SECRET: str = os.getenv('EXCHANGE_API_SECRET')
    TRADING_SYMBOLS: List[str] = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    TIMEFRAMES: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']
    RISK_PER_TRADE: float = 0.01  # 1% of capital per trade
    STOP_LOSS_MULTIPLIER: float = 1.5  # Stop loss at 1.5 times ATR
    TAKE_PROFIT_MULTIPLIER: float = 2.0  # Take profit at 2 times ATR

    # Risk Management
    MAX_PORTFOLIO_RISK: float = 0.02
    MAX_SINGLE_TRADE_RISK: float = 0.01

    # Alerts (Example configuration)
    TELEGRAM_TOKEN: str = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID')
    EMAIL_HOST: str = os.getenv('EMAIL_HOST')
    EMAIL_PORT: int = int(os.getenv('EMAIL_PORT', 587))
    EMAIL_USER: str = os.getenv('EMAIL_USER')
    EMAIL_PASSWORD: str = os.getenv('EMAIL_PASSWORD')
    EMAIL_RECIPIENTS: List[str] = []  # Add recipient emails here

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_PATH: str = os.getenv('LOG_PATH', './logs')

    @classmethod
    def is_production(cls) -> bool:
        return False

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Returns a dictionary with all configuration values."""
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v)}

class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(BaseConfig):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'
    # Add production-specific settings here

class TestingConfig(BaseConfig):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    # Add testing-specific settings here

def get_config(env: str = None) -> type[BaseConfig]:
    """
    Returns the appropriate configuration class based on the environment.
    """
    env = env or os.getenv('ENVIRONMENT', 'development').lower()
    if env == 'production':
        return ProductionConfig
    elif env == 'testing':
        return TestingConfig
    else:
        return DevelopmentConfig

# Example usage (you can adapt this to your needs):
config = get_config()
