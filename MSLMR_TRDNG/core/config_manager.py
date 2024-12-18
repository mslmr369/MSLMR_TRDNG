import os
from typing import Dict, Any

from core.config import get_config

class ConfigManager:
    """
    Manages the application's configuration.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None  # Initialize _config attribute here
        return cls._instance

    def __init__(self):
        if self._config is None:  # Check if _config has been initialized
            environment = os.getenv('ENVIRONMENT', 'development')
            self._config = get_config(environment)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration dictionary.
        """
        return self._config.get_config_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a specific configuration value.

        :param key: The configuration key.
        :param default: The default value if the key is not found.
        :return: The configuration value.
        """
        return getattr(self._config, key, default)
