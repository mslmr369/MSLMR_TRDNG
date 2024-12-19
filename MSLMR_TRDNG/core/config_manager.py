import os
from typing import Dict, Any
from dotenv import load_dotenv
from core.config import get_config  # Import the get_config function

class ConfigManager:
    """
    Manages the application's configuration.
    Loads settings from default configuration files, environment variables, and command-line arguments.
    Ensures sensitive information is handled securely.
    """
    _instance = None

    def __new__(cls, config_file: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance.config_file = config_file
        return cls._instance

    def __init__(self, config_file: str = None):
        if self._config is None:
            # Determine the environment
            environment = os.getenv('ENVIRONMENT', 'development').lower()

            # Get the appropriate configuration class
            config_class = get_config(environment)
            self._config = config_class()

            # Load environment variables from .env file
            load_dotenv()

            # Override with environment variables
            self._load_from_env()

    def _load_from_env(self):
        """
        Loads configuration from environment variables, overriding defaults.
        """
        for key, value in self._config.get_config_dict().items():
            env_value = os.getenv(key)
            if env_value is not None:
                # Convert to appropriate type
                if isinstance(value, bool):
                    env_value = env_value.lower() == 'true'
                elif isinstance(value, int):
                    env_value = int(env_value)
                elif isinstance(value, float):
                    env_value = float(env_value)
                elif isinstance(value, list):
                    env_value = env_value.split(',')
                setattr(self._config, key, env_value)

    def override_with_cli_args(self, args: Dict[str, Any]):
        """
        Overrides configuration with command-line arguments.

        :param args: A dictionary containing command-line arguments.
        """
        for key, value in args.items():
            if hasattr(self._config, key.upper()) and value is not None:
                setattr(self._config, key.upper(), value)

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

