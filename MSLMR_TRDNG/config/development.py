from .base import BaseConfig

class DevelopmentConfig(BaseConfig):
    """Configuración específica para desarrollo"""
        DEBUG = True
	    TESTING = True

	        # Sobreescribir configuraciones base para desarrollo
		    TRADING_SYMBOLS = ['BTC/USDT']  # Símbolos limitados
		        TIMEFRAMES = ['1m', '5m', '1h']  # Intervalos reducidos

			    # Configuraciones especiales de desarrollo
			        SIMULATE_TRADING = True
				    USE_MOCK_EXCHANGE = True

				        @classmethod
					    def is_production(cls) -> bool:
					            return False
