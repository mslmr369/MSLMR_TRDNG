import os
from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Configuración específica para producción"""
        DEBUG = False
	    TESTING = False

	        # Símbolos y timeframes más extensos para producción
		    TRADING_SYMBOLS = [
		            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 
			            'XRP/USDT', 'ADA/USDT', 'DOT/USDT'
				        ]
					    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

					        # Configuraciones de seguridad
						    SECRET_KEY = os.getenv('SECRET_KEY')

						        # Límites de riesgo más conservadores
							    MAX_PORTFOLIO_RISK = 0.015  # 1.5%
							        MAX_SINGLE_TRADE_RISK = 0.005  # 0.5%

								    # Configuraciones de logging más detalladas
								        LOG_LEVEL = 'WARNING'

									    @classmethod
									        def is_production(cls) -> bool:
										        return True
