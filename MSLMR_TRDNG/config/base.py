import os
from typing import Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class BaseConfig:
    """Configuración base para la aplicación de trading"""
        
	    # Configuraciones generales
	        PROJECT_NAME = "MSLMR_TRDNG"
		    DEBUG = False
		        TESTING = False

			    # Bases de datos
			        POSTGRES_URL = os.getenv('DATABASE_URL')
				    REDIS_URL = os.getenv('REDIS_URL')

				        # Configuraciones de trading
					    EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
					        EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET')
						    
						        # Símbolos para trading
							    TRADING_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
							        
								    # Intervalos de tiempo
								        TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

									    # Gestión de riesgos
									        MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.02))
										    MAX_SINGLE_TRADE_RISK = float(os.getenv('MAX_SINGLE_TRADE_RISK', 0.01))

										        # Logging
											    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
											        LOG_PATH = os.getenv('LOG_PATH', './logs/')

												    @classmethod
												        def is_production(cls) -> bool:
													        return False

														    @classmethod
														        def get_config_dict(cls) -> Dict[str, Any]:
															        """Devuelve un diccionario con toda la configuración"""
																        return {k: v for k, v in cls.__dict__.items() 
																	                if not k.startswith('__') and not callable(v)}
                                                                         # Flags for enabling/disabling components
    ENABLE_DATA_INGESTION: bool = True
    ENABLE_LIVE_TRADING: bool = True
    ENABLE_BACKTESTING: bool = True
    ENABLE_DASHBOARD: bool = True
    ENABLE_ALERTS: bool = True
    ENABLE_ML_MODELS: bool = True
