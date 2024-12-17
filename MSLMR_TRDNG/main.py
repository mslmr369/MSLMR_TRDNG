import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def select_config():
    """Selecciona la configuración según el entorno"""
        env = os.getenv('ENVIRONMENT', 'development').lower()
	    
	        if env == 'production':
		        from config.production import ProductionConfig
			        return ProductionConfig
				    elif env == 'testing':
				            from config.testing import TestingConfig
					            return TestingConfig
						        else:
							        from config.development import DevelopmentConfig
								        return DevelopmentConfig

									def main():
									    # Seleccionar configuración
									        config = select_config()
										    
										        # Configurar logging
											    from core.logging_system import setup_logging
											        logger = setup_logging(config)
												    
												        try:
													        # Inicializar componentes principales
														        logger.info(f"Iniciando {config.PROJECT_NAME}")
															        
																        # Aquí irá la lógica principal de inicio del sistema
																	        from core.database import initialize_databases
																		        initialize_databases(config)
																			        
																				        # Iniciar sistema de trading
																					        from trading.live_trading import start_trading_system
																						        start_trading_system(config)
																							    
																							        except Exception as e:
																								        logger.exception(f"Error crítico al iniciar {config.PROJECT_NAME}")
																									        sys.exit(1)

																										if __name__ == "__main__":
																										    main()
