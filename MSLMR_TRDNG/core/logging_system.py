import os
import logging
import structlog
from pythonjsonlogger import jsonlogger
import socket
from typing import Any, Dict

def setup_logging(config):
    """
    Configura el sistema de logging con salida estructurada
    """
    # Crear directorio de logs si no existe
    os.makedirs(config.LOG_PATH, exist_ok=True)

    # Configuración de logging estructurado
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configurar manejadores de log
    def create_log_handler(filename: str, log_level: int):
        """Crear manejador de logs para archivo"""
        file_handler = logging.FileHandler(
            os.path.join(config.LOG_PATH, filename)
        )
        file_handler.setLevel(log_level)
        
        # Formatter JSON
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d'
        )
        file_handler.setFormatter(json_formatter)
        
        return file_handler

    # Logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

    # Manejadores de log
    handlers = [
        create_log_handler('system.log', logging.INFO),
        create_log_handler('error.log', logging.ERROR)
    ]

    # Añadir manejadores de consola y archivo
    for handler in handlers:
        root_logger.addHandler(handler)

    # Añadir manejador de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        structlog.dev.ConsoleRenderer(colors=True)
    )
    root_logger.addHandler(console_handler)

    # Contexto de logging
    def add_log_context(logger, method_name, event_dict):
        """Añadir contexto adicional a los logs"""
        event_dict['hostname'] = socket.gethostname()
        event_dict['environment'] = 'development' if not config.is_production() else 'production'
        return event_dict

    # Añadir procesador de contexto
    structlog.configure(
        processors=[
            add_log_context,
            structlog.processors.JSONRenderer()
        ]
    )

    return structlog.get_logger()

class LoggingMixin:
    """
    Mixin para añadir logging a cualquier clase
    """
    @property
    def logger(self):
        """Devuelve un logger específico para la instancia"""
        name = '.'.join([
            self.__class__.__module__, 
            self.__class__.__name__
        ])
        return structlog.get_logger(name)

def log_method(log_level='info', message=None):
    """
    Decorador para añadir logging a métodos
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = structlog.get_logger(func.__module__)
            log_method = getattr(logger, log_level)
            
            try:
                result = func(*args, **kwargs)
                
                # Mensaje personalizado o por defecto
                log_msg = message or f"Ejecutando {func.__name__}"
                log_method(log_msg, 
                           function=func.__name__, 
                           args=args, 
                           kwargs=kwargs)
                
                return result
            except Exception as e:
                logger.exception(f"Error en {func.__name__}", 
                                 error=str(e))
                raise
        return wrapper
    return decorator
