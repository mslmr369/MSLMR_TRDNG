from typing import Dict, Type, Any, Optional, List
from models.base import BaseModel
from core.logging_system import LoggingMixin

class ModelRegistry(LoggingMixin):
    """
    Registro centralizado de modelos de machine learning
    """
    _instance = None
    _models: Dict[str, Type[BaseModel]] = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type[BaseModel]
    ):
        """
        Registra un modelo en el sistema

        :param name: Nombre del modelo
        :param model_class: Clase del modelo
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("El modelo debe heredar de BaseModel")

        cls._models[name] = model_class
        cls._instance.logger.info(f"Modelo registrado: {name}")

    @classmethod
    def get_model(
        cls,
        name: str
    ) -> Optional[Type[BaseModel]]:
        """
        Obtiene un modelo registrado

        :param name: Nombre del modelo
        :return: Clase del modelo o None
        """
        model = cls._models.get(name)
        if not model:
            cls._instance.logger.warning(f"Modelo no encontrado: {name}")
        return model

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Lista todos los modelos registrados

        :return: Lista de nombres de modelos
        """
        return list(cls._models.keys())

    @classmethod
    def create_model(
        cls,
        name: str,
        **kwargs
    ) -> Optional[BaseModel]:
        """
        Crea una instancia de modelo

        :param name: Nombre del modelo
        :param kwargs: Parámetros para inicializar el modelo
        :return: Instancia de modelo
        """
        model_class = cls.get_model(name)
        if model_class:
            return model_class(**kwargs)
        return None

    @classmethod
    def remove_model(cls, name: str):
        """
        Elimina un modelo del registro

        :param name: Nombre del modelo a eliminar
        """
        if name in cls._models:
            del cls._models[name]
            cls._instance.logger.info(f"Modelo eliminado: {name}")
