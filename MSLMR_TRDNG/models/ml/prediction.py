import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
from models.base import BaseModel

class MarketNeuralNetwork(BaseModel):
    """
    Modelo de red neuronal para predicción y clasificación de mercados
    Soporta múltiples arquitecturas y tareas
    """
    def __init__(
        self,
        task: str = 'classification',
        model_type: str = 'dense',
        input_features: List[str] = None,
        output_size: int = 1,
        scaler: Optional[StandardScaler] = None
    ):
        """
        Inicializa el modelo de red neuronal

        :param task: Tarea ('classification', 'regression')
        :param model_type: Tipo de modelo ('dense', 'lstm', 'transformer')
        :param input_features: Características de entrada
        :param output_size: Tamaño de salida
        :param scaler: Objeto StandardScaler para escalar datos
        """
        super().__init__()
        self.task = task
        self.model_type = model_type
        self.input_features = input_features or ['close', 'volume']
        self.output_size = output_size

        self.model = None
        self.scaler = scaler or StandardScaler()
        self.is_trained = False

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target: str = 'next_price_movement'
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ... (Implementation remains the same) ...

    def _build_dense_model(
        self,
        input_shape: Tuple[int]
    ) -> tf.keras.Model:
        # ... (Implementation remains the same) ...

    def _build_lstm_model(
        self,
        input_shape: Tuple[int, int]
    ) -> tf.keras.Model:
        # ... (Implementation remains the same) ...

    def train(
        self,
        data: pd.DataFrame,
        target: str = 'next_price_movement',
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        # ... (Implementation remains the same) ...

    def predict(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        # ... (Implementation remains the same) ...

    def evaluate(
        self,
        data: pd.DataFrame,
        target: str = 'next_price_movement'
    ) -> Dict[str, float]:
        # ... (Implementation remains the same) ...

    def save(self, filepath: str):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Obtiene los hiperparámetros del modelo.

        :return: Diccionario con los hiperparámetros del modelo
        """
        hyperparameters = {
            'task': self.task,
            'model_type': self.model_type,
            'input_features': self.input_features,
            'output_size': self.output_size
        }
        return hyperparameters

    def set_hyperparameters(self, **kwargs):
        """
        Establece los hiperparámetros del modelo.

        :param kwargs: Diccionario con los hiperparámetros a establecer
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Advertencia: El hiperparámetro '{key}' no existe en el modelo.")

