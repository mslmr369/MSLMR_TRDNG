import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple
from models.base import BaseModel

class TimeSeriesForecaster(BaseModel):
    """
    Modelo de predicción de series temporales usando redes neuronales
    Soporta múltiples arquitecturas y configuraciones
    """
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 60,
        forecast_horizon: int = 1,
        features: List[str] = None,
        scaler: Optional[MinMaxScaler] = None
    ):
        """
        Inicializa el modelo de forecasting

        :param model_type: Tipo de modelo (lstm, gru, transformer)
        :param sequence_length: Longitud de secuencia de entrada
        :param forecast_horizon: Número de pasos a predecir
        :param features: Características para predicción
        :param scaler: Objeto MinMaxScaler para escalar datos
        """
        super().__init__()
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.features = features or ['close']

        self.model = None
        self.scaler = scaler or MinMaxScaler()
        self.is_trained = False

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento de modelo

        :param data: DataFrame con datos
        :return: X (secuencias), y (objetivo)
        """
        # Seleccionar características
        X_data = data[self.features]
        y_data = data['close']

        # Escalar datos
        X_scaled = self.scaler.fit_transform(X_data)

        # Preparar secuencias
        X, y = [], []
        for i in range(len(X_scaled) - self.sequence_length - self.forecast_horizon + 1):
            X.append(X_scaled[i:i+self.sequence_length])
            y.append(y_data.iloc[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon].values)

        return np.array(X), np.array(y)

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Construye modelo LSTM

        :param input_shape: Forma de entrada
        :return: Modelo Keras
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(self.forecast_horizon)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def _build_gru_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Construye modelo GRU

        :param input_shape: Forma de entrada
        :return: Modelo Keras
        """
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(self.forecast_horizon)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Construye modelo Transformer

        :param input_shape: Forma de entrada
        :return: Modelo Keras
        """
        inputs = tf.keras.Input(shape=input_shape)

        # Layers de embeddings
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(inputs, inputs)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(
            self.forecast_horizon,
            activation='linear'
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(
        self,
        data: pd.DataFrame,
        target: str = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Entrena el modelo de forecasting

        :param data: DataFrame con datos
        :param target: This parameter is not used in this model but kept for compatibility with BaseModel, it uses 'close' as target.
        :param epochs: Número de épocas
        :param batch_size: Tamaño de lote
        :param validation_split: Proporción de datos para validación
        """
        # Preparar datos
        X, y = self._prepare_data(data)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split
        )

        # Construir modelo según tipo
        if self.model_type == 'lstm':
            self.model = self._build_lstm_model(X_train.shape[1:])
        elif self.model_type == 'gru':
            self.model = self._build_gru_model(X_train.shape[1:])
        elif self.model_type == 'transformer':
            self.model = self._build_transformer_model(X_train.shape[1:])
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

        # Entrenar
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        self.is_trained = True
        return history

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        # Preparar datos de entrada
        X_input = self.scaler.transform(data[self.features].iloc[-self.sequence_length:])
        X_input = X_input.reshape(1, *X_input.shape)
        
        # Predecir
        prediction = self.model.predict(X_input)
        
        # Invertir escalado
        # Create a zero-filled array with the same number of samples and features as the input data
        dummy = np.zeros((prediction.shape[0], len(self.features)))
        # Replace the first column with the prediction
        dummy[:, 0] = prediction.flatten()
        # Invert the scaling
        prediction_original_scale = self.scaler.inverse_transform(dummy)[:, 0]
    
        return prediction_original_scale

    def save(self, filepath: str):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True

    def evaluate(self, data: pd.DataFrame, target: str = None, metrics: List[str] = ['mse', 'mae']) -> Dict[str, Any]:
        """
        Evalúa el modelo de forecasting

        :param data: DataFrame con datos
        :param target: This parameter is not used in this model but kept for compatibility with BaseModel, it uses 'close' as target.
        :param metrics: Lista de métricas a evaluar
        :return: Diccionario con los resultados de la evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")

        X, y = self._prepare_data(data)
        y_pred = self.model.predict(X)

        results = {}
        if 'mse' in metrics:
            results['mse'] = tf.keras.metrics.mean_squared_error(
                y.flatten(), y_pred.flatten()
            ).numpy()
        if 'mae' in metrics:
            results['mae'] = tf.keras.metrics.mean_absolute_error(
                y.flatten(), y_pred.flatten()
            ).numpy()

        return results
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Obtiene los hiperparámetros del modelo.

        :return: Diccionario con los hiperparámetros del modelo
        """
        hyperparameters = {
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'features': self.features
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
