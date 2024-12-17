import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

class TimeSeriesForecaster:
    """
    Modelo de predicción de series temporales usando redes neuronales
    Soporta múltiples arquitecturas y configuraciones
    """
    def __init__(
        self, 
        model_type: str = 'lstm',
        sequence_length: int = 60,
        forecast_horizon: int = 1,
        features: List[str] = None
    ):
        """
        Inicializa el modelo de forecasting
        
        :param model_type: Tipo de modelo (lstm, gru, transformer)
        :param sequence_length: Longitud de secuencia de entrada
        :param forecast_horizon: Número de pasos a predecir
        :param features: Características para predicción
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.features = features or ['close']
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def _prepare_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Entrena el modelo de forecasting
        
        :param data: DataFrame con datos
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
        """
        Realiza predicciones
        
        :param data: DataFrame con datos
        :return: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        # Preparar datos de entrada
        X_input = self.scaler.transform(data[self.features].iloc[-self.sequence_length:])
        X_input = X_input.reshape(1, *X_input.shape)
        
        # Predecir
        prediction = self.model.predict(X_input)
        
        # Invertir escalado
        return self.scaler.inverse_transform(prediction)
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado
        
        :param filepath: Ruta para guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Carga un modelo previamente guardado
        
        :param filepath: Ruta del modelo guardado
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True

# Ejemplo de uso
def main():
    # Cargar datos
    dates = pd.date_range(start='2022-01-01', end='2023-01-01')
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.random(len(dates))
    }, index=dates)
    
    # Inicializar y entrenar modelo
    forecaster = TimeSeriesForecaster(
        model_type='lstm', 
        sequence_length=30, 
        forecast_horizon=5,
        features=['close', 'volume']
    )
    
    history = forecaster.train(df, epochs=20)
    
    # Predecir
    prediction = forecaster.predict(df)
    print("Predicción:", prediction)
    
    # Guardar modelo
    forecaster.save_model('market_forecast_model.h5')

if __name__ == "__main__":
    main()
