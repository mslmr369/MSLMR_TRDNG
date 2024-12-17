import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple

class MarketNeuralNetwork:
    """
    Modelo de red neuronal para predicción y clasificación de mercados
    Soporta múltiples arquitecturas y tareas
    """
    def __init__(
        self, 
        task: str = 'classification',
        model_type: str = 'dense',
        input_features: List[str] = None,
        output_size: int = 1
    ):
        """
        Inicializa el modelo de red neuronal
        
        :param task: Tarea ('classification', 'regression')
        :param model_type: Tipo de modelo ('dense', 'lstm', 'transformer')
        :param input_features: Características de entrada
        :param output_size: Tamaño de salida
        """
        self.task = task
        self.model_type = model_type
        self.input_features = input_features or ['close', 'volume']
        self.output_size = output_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        target: str = 'next_price_movement'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento
        
        :param data: DataFrame con datos
        :param target: Columna objetivo
        :return: X (características), y (objetivo)
        """
        # Seleccionar características
        X = data[self.input_features]
        
        # Preparar objetivo según tarea
        if self.task == 'classification':
            y = (data[target] > 0).astype(int)
        else:  # Regresión
            y = data[target]
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def _build_dense_model(
        self, 
        input_shape: Tuple[int]
    ) -> tf.keras.Model:
        """
        Construye modelo de red neuronal densa
        
        :param input_shape: Forma de entrada
        :return: Modelo Keras
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2)
        ])
        
        # Capa de salida según tarea
        if self.task == 'classification':
            model.add(tf.keras.layers.Dense(
                self.output_size, 
                activation='sigmoid'
            ))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(tf.keras.layers.Dense(
                self.output_size, 
                activation='linear'
            ))
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer='adam', 
            loss=loss, 
            metrics=metrics
        )
        
        return model
    
    def _build_lstm_model(
        self, 
        input_shape: Tuple[int, int]
    ) -> tf.keras.Model:
        """
        Construye modelo LSTM
        
        :param input_shape: Forma de entrada
        :return: Modelo Keras
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2)
        ])
        
        # Capa de salida según tarea
        if self.task == 'classification':
            model.add(tf.keras.layers.Dense(
                self.output_size, 
                activation='sigmoid'
            ))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(tf.keras.layers.Dense(
                self.output_size, 
                activation='linear'
            ))
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer='adam', 
            loss=loss, 
            metrics=metrics
        )
        
        return model
    
    def train(
        self, 
        data: pd.DataFrame, 
        target: str = 'next_price_movement',
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Entrena el modelo de red neuronal
        
        :param data: DataFrame con datos
        :param target: Columna objetivo
        :param epochs: Número de épocas
        :param batch_size: Tamaño de lote
        :param validation_split: Proporción de datos para validación
        """
        # Preparar datos
        X, y = self._prepare_data(data, target)
        
        # Ajustar forma de entrada según tipo de modelo
        if self.model_type == 'lstm':
            # Reshape para LSTM: (samples, time steps, features)
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            input_shape = (1, X.shape[2])
        else:
            input_shape = (X.shape[1],)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split
        )
        
        # Construir modelo
        if self.model_type == 'dense':
            self.model = self._build_dense_model(input_shape)
        elif self.model_type == 'lstm':
            self.model = self._build_lstm_model(input_shape)
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
    
    def predict(
        self, 
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Realiza predicciones
        
        :param data: DataFrame con datos
        :return: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        # Preparar datos de entrada
        X_input = self.scaler.transform(data[self.input_features])
        
        # Ajustar forma según tipo de modelo
        if self.model_type == 'lstm':
            X_input = X_input.reshape((X_input.shape[0], 1, X_input.shape[1]))
        
        # Predecir
        return self.model.predict(X_input)
    
    def evaluate(
        self, 
        data: pd.DataFrame, 
        target: str = 'next_price_movement'
    ) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo
        
        :param data: DataFrame con datos
        :param target: Columna objetivo
        :return: Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        # Preparar datos
        X, y = self._prepare_data(data, target)
        
        # Ajustar forma según tipo de modelo
        if self.model_type == 'lstm':
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Evaluar
        metrics = self.model.evaluate(X, y, verbose=0)
        
        if self.task == 'classification':
            return {
                'loss': metrics[0],
                'accuracy': metrics[1]
            }
        else:
            return {
                'loss': metrics[0],
                'mae': metrics[1]
            }
    
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
    # Generar datos sintéticos
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-01-01')
    
    # Simular datos de mercado
    close_prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
    volume = np.random.random(len(dates)) * 1000000
    next_price_movement = np.diff(close_prices)
    next_price_movement = np.append(next_price_movement, 0)
    
    df = pd.DataFrame({
        'close': close_prices,
        'volume': volume,
        'next_price_movement': next_price_movement
    }, index=dates)
    
    # Inicializar modelo para clasificación
    classifier = MarketNeuralNetwork(
        task='classification', 
        model_type='dense',
        input_features=['close', 'volume']
    )
    
    # Entrenar modelo de clasificación
    classification_history = classifier.train(
        df, 
        target='next_price_movement', 
        epochs=20
    )
    
    # Evaluar modelo de clasificación
    classification_metrics = classifier.evaluate(df)
    print("Métricas de Clasificación:", classification_metrics)
    
    # Inicializar modelo para regresión
    regressor = MarketNeuralNetwork(
        task='regression', 
        model_type='lstm',
        input_features=['close', 'volume']
    )
    
    # Entrenar modelo de regresión
    regression_history = regressor.train(
        df, 
        target='next_price_movement', 
        epochs=20
    )
    
    # Evaluar modelo de regresión
    regression_metrics = regressor.evaluate(df)
    print("Métricas de Regresión:", regression_metrics)
    
    # Predecir
    predictions = regressor.predict(df.head())
    print("Predicciones:", predictions)

if __name__ == "__main__":
    main()
