import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Importaciones de modelos
from models.ml.forecasting import TimeSeriesForecaster
from models.ml.neural_networks import MarketNeuralNetwork

# Importaciones de utilidades
from utils.validators import DataValidator
from core.logging_system import setup_logging

class ModelTrainer:
    """
    Clase para entrenar y gestionar modelos de machine learning para trading
    """
    
    def __init__(
        self, 
        output_dir: str = './trained_models',
        log_level: str = 'INFO'
    ):
        """
        Inicializa el entrenador de modelos
        
        :param output_dir: Directorio para guardar modelos entrenados
        :param log_level: Nivel de logging
        """
        # Configurar logging
        self.logger = setup_logging()
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Crear directorio de salida
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_training_data(
        self, 
        data_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carga datos de entrenamiento
        
        :param data_path: Ruta del archivo de datos
        :return: DataFrame con datos de entrenamiento
        """
        try:
            if not data_path:
                # Generar datos sintéticos si no se proporciona archivo
                return self._generate_synthetic_data()
            
            # Cargar datos desde archivo
            return pd.read_csv(data_path, parse_dates=True, index_col='timestamp')
        
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            raise
    
    def _generate_synthetic_data(
        self, 
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Genera datos sintéticos para entrenamiento
        
        :param num_samples: Número de muestras a generar
        :return: DataFrame con datos sintéticos
        """
        dates = pd.date_range(start='2020-01-01', periods=num_samples)
        np.random.seed(42)
        
        # Simular precios y volúmenes
        close_prices = np.cumsum(np.random.normal(0, 1, num_samples)) + 100
        volume = np.random.random(num_samples) * 1000000
        
        # Calcular objetivo de predicción (movimiento de precio)
        next_price_movement = np.diff(close_prices)
        next_price_movement = np.append(next_price_movement, 0)
        
        df = pd.DataFrame({
            'close': close_prices,
            'volume': volume,
            'rsi': self._calculate_rsi(close_prices),
            'macd_line': self._calculate_macd(close_prices)['macd_line'],
            'next_price_movement': next_price_movement
        }, index=dates)
        
        return df
    
    def _calculate_rsi(
        self, 
        prices: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        Calcula RSI para datos sintéticos
        
        :param prices: Precios de cierre
        :param period: Período de RSI
        :return: Serie de RSI
        """
        delta = np.diff(prices)
        gain = (delta * (delta > 0)).rolling(window=period).mean()
        loss = (-delta * (delta < 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self, 
        prices: np.ndarray, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, np.ndarray]:
        """
        Calcula MACD para datos sintéticos
        
        :param prices: Precios de cierre
        :return: Diccionario con líneas MACD
        """
        # Implementación simplificada de MACD
        fast_ema = pd.Series(prices).ewm(span=fast_period, adjust=False).mean()
        slow_ema = pd.Series(prices).ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return {
            'macd_line': macd_line.values,
            'signal_line': signal_line.values
        }
    
    def train_forecasting_model(
        self, 
        data: pd.DataFrame,
        model_type: str = 'lstm',
        features: Optional[List[str]] = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Entrena modelo de forecasting
        
        :param data: Datos de entrenamiento
        :param model_type: Tipo de modelo
        :param features: Características a usar
        :param epochs: Número de épocas
        :param batch_size: Tamaño de lote
        :return: Modelo entrenado
        """
        features = features or ['close', 'volume']
        
        forecaster = TimeSeriesForecaster(
            model_type=model_type, 
            sequence_length=30, 
            forecast_horizon=5,
            features=features
        )
        
        # Entrenar modelo
        history = forecaster.train(
            data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Guardar modelo
        model_path = os.path.join(
            self.output_dir, 
            f'forecasting_model_{model_type}.h5'
        )
        forecaster.save_model(model_path)
        
        self.logger.info(f"Modelo {model_type} entrenado y guardado en {model_path}")
        
        return forecaster
    
    def train_classification_model(
        self, 
        data: pd.DataFrame,
        model_type: str = 'dense',
        features: Optional[List[str]] = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Entrena modelo de clasificación de movimiento de precio
        
        :param data: Datos de entrenamiento
        :param model_type: Tipo de modelo
        :param features: Características a usar
        :param epochs: Número de épocas
        :param batch_size: Tamaño de lote
        :return: Modelo entrenado
        """
        features = features or ['close', 'volume', 'rsi', 'macd_line']
        
        # Preparar objetivo binario
        data['price_movement_class'] = (
            data['next_price_movement'] > 0
        ).astype(int)
        
        classifier = MarketNeuralNetwork(
            task='classification', 
            model_type=model_type,
            input_features=features
        )
        
        # Entrenar modelo
        history = classifier.train(
            data, 
            target='price_movement_class',
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Guardar modelo
        model_path = os.path.join(
            self.output_dir, 
            f'classification_model_{model_type}.h5'
        )
        classifier.save_model(model_path)
        
        self.logger.info(f"Modelo de clasificación {model_type} entrenado y guardado en {model_path}")
        
        return classifier

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenador de Modelos de Machine Learning para Trading')
    parser.add_argument(
        '--data', 
        type=str, 
        help='Ruta del archivo de datos de entrenamiento'
    )
    parser.add_argument(
        '--model-type', 
        type=str, 
        choices=['forecasting', 'classification'], 
        default='forecasting',
        help='Tipo de modelo a entrenar'
    )
    parser.add_argument(
        '--nn-type', 
        type=str, 
        choices=['lstm', 'dense'], 
        default='lstm',
        help='Tipo de red neuronal'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help='Número de épocas de entrenamiento'
    )
    
    args = parser.parse_args()
    
    # Inicializar entrenador
    trainer = ModelTrainer()
    
    # Cargar datos
    data = trainer.load_training_data(args.data)
    
    # Entrenar modelo
    if args.model_type == 'forecasting':
        trainer.train_forecasting_model(
            data, 
            model_type=args.nn_type,
            epochs=args.epochs
        )
    else:
        trainer.train_classification_model(
            data, 
            model_type=args.nn_type,
            epochs=args.epochs
        )

if __name__ == '__main__':
    main()
