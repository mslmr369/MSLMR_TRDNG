import unittest
import pandas as pd
import numpy as np
from typing import List, Dict

# Importar modelos a testear
from models.traditional.rsi_macd import RSIMACDStrategy
from models.traditional.moving_average import MovingAverageStrategy
from models.ml.forecasting import TimeSeriesForecaster

class TestTradingModels(unittest.TestCase):
    def setUp(self):
        """
        Preparar datos de prueba comunes
        """
        # Generar datos sintéticos
        dates = pd.date_range(start='2022-01-01', end='2023-01-01')
        np.random.seed(42)
        
        # Simular precios
        close_prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
        volumes = np.random.random(len(dates)) * 1000000
        
        self.test_df = pd.DataFrame({
            'close': close_prices,
            'volume': volumes,
            'high': close_prices + np.random.random(len(dates)),
            'low': close_prices - np.random.random(len(dates))
        }, index=dates)
    
    def test_rsi_macd_strategy_signal_generation(self):
        """
        Prueba la generación de señales de la estrategia RSI MACD
        """
        strategy = RSIMACDStrategy()
        
        # Generar señal con datos de prueba
        signal = strategy.generate_signal(self.test_df)
        
        # Verificaciones
        self.assertIsNotNone(signal, "La estrategia no generó señal")
        self.assertIn('type', signal, "La señal debe tener un tipo")
        self.assertIn(signal['type'], ['buy', 'sell'], "Tipo de señal inválido")
        
        # Verificar campos adicionales
        self.assertIn('entry_price', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('take_profit', signal)
    
    def test_moving_average_strategy_signal_generation(self):
        """
        Prueba la generación de señales de la estrategia de Media Móvil
        """
        strategy = MovingAverageStrategy()
        
        # Generar señal con datos de prueba
        signal = strategy.generate_signal(self.test_df)
        
        # Verificaciones
        self.assertIsNotNone(signal, "La estrategia no generó señal")
        self.assertIn('type', signal, "La señal debe tener un tipo")
        self.assertIn(signal['type'], ['buy', 'sell'], "Tipo de señal inválido")
        
        # Verificar campos adicionales
        self.assertIn('price', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('take_profit', signal)
    
    def test_time_series_forecaster(self):
        """
        Prueba el modelo de forecasting de series temporales
        """
        # Inicializar forecaster
        forecaster = TimeSeriesForecaster(
            model_type='lstm', 
            sequence_length=30, 
            forecast_horizon=5,
            features=['close', 'volume']
        )
        
        # Entrenar modelo
        history = forecaster.train(self.test_df, epochs=10)
        
        # Verificar historial de entrenamiento
        self.assertIsNotNone(history)
        self.assertTrue(hasattr(history, 'history'))
        
        # Realizar predicción
        prediction = forecaster.predict(self.test_df.tail(30))
        
        # Verificaciones de predicción
        self.assertIsNotNone(prediction)
        self.assertTrue(prediction.size > 0)
    
    def test_strategy_required_indicators(self):
        """
        Prueba los indicadores requeridos por las estrategias
        """
        strategies = [
            RSIMACDStrategy(),
            MovingAverageStrategy()
        ]
        
        for strategy in strategies:
            required_indicators = strategy.get_required_indicators()
            
            # Verificaciones
            self.assertIsInstance(required_indicators, list)
            self.assertTrue(len(required_indicators) > 0)
            
            # Verificar que los indicadores sean válidos
            valid_indicators = ['close', 'open', 'high', 'low', 'volume']
            for indicator in required_indicators:
                self.assertIn(indicator, valid_indicators)
    
    def test_ml_model_saving_loading(self):
        """
        Prueba guardado y cargado de modelos ML
        """
        import tempfile
        import os
        
        # Inicializar y entrenar modelo
        forecaster = TimeSeriesForecaster(
            model_type='lstm', 
            sequence_length=30, 
            forecast_horizon=5
        )
        forecaster.train(self.test_df, epochs=10)
        
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'test_model.h5')
            
            # Guardar modelo
            forecaster.save_model(model_path)
            
            # Verificar que el archivo existe
            self.assertTrue(os.path.exists(model_path))
            
            # Cargar modelo
            new_forecaster = TimeSeriesForecaster()
            new_forecaster.load_model(model_path)
            
            # Verificar que el modelo se cargó correctamente
            self.assertTrue(new_forecaster.is_trained)

def generate_mock_trades() -> List[Dict]:
    """
    Genera datos de trades simulados para pruebas
    """
    return [
        {
            'symbol': 'BTC/USDT',
            'timestamp': '2023-01-01',
            'side': 'buy',
            'entry_price': 50000,
            'exit_price': 51000,
            'profit_loss': 1000
        },
        {
            'symbol': 'ETH/USDT',
            'timestamp': '2023-01-02',
            'side': 'sell',
            'entry_price': 4000,
            'exit_price': 3900,
            'profit_loss': -100
        }
    ]

if __name__ == '__main__':
    unittest.main()
