import unittest
import pandas as pd
import numpy as np
from typing import List, Dict

# Importar estrategias y componentes a testear
from strategies.base import BaseStrategy
from strategies.strategy_registry import StrategyRegistry
from strategies.portfolio_manager import PortfolioManager
from strategies.risk_management import RiskManager

# Estrategias de ejemplo para pruebas
from models.traditional.rsi_macd import RSIMACDStrategy
from models.traditional.moving_average import MovingAverageStrategy

class TestStrategies(unittest.TestCase):
    def setUp(self):
        """
        Preparar datos y componentes para pruebas
        """
        # Generar datos sintéticos
        dates = pd.date_range(start='2022-01-01', end='2023-01-01')
        np.random.seed(42)
        
        close_prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
        volumes = np.random.random(len(dates)) * 1000000
        
        self.test_df = pd.DataFrame({
            'close': close_prices,
            'volume': volumes,
            'high': close_prices + np.random.random(len(dates)),
            'low': close_prices - np.random.random(len(dates))
        }, index=dates)
        
        # Inicializar componentes
        self.strategy_registry = StrategyRegistry()
        self.portfolio_manager = PortfolioManager(initial_capital=10000)
        self.risk_manager = RiskManager()
    
    def test_strategy_registry_registration(self):
        """
        Prueba el registro de estrategias
        """
        # Registrar estrategias
        self.strategy_registry.register_strategy('RSI_MACD', RSIMACDStrategy)
        self.strategy_registry.register_strategy('Moving_Average', MovingAverageStrategy)
        
        # Verificaciones
        strategies = self.strategy_registry.list_strategies()
        self.assertIn('RSI_MACD', strategies)
        self.assertIn('Moving_Average', strategies)
        
        # Obtener estrategia
        strategy_class = self.strategy_registry.get_strategy('RSI_MACD')
        self.assertIsNotNone(strategy_class)
        
        # Crear instancia de estrategia
        strategy = self.strategy_registry.create_strategy('RSI_MACD')
        self.assertIsInstance(strategy, RSIMACDStrategy)
    
    def test_strategy_parameter_management(self):
        """
        Prueba la gestión de parámetros de estrategias
        """
        strategy = RSIMACDStrategy()
        
        # Establecer parámetros
        strategy.set_parameters(
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70
        )
        
        # Verificar parámetros
        parameters = strategy.get_parameters()
        self.assertEqual(parameters.get('rsi_period'), 14)
        self.assertEqual(parameters.get('rsi_oversold'), 30)
        self.assertEqual(parameters.get('rsi_overbought'), 70)
        
        # Validar parámetros
        self.assertTrue(strategy.validate_parameters())
    
    def test_portfolio_manager_position_sizing(self):
        """
        Prueba el cálculo de tamaño de posición
        """
        # Caso de prueba
        current_price = 50000
        stop_loss_price = 49000
        
        # Calcular tamaño de posición
        position_size = self.portfolio_manager.calculate_position_size(
            current_price, 
            stop_loss_price
        )
        
        # Verificaciones
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, self.portfolio_manager.initial_capital)
    
    def test_risk_manager_trade_validation(self):
        """
        Prueba la validación de riesgo de trades
        """
        entry_price = 50000
        stop_loss = 49000
        position_size = 0.1
        account_balance = 10000
        
        # Validar riesgo del trade
        is_valid = self.risk_manager.validate_trade_risk(
            entry_price, 
            stop_loss, 
            position_size, 
            account_balance
        )
        
        self.assertTrue(is_valid)
    
    def test_strategy_trade_simulation(self):
        """
        Prueba simulación de trades para estrategias
        """
        strategies = [
            RSIMACDStrategy(),
            MovingAverageStrategy()
        ]
        
        for strategy in strategies:
            # Generar señales
            signal = strategy.generate_signal(self.test_df)
            
            if signal:
                # Verificar estructura de señal
                self.assertIn('type', signal)
[O                self.assertIn(signal['type'], ['buy', 'sell'])
                self.assertIn('entry_price', signal)
                self.assertIn('stop_loss', signal)
                self.assertIn('take_profit', signal)
    
    def test_kelly_criterion(self):
        """
        Prueba el cálculo del criterio de Kelly
        """
        # Caso de prueba
        win_rate = 0.55
        win_loss_ratio = 2.0
        
        # Calcular fracción de Kelly
        kelly_fraction = self.risk_manager.calculate_kelly_criterion(
            win_rate, 
            win_loss_ratio
        )
        
        # Verificaciones
        self.assertGreaterEqual(kelly_fraction, 0)
        self.assertLessEqual(kelly_fraction, 1)
    
    def test_portfolio_drawdown_analysis(self):
        """
        Prueba el análisis de drawdown del portfolio
        """
        # Datos de trades simulados
        trades = [
            {'profit_loss': 100},
            {'profit_loss': -50},
            {'profit_loss': 75},
            {'profit_loss': -25}
        ]
        
        # Simular drawdown
        drawdown_metrics = self.risk_manager.simulate_monte_carlo_drawdown(trades)
        
        # Verificaciones
        self.assertIn('max_drawdown', drawdown_metrics)
        self.assertIn('average_drawdown', drawdown_metrics)
        self.assertIn('drawdown_std', drawdown_metrics)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
