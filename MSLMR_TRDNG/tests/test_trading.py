import unittest
import pandas as pd
import numpy as np
from typing import List, Dict
import asyncio

# Importar componentes de trading a testear
from trading.execution import TradingExecutor
from trading.backtesting import BacktestEngine, BacktestConfiguration
from trading.live_trading import LiveTradingSystem

# Estrategias para pruebas
from models.traditional.rsi_macd import RSIMACDStrategy
from models.traditional.moving_average import MovingAverageStrategy

class TestTradingComponents(unittest.TestCase):
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
        
        # Configuración de pruebas
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.timeframes = ['1h', '4h']
    
    def test_trading_executor_initialization(self):
        """
        Prueba la inicialización del ejecutor de trading
        """
        executor = TradingExecutor(
            exchange_id='binance',
            symbols=self.symbols,
            timeframes=self.timeframes,
            initial_capital=10000.0,
            dry_run=True
        )
        
        # Verificaciones
        self.assertEqual(executor.symbols, self.symbols)
        self.assertEqual(executor.timeframes, self.timeframes)
        self.assertTrue(executor.dry_run)
    
    def test_trading_executor_market_data_fetch(self):
        """
        Prueba la obtención de datos de mercado
        """
        executor = TradingExecutor(
            exchange_id='binance',
            symbols=self.symbols,
            timeframes=self.timeframes,
            dry_run=True
        )
        
        # Intentar obtener datos para un símbolo y timeframe
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    market_data = executor.fetch_live_market_data(
                        symbol, timeframe, limit=100
                    )
                    
                    # Verificaciones
                    self.assertIsInstance(market_data, pd.DataFrame)
                    self.assertGreater(len(market_data), 0)
                    
                    # Columnas esperadas
                    expected_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in expected_columns:
                        self.assertIn(col, market_data.columns)
                
                except Exception as e:
                    self.fail(f"Error obteniendo datos para {symbol} en {timeframe}: {e}")
    
    def test_backtesting_engine(self):
        """
        Prueba el motor de backtesting
        """
        # Configuración de backtesting
        config = BacktestConfiguration(
            initial_capital=10000.0,
            start_date=pd.Timestamp('2022-01-01'),
            end_date=pd.Timestamp('2022-12-31'),
            symbols=self.symbols,
            timeframes=self.timeframes
        )
        
        # Inicializar motor de backtesting
        backtest_engine = BacktestEngine(config)
        
        # Estrategias a testear
        strategies = [
            RSIMACDStrategy(),
            MovingAverageStrategy()
        ]
        
        # Ejecutar backtesting
        results = backtest_engine.run_multiple_strategies(strategies)
        
        # Verificaciones
        for strategy_name, metrics in results.items():
            self.assertIn('total_trades', metrics)
            self.assertIn('win_rate', metrics)
            self.assertIn('total_profit', metrics)
            
            # Verificar métricas
            self.assertGreaterEqual(metrics['total_trades'], 0)
            self.assertGreaterEqual(metrics['win_rate'], 0)
            self.assertIsInstance(metrics['total_profit'], (int, float))
    
    async def test_live_trading_system_initialization(self):
        """
        Prueba la inicialización del sistema de trading en vivo
        """
        trading_system = LiveTradingSystem(
            exchange_id='binance',
            symbols=self.symbols,
            timeframes=self.timeframes,
            initial_capital=10000.0,
            dry_run=True
        )
        
        # Verificaciones
        self.assertEqual(trading_system.symbols, self.symbols)
        self.assertEqual(trading_system.timeframes, self.timeframes)
        self.assertTrue(trading_system.dry_run)
    
    def test_trade_simulation(self):
        """
        Prueba la simulación de trades
        """
        executor = TradingExecutor(
[O            exchange_id='binance',
            symbols=self.symbols,
            timeframes=self.timeframes,
            dry_run=True
        )
        
        # Simular trade
        for symbol in self.symbols:
            trade_result = executor.execute_live_trade(
                symbol=symbol, 
                side='buy', 
                amount=0.001
            )
            
            # Verificaciones
            self.assertIsNotNone(trade_result)
            self.assertEqual(trade_result['status'], 'simulated')
            self.assertEqual(trade_result['symbol'], symbol)
            self.assertEqual(trade_result['side'], 'buy')
    
    def test_trade_execution_constraints(self):
        """
        Prueba restricciones de ejecución de trades
        """
        executor = TradingExecutor(
            exchange_id='binance',
            symbols=self.symbols,
            timeframes=self.timeframes,
            initial_capital=10000.0,
            dry_run=True
        )
        
        # Verificar cálculo de tamaño de posición
        current_price = 50000
        stop_loss_price = 49000
        
        position_size = executor.portfolio_manager.calculate_position_size(
            current_price, 
            stop_loss_price
        )
        
        # Verificaciones
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, executor.portfolio_manager.initial_capital)
        
        # Verificar validación de riesgo
        is_risk_valid = executor.risk_manager.validate_trade_risk(
            current_price, 
            stop_loss_price, 
            position_size, 
            executor.portfolio_manager.initial_capital
        )
        
        self.assertTrue(is_risk_valid)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
