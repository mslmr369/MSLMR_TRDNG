import os
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Importaciones de componentes de backtesting
from trading.backtesting import BacktestEngine, BacktestConfiguration
from strategies.strategy_registry import StrategyRegistry

# Importaciones de estrategias
from models.traditional.rsi_macd import RSIMACDStrategy
from models.traditional.moving_average import MovingAverageStrategy

# Importaciones de análisis
from analytics.performance import PerformanceAnalyzer
from analytics.reporting import StrategyReporter

class BacktestRunner:
    """
    Ejecutor de backtesting para múltiples estrategias
    """
    
    def __init__(
        self, 
        output_dir: str = './backtest_results',
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ):
        """
        Inicializa el ejecutor de backtests
        
        :param output_dir: Directorio para guardar resultados
        :param symbols: Símbolos a testear
        :param timeframes: Timeframes a testear
        """
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Configurar símbolos y timeframes
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.timeframes = timeframes or ['1h', '4h', '1d']
        
        # Registro de estrategias
        self.strategy_registry = StrategyRegistry()
        self._register_strategies()
    
    def _register_strategies(self):
        """
        Registra estrategias disponibles para backtesting
        """
        self.strategy_registry.register_strategy('RSI_MACD', RSIMACDStrategy)
        self.strategy_registry.register_strategy('Moving_Average', MovingAverageStrategy)
    
    def run_backtest(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0
    ) -> Dict[str, Dict]:
        """
        Ejecuta backtesting para todas las estrategias registradas
        
        :param start_date: Fecha de inicio del backtesting
        :param end_date: Fecha de fin del backtesting
        :param initial_capital: Capital inicial
        :return: Resultados de backtesting
        """
        # Configurar fechas si no se proporcionan
        start_date = start_date or datetime.now() - timedelta(days=365)
        end_date = end_date or datetime.now()
        
        # Configuración de backtesting
        config = BacktestConfiguration(
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            symbols=self.symbols,
            timeframes=self.timeframes
        )
        
        # Inicializar motor de backtesting
        backtest_engine = BacktestEngine(config)
        
        # Obtener estrategias registradas
        strategies = [
            self.strategy_registry.create_strategy(name) 
            for name in self.strategy_registry.list_strategies()
        ]
        
        # Ejecutar backtesting
        results = backtest_engine.run_multiple_strategies(strategies)
        
        return results
    
    def generate_reports(
        self, 
        backtest_results: Dict[str, Dict],
        output_format: str = 'json'
    ):
        """
        Genera informes de rendimiento
        
        :param backtest_results: Resultados de backtesting
        :param output_format: Formato de salida (json, html)
        """
        # Analizar rendimiento
        performance_analyzer = PerformanceAnalyzer(
            [trade for strategy in backtest_results.values() for trade in strategy.get('trades', [])]
        )
        
        # Generar informe de rendimiento
        performance_report = performance_analyzer.generate_detailed_report(
            output_path=os.path.join(self.output_dir, 'performance_reports')
        )
        
        # Reporteador de estrategias
        strategy_reporter = StrategyReporter(
            [{'name': name, 'metrics': metrics} for name, metrics in backtest_results.items()]
        )
        
        # Generar dashboard de rendimiento
        dashboard_path = strategy_reporter.create_performance_dashboard()
        
        # Guardar resultados
        if output_format == 'json':
            # Guardar resultados como JSON
            results_path = os.path.join(self.output_dir, 'backtest_results.json')
            with open(results_path, 'w') as f:
                json.dump(backtest_results, f, indent=2)
            
            performance_path = os.path.join(self.output_dir, 'performance_report.json')
            with open(performance_path, 'w') as f:
                json.dump(performance_report, f, indent=2)
        
        return {
            'results_path': results_path,
            'performance_path': performance_path,
            'dashboard_path': dashboard_path
        }
    
    def compare_strategies(
        self, 
        backtest_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Compara rendimiento de estrategias
        
        :param backtest_results: Resultados de backtesting
        :return: Comparación de estrategias
        """
        comparison = {}
        
        for strategy_name, metrics in backtest_results.items():
            comparison[strategy_name] = {
                'total_profit': metrics.get('total_profit', 0),
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            }
        
        # Encontrar mejor estrategia
        best_strategy = max(
            comparison.items(), 
            key=lambda x: x[1]['total_profit']
        )[0]
        
        comparison['best_strategy'] = best_strategy
        
        return comparison

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutor de Backtesting')
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        help='Fecha de inicio del backtesting (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        help='Fecha de fin del backtesting (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--initial-capital', 
        type=float, 
        default=10000.0, 
        help='Capital inicial para backtesting'
    )
    parser.add_argument(
        '--output-format', 
        type=str, 
        choices=['json', 'html'], 
        default='json', 
        help='Formato de salida de los resultados'
    )
    
    args = parser.parse_args()
    
    # Parsear fechas si se proporcionan
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # Inicializar ejecutor de backtests
    backtest_runner = BacktestRunner()
    
    # Ejecutar backtesting
    backtest_results = backtest_runner.run_backtest(
        start_date=start_date, 
        end_date=end_date,
        initial_capital=args.initial_capital
    )
    
    # Generar reportes
    reports = backtest_runner.generate_reports(
        backtest_results, 
        output_format=args.output_format
    )
    
    # Comparar estrategias
    strategy_comparison = backtest_runner.compare_strategies(backtest_results)
    
    # Imprimir resultados
    print("Mejores estrategias:")
    print(json.dumps(strategy_comparison, indent=2))
    print("\nReportes generados:")
    print(json.dumps(reports, indent=2))

if __name__ == '__main__':
    main()
