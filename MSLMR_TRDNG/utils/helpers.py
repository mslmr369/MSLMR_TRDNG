import os
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from typing import List, Dict, Optional, Any

from core.logging_system import LoggingMixin

class StrategyReporter(LoggingMixin):
    """
    Sistema avanzado de generación de reportes de estrategias de trading
    """
    
    def __init__(
        self, 
        strategies_data: List[Dict[str, Any]], 
        output_dir: str = './reports'
    ):
        """
        Inicializa el sistema de reporting
        
        :param strategies_data: Datos de rendimiento de estrategias
        :param output_dir: Directorio para guardar reportes
        """
        self.strategies_data = strategies_data
        self.output_dir = output_dir
        
        # Crear directorio de reportes si no existe
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comparative_report(self) -> Dict[str, Any]:
        """
        Genera un informe comparativo de estrategias
        
        :return: Diccionario con métricas comparativas
        """
        comparison_metrics = {}
        
        for strategy in self.strategies_data:
            name = strategy.get('name', 'Estrategia Sin Nombre')
            metrics = strategy.get('metrics', {})
            
            comparison_metrics[name] = {
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_profit': metrics.get('total_profit', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            }
        
        return comparison_metrics
    
    def create_performance_dashboard(self) -> Optional[str]:
        """
        Crea un dashboard interactivo de rendimiento
        
        :return: Ruta del archivo HTML generado
        """
        try:
            # Convertir datos a DataFrame
            df = pd.DataFrame(
                [strategy.get('metrics', {}) for strategy in self.strategies_data]
            )
            df['strategy_name'] = [
                strategy.get('name', 'Estrategia Sin Nombre') 
                for strategy in self.strategies_data
            ]
            
            # Crear figuras Plotly
            figs = {
                'win_rate': px.bar(
                    df, 
                    x='strategy_name', 
                    y='win_rate', 
                    title='Win Rate por Estrategia'
                ),
                'total_profit': px.bar(
                    df, 
                    x='strategy_name', 
                    y='total_profit', 
                    title='Ganancia Total por Estrategia'
                ),
                'sharpe_ratio': px.bar(
                    df, 
                    x='strategy_name', 
                    y='sharpe_ratio', 
                    title='Ratio de Sharpe por Estrategia'
                )
            }
            
            # Crear dashboard HTML
            dashboard_path = os.path.join(self.output_dir, 'performance_dashboard.html')
            
            with open(dashboard_path, 'w') as f:
                f.write('<html><head><title>Performance Dashboard</title>')
                f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
                f.write('</head><body>')
                
                for name, fig in figs.items():
                    f.write(f'<div id="{name}"></div>')
                    f.write(fig.to_html(full_html=False, include_plotlyjs=False))
                
                f.write('</body></html>')
            
            return dashboard_path
        
        except Exception as e:
            self.logger.error(f"Error creando dashboard: {e}")
            return None
    
    def generate_detailed_strategy_report(
        self, 
        strategy_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Genera un informe detallado para una estrategia específica
        
        :param strategy_name: Nombre de la estrategia
        :return: Diccionario con informe detallado
        """
        strategy = next(
            (s for s in self.strategies_data if s.get('name') == strategy_name), 
            None
        )
        
        if not strategy:
            self.logger.warning(f"Estrategia no encontrada: {strategy_name}")
            return None
        
        # Generar informe completo
        detailed_report = {
            'strategy_name': strategy_name,
            'metrics': strategy.get('metrics', {}),
            'trade_details': strategy.get('trades', [])
        }
        
        # Guardar como JSON
        report_path = os.path.join(
            self.output_dir, 
            f'{strategy_name.replace("/", "_")}_report.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        return detailed_report
    
    def export_trade_log(
        self, 
        strategy_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Exporta log de trades para una o todas las estrategias
        
        :param strategy_name: Nombre de estrategia específica
        :return: Ruta del archivo CSV generado
        """
        try:
            # Filtrar trades por estrategia si se especifica
            if strategy_name:
                trades = next(
                    (s.get('trades', []) for s in self.strategies_data 
                     if s.get('name') == strategy_name), 
                    []
                )
            else:
                # Combinar trades de todas las estrategias
                trades = [
                    trade for strategy in self.strategies_data 
                    for trade in strategy.get('trades', [])
                ]
            
            # Convertir a DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Añadir columna de estrategia si no existe
            if 'strategy_name' not in trades_df.columns:
                trades_df['strategy_name'] = strategy_name or 'Multiple'
            
            # Exportar a CSV
            filename = (
                f'{strategy_name.replace("/", "_") if strategy_name else "all_strategies"}_trades.csv'
            )
            export_path = os.path.join(self.output_dir, filename)
            
            trades_df.to_csv(export_path, index=False)
            return export_path
        
        except Exception as e:
            self.logger.error(f"Error exportando log de trades: {e}")
            return None
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Genera un resumen ejecutivo de rendimiento
        
        :return: Diccionario con resumen de rendimiento
        """
        summary = {
            'total_strategies': len(self.strategies_data),
            'overall_performance': {
                'total_trades': sum(
                    strategy.get('metrics', {}).get('total_trades', 0) 
                    for strategy in self.strategies_data
                ),
                'total_profit': sum(
                    strategy.get('metrics', {}).get('total_profit', 0) 
                    for strategy in self.strategies_data
                ),
                'average_win_rate': sum(
                    strategy.get('metrics', {}).get('win_rate', 0) 
                    for strategy in self.strategies_data
                ) / max(len(self.strategies_data), 1)
            },
            'best_performing_strategy': max(
                self.strategies_data, 
                key=lambda s: s.get('metrics', {}).get('total_profit', 0)
            ).get('name', 'N/A')
        }
        
        # Guardar resumen
        summary_path = os.path.join(self.output_dir, 'performance_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

# Ejemplo de uso
def main():
    # Datos de ejemplo de estrategias
    strategies_data = [
        {
            'name': 'RSI MACD Strategy',
            'metrics': {
                'total_trades': 50,
                'win_rate': 0.6,
                'total_profit': 5000,
                'sharpe_ratio': 1.5,
                'max_drawdown': -10
            },
            'trades': [
                {'symbol': 'BTC/USDT', 'profit_loss': 100},
                {'symbol': 'BTC/USDT', 'profit_loss': -50}
            ]
        },
        {
            'name': 'Moving Average Strategy',
            'metrics': {
                'total_trades': 40,
                'win_rate': 0.55,
                'total_profit': 4000,
                'sharpe_ratio': 1.2,
                'max_drawdown': -15
            },
            'trades': [
                {'symbol': 'ETH/USDT', 'profit_loss': 75},
                {'symbol': 'ETH/USDT', 'profit_loss': -25}
            ]
        }
    ]
    
    # Inicializar reporteador
    reporter = StrategyReporter(strategies_data)
    
    # Generar diferentes tipos de reportes
    comparative_report = reporter.generate_comparative_report()
    dashboard_path = reporter.create_performance_dashboard()
    detailed_report = reporter.generate_detailed_strategy_report('RSI MACD Strategy')
    trade_log_path = reporter.export_trade_log()
    summary_report = reporter.generate_summary_report()
    
    # Imprimir resultados
    print("Comparative Report:", comparative_report)
    print("Dashboard Path:", dashboard_path)
    print("Trade Log Path:", trade_log_path)
    print("Summary Report:", summary_report)

if __name__ == "__main__":
    main()
