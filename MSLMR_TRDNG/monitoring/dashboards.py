import os
import json
import asyncio
from typing import Dict, List, Optional

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

from core.logging_system import LoggingMixin
from strategies.strategy_registry import StrategyRegistry

class TradingDashboard(LoggingMixin):
    """
    Dashboard interactivo para monitoreo de sistema de trading
    """
    
    def __init__(
        self, 
        data_source: Optional[str] = None,
        port: int = 8050
    ):
        """
        Inicializa el dashboard de trading
        
        :param data_source: Ruta del archivo de datos
        :param port: Puerto para servir el dashboard
        """
        self.data_source = data_source or self._generate_sample_data()
        self.port = port
        self.app = dash.Dash(__name__)
        self._configure_layout()
        self._register_callbacks()
    
    def _generate_sample_data(self) -> str:
        """
        Genera datos de ejemplo si no se proporciona fuente
        
        :return: Ruta del archivo de datos generado
        """
        sample_data = {
            'strategies': [
                {
                    'name': 'RSI MACD Strategy',
                    'trades': [
                        {'timestamp': '2023-01-01', 'profit_loss': 100, 'symbol': 'BTC/USDT'},
                        {'timestamp': '2023-01-02', 'profit_loss': -50, 'symbol': 'BTC/USDT'}
                    ],
                    'metrics': {
                        'total_trades': 50,
                        'win_rate': 0.6,
                        'total_profit': 5000
                    }
                },
                {
                    'name': 'Moving Average Strategy',
                    'trades': [
                        {'timestamp': '2023-01-01', 'profit_loss': 75, 'symbol': 'ETH/USDT'},
                        {'timestamp': '2023-01-02', 'profit_loss': -25, 'symbol': 'ETH/USDT'}
                    ],
                    'metrics': {
                        'total_trades': 40,
                        'win_rate': 0.55,
                        'total_profit': 4000
                    }
                }
            ]
        }
        
        # Guardar datos en archivo temporal
        filepath = './sample_trading_data.json'
        with open(filepath, 'w') as f:
            json.dump(sample_data, f)
        
        return filepath
    
    def _load_data(self) -> Dict:
        """
        Carga datos desde la fuente
        
        :return: Datos de trading
        """
        try:
            with open(self.data_source, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            return {'strategies': []}
    
    def _configure_layout(self):
        """
        Configura el diseño del dashboard
        """
        self.app.layout = html.Div([
            html.H1('Trading System Dashboard'),
            
            # Sección de métricas generales
            html.Div([
                html.Div([
                    html.H3('Estrategias'),
                    dcc.Dropdown(
                        id='strategy-selector',
                        options=[],
                        value=None,
                        placeholder='Seleccionar Estrategia'
                    )
                ], className='col-md-4'),
                
                html.Div([
                    html.H3('Métricas Generales'),
                    html.Div(id='general-metrics')
                ], className='col-md-8')
            ], className='row'),
            
            # Sección de gráficos
            html.Div([
                html.Div([
                    html.H3('Rendimiento por Estrategia'),
                    dcc.Graph(id='performance-chart')
                ], className='col-md-6'),
                
                html.Div([
                    html.H3('Distribución de Trades'),
                    dcc.Graph(id='trades-distribution')
                ], className='col-md-6')
            ], className='row'),
            
            # Tabla de trades
            html.Div([
                html.H3('Historial de Trades'),
                html.Div(id='trades-table')
            ])
        ])
    
    def _register_callbacks(self):
        """
        Registra callbacks para interactividad del dashboard
        """
        @self.app.callback(
            [Output('strategy-selector', 'options'),
             Output('general-metrics', 'children'),
             Output('performance-chart', 'figure'),
             Output('trades-distribution', 'figure'),
             Output('trades-table', 'children')],
            [Input('strategy-selector', 'value')]
        )
        def update_dashboard(selected_strategy):
            data = self._load_data()
            strategies = data.get('strategies', [])
            
            # Opciones de estrategias
            strategy_options = [
                {'label': s['name'], 'value': s['name']} 
                for s in strategies
            ]
            
            # Seleccionar estrategia
            strategy = next(
                (s for s in strategies if s['name'] == selected_strategy), 
                strategies[0] if strategies else None
            )
            
            if not strategy:
                return [], [], go.Figure(), go.Figure(), []
            
            # Métricas generales
            metrics_display = [
                html.Div(f"Total Trades: {strategy['metrics']['total_trades']}"),
                html.Div(f"Win Rate: {strategy['metrics']['win_rate']:.2%}"),
                html.Div(f"Total Profit: ${strategy['metrics']['total_profit']}")
            ]
            
            # Gráfico de rendimiento
            trades_df = pd.DataFrame(strategy['trades'])
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            performance_fig = go.Figure()
            performance_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'], 
                y=trades_df['profit_loss'].cumsum(),
                mode='lines+markers',
                name='Cumulative Profit'
            ))
            
            # Distribución de trades
            trades_dist_fig = px.histogram(
                trades_df, 
                x='profit_loss', 
                title='Distribución de Ganancias/Pérdidas'
            )
            
            # Tabla de trades
            trades_table = html.Table([
                html.Thead(html.Tr([
                    html.Th(col) for col in trades_df.columns
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(trades_df.iloc[i][col]) for col in trades_df.columns
                    ]) for i in range(len(trades_df))
                ])
            ])
            
            return (
                strategy_options, 
                metrics_display, 
                performance_fig, 
                trades_dist_fig, 
                trades_table
            )
    
    def run(self, debug: bool = False):
        """
        Inicia el servidor del dashboard
        
        :param debug: Modo de depuración
        """
        try:
            self.logger.info(f"Iniciando dashboard en puerto {self.port}")
            self.app.run_server(
                debug=debug, 
                port=self.port, 
                host='0.0.0.0'
            )
        except Exception as e:
            self.logger.error(f"Error iniciando dashboard: {e}")

# Ejecutar dashboard desde línea de comandos
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading System Dashboard')
    parser.add_argument(
        '--data', 
        type=str, 
        help='Ruta del archivo de datos de trading'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8050, 
        help='Puerto para servir el dashboard'
    )
    
    args = parser.parse_args()
    
    dashboard = TradingDashboard(
        data_source=args.data, 
        port=args.port
    )
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()
