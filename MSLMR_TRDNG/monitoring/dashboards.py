import os
import json
import asyncio
from typing import Dict, List, Optional

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

from core.logging_system import LoggingMixin
from strategies.strategy_registry import StrategyRegistry
from data.models import Alert
from core.database_interactor import DatabaseInteractor
from core.config_manager import ConfigManager

class TradingDashboard(LoggingMixin):
    """
    Dashboard interactivo para monitoreo de sistema de trading
    """

    def __init__(
        self,
        db_interactor: DatabaseInteractor,
        strategy_registry: StrategyRegistry,
        port: int = 8050,
    ):
        """
        Inicializa el dashboard de trading

        :param db_interactor:  Instancia de DatabaseInteractor para interactuar con la base de datos
        :param strategy_registry: Instancia de StrategyRegistry para acceder a las estrategias
        :param port: Puerto para servir el dashboard
        """
        self.db_interactor = db_interactor
        self.strategy_registry = strategy_registry
        self.port = port
        self.app = dash.Dash(__name__)
        self._configure_layout()
        self._register_callbacks()

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
            ]),

            # Feed de Alertas
            html.Div([
                html.H3('Alertas en Tiempo Real'),
                html.Ul(id='alerts-feed')
            ]),

            # Componente de intervalo para actualizaciones
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ])

    def _register_callbacks(self):
      #Rest of the implementation
