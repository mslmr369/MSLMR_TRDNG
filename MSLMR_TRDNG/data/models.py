import os
import json
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, List, Optional
import logging
from datetime import datetime
from core.database import Base

class MarketDataModel(Base):
    """
    Modelo de almacenamiento de datos de mercado
    """
    __tablename__ = 'market_data'

    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    timeframe = sa.Column(sa.String(10), nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    open = sa.Column(sa.Float, nullable=False)
    high = sa.Column(sa.Float, nullable=False)
    low = sa.Column(sa.Float, nullable=False)
    close = sa.Column(sa.Float, nullable=False)
    volume = sa.Column(sa.Float, nullable=False)

    # Índices para mejorar rendimiento
    __table_args__ = (
        sa.Index('idx_symbol_timeframe', 'symbol', 'timeframe'),
        sa.Index('idx_timestamp', 'timestamp')
    )

class IndicatorDataModel(Base):
    """
    Modelo de almacenamiento de indicadores
    """
    __tablename__ = 'indicators'

    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False)
    timeframe = sa.Column(sa.String(10), nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False)

    # Indicadores genéricos
    rsi = sa.Column(sa.Float)
    macd_line = sa.Column(sa.Float)
    macd_signal = sa.Column(sa.Float)
    macd_histogram = sa.Column(sa.Float)

    # Índices
    __table_args__ = (
        sa.Index('idx_symbol_timeframe_ind', 'symbol', 'timeframe'),
        sa.Index('idx_timestamp_ind', 'timestamp')
    )
