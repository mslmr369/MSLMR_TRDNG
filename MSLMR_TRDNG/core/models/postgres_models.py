# core/models/postgres_models.py
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.orm import relationship
from core.database import Base

class Price(Base):
    __tablename__ = 'prices'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'timeframe', 'timestamp', name='_symbol_timeframe_timestamp_uc'),)

class Indicator(Base):
    __tablename__ = 'indicators'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    rsi = Column(Float)
    macd_line = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    ichimoku_conv = Column(Float)
    ichimoku_base = Column(Float)
    ichimoku_span_a = Column(Float)
    ichimoku_span_b = Column(Float)
    ichimoku_chikou = Column(Float)
    adx = Column(Float)
    obv = Column(Float)
    dema = Column(Float)
    stoch_rsi_k = Column(Float)
    stoch_rsi_d = Column(Float)
    vwap = Column(Float)
    pivot_high = Column(Float)
    pivot_low = Column(Float)
    fractal_up = Column(Float)
    fractal_down = Column(Float)
    correlation = Column(Float)
    divergence_bull = Column(Boolean)
    divergence_bear = Column(Boolean)
    __table_args__ = (UniqueConstraint('symbol', 'timeframe', 'timestamp', name='_symbol_timeframe_timestamp_uc'),)

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    signal_type = Column(String, nullable=False)
    strategy_name = Column(String)
    __table_args__ = (UniqueConstraint('symbol', 'timeframe', 'timestamp', 'signal_type', name='_symbol_timeframe_timestamp_signal_uc'),)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    side = Column(String, nullable=False)
    price = Column(Float)
    size = Column(Float)
    strategy_name = Column(String)
    signal_id = Column(Integer, ForeignKey('signals.id'))
    profit_loss = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'timestamp', 'side', name='_symbol_timestamp_side_uc'),)

class Backtest(Base):
    __tablename__ = 'backtests'
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, nullable=False)
    start_date = Column(TIMESTAMP, nullable=False)
    end_date = Column(TIMESTAMP, nullable=False)
    total_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    trades = Column(Integer)
    profitable_trades = Column(Integer)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, nullable=False)
    message = Column(String, nullable=False)
