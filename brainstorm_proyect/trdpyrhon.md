
Entiendo que deseas obtener toda la documentación en un formato descargable. A continuación, te proporcionaré la documentación completa en formato Markdown. Puedes copiar este contenido y guardarlo en un archivo con extensión .md. Luego, podrás convertirlo a PDF, Word u otros formatos utilizando herramientas como Pandoc, Typora, Markdown Editor o cualquier editor de texto compatible con Markdown.


---

Trading Strategy Automation with Python and PostgreSQL

Table of Contents

1. Introduction


2. System Overview


3. Requirements


4. Environment Setup


5. PostgreSQL Configuration


6. Project Structure


7. Data Ingestion Module


8. Indicator Calculation Module


9. Strategy Implementation Module


10. Backtesting Module


11. Live Execution Module


12. Dashboard Module


13. Risk Management Module


14. Advanced Alerts Module


15. Automation and Deployment


16. Security and Robustness


17. Monitoring and Logging


18. Additional Recommendations


19. Appendix: Complete Code Listings




---

1. Introduction

This document provides a comprehensive guide to designing, implementing, and deploying an automated trading system using Python and PostgreSQL. The system includes modules for data ingestion, indicator calculation, strategy generation, backtesting, live trading execution, dashboard visualization, risk management, and advanced alerts. The aim is to create a modular, scalable, and efficient trading platform capable of handling real-time data and executing strategies with robust risk controls and monitoring capabilities.


---

2. System Overview

The automated trading system is divided into several interconnected modules:

1. Data Ingestion: Fetches market data from external APIs and stores it in a PostgreSQL database.


2. Indicator Calculation: Computes technical indicators based on the ingested data.


3. Strategy Implementation: Generates trading signals based on indicator values.


4. Backtesting: Simulates the trading strategy on historical data to evaluate performance.


5. Live Execution: Executes trades in real-time based on generated signals.


6. Dashboard: Visualizes trading activities, signals, trades, and performance metrics.


7. Risk Management: Manages position sizing, stop-loss, take-profit, and exposure limits.


8. Advanced Alerts: Sends notifications via Telegram and Email upon trade executions and significant events.


9. Automation and Deployment: Ensures the system runs continuously using Docker and cron jobs.


10. Security and Robustness: Implements best practices to secure credentials and ensure system stability.


11. Monitoring and Logging: Tracks system activities and logs events for auditing and debugging.




---

3. Requirements

3.1. Software Requirements

Operating System: Linux (Ubuntu 20.04 or higher recommended) or Windows

Python: Version 3.8 or higher

PostgreSQL: Version 12 or higher

Docker & Docker Compose: For containerization and deployment

APIs:

CCXT: For cryptocurrency exchange data and trading

Telegram Bot API: For sending alerts

SMTP Server: For sending email notifications



3.2. Python Libraries

pandas

numpy

ta (Technical Analysis Library)

sqlalchemy

psycopg2-binary

ccxt

Flask

plotly

Flask-Login

requests


3.3. Hardware Requirements

Server/VPS: Sufficient CPU and memory to handle data processing and real-time trading

Storage: Adequate disk space for PostgreSQL database and logs



---

4. Environment Setup

4.1. Install Python

On Ubuntu:

sudo apt update
sudo apt install python3 python3-pip python3-venv -y

On Windows:

Descarga e instala Python desde python.org. Asegúrate de marcar la opción "Add Python to PATH" durante la instalación.

4.2. Create a Project Directory

mkdir trading_strategy
cd trading_strategy

4.3. Create and Activate a Virtual Environment

python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

4.4. Install Python Dependencies

Crea un archivo requirements.txt con el siguiente contenido:

pandas
numpy
ta
sqlalchemy
psycopg2-binary
ccxt
Flask
plotly
Flask-Login
requests

Instala las dependencias:

pip install -r requirements.txt


---

5. PostgreSQL Configuration

5.1. Install PostgreSQL

On Ubuntu:

sudo apt update
sudo apt install postgresql postgresql-contrib -y

On Windows:

Descarga e instala PostgreSQL desde postgresql.org.

5.2. Initial Configuration

1. Switch to PostgreSQL User and Create a Database User:

sudo -i -u postgres


2. Open PostgreSQL Shell:

psql


3. Create a New Role and Database:

-- Create role
CREATE ROLE trading_user WITH LOGIN PASSWORD 'secure_password';

-- Create database
CREATE DATABASE trading_db OWNER trading_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Exit psql
\q


4. Exit PostgreSQL User:

exit



5.3. Create Database Tables

Conéctate a la base de datos usando el usuario creado y ejecuta los comandos SQL para crear las tablas necesarias.

psql -U trading_user -d trading_db -h localhost

Ejecuta los siguientes scripts SQL:

-- Table: prices
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_prices_symbol_timeframe_timestamp ON prices (symbol, timeframe, timestamp);

-- Table: indicators
CREATE TABLE indicators (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    rsi DOUBLE PRECISION,
    macd_line DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    ichimoku_conv DOUBLE PRECISION,
    ichimoku_base DOUBLE PRECISION,
    ichimoku_span_a DOUBLE PRECISION,
    ichimoku_span_b DOUBLE PRECISION,
    ichimoku_chikou DOUBLE PRECISION,
    adx DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    dema DOUBLE PRECISION,
    stoch_rsi_k DOUBLE PRECISION,
    stoch_rsi_d DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    pivot_high DOUBLE PRECISION,
    pivot_low DOUBLE PRECISION,
    fractal_up DOUBLE PRECISION,
    fractal_down DOUBLE PRECISION,
    correlation DOUBLE PRECISION,
    divergence_bull BOOLEAN,
    divergence_bear BOOLEAN,
    PRIMARY KEY(symbol, timeframe, timestamp)
);

CREATE INDEX idx_indicators_symbol_timeframe_timestamp ON indicators (symbol, timeframe, timestamp);

-- Table: signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type TEXT NOT NULL, -- 'long', 'short', 'close_long', 'close_short'
    strategy_name TEXT,
    UNIQUE(symbol, timeframe, timestamp, signal_type)
);

CREATE INDEX idx_signals_symbol_timeframe_timestamp ON signals (symbol, timeframe, timestamp);

-- Table: trades
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side TEXT NOT NULL, -- 'buy' o 'sell'
    price DOUBLE PRECISION,
    size DOUBLE PRECISION,
    strategy_name TEXT,
    signal_id INTEGER REFERENCES signals(id),
    profit_loss DOUBLE PRECISION,
    UNIQUE(symbol, timestamp, side)
);

CREATE INDEX idx_trades_symbol_timestamp ON trades (symbol, timestamp);

-- Table: backtests
CREATE TABLE backtests (
    id SERIAL PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    total_return DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    trades INTEGER,
    profitable_trades INTEGER
);

Salir de psql:

\q


---

6. Project Structure

Organiza el proyecto con una estructura clara y modular para mejorar la mantenibilidad y escalabilidad.

trading_strategy/
│
├── config.py
├── requirements.txt
├── run_backtest.py
├── run_live.py
├── fetch_data.py
├── calculate_indicators.py
├── generate_signals.py
├── dashboard.py
│
├── db/
│   ├── __init__.py
│   ├── models.py
│   └── db_connection.py
│
├── strategies/
│   ├── __init__.py
│   └── advanced_strategy.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── helpers.py
│   ├── risk_management.py
│   └── alerts.py
│
├── templates/
│   ├── index.html
│   └── login.html
│
└── logs/
    ├── live.log
    ├── backtest.log
    ├── dashboard.log
    └── alerts.log

6.1. Description of Files and Directories

config.py: Archivo de configuración que contiene ajustes y credenciales.

requirements.txt: Lista de dependencias de Python.

fetch_data.py: Script para la ingesta de datos del mercado.

calculate_indicators.py: Script para calcular indicadores técnicos.

generate_signals.py: Script para generar señales de trading basadas en indicadores.

run_backtest.py: Script para realizar backtesting de estrategias.

run_live.py: Script para ejecutar trades en tiempo real basado en señales.

dashboard.py: Aplicación Flask para visualizar datos y actividades de trading.

db/: Contiene la conexión a la base de datos y los modelos ORM.

strategies/: Contiene las implementaciones de estrategias de trading.

utils/: Módulos de utilidad para logging, gestión de riesgos, alertas, etc.

templates/: Plantillas HTML para el dashboard de Flask.

logs/: Directorio para almacenar archivos de log.



---

7. Data Ingestion Module

Este módulo obtiene datos del mercado desde APIs externas (por ejemplo, exchanges de criptomonedas a través de CCXT) y los almacena en la base de datos PostgreSQL.

7.1. Configuration (config.py)

Añade configuraciones para la ingesta de datos.

# config.py

import os

# Database Configuration
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'trading_user'),
    'password': os.getenv('DB_PASSWORD', 'secure_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'trading_db')
}

# Exchange Configuration
EXCHANGE_ID = 'binance'
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Lista de símbolos a rastrear
TIMEFRAMES = ['1m', '5m', '1h', '1d']  # Timeframes a almacenar

# Data Ingestion Parameters
FETCH_LIMIT = 1000  # Número de velas por solicitud a la API

# Logging Directory
LOG_DIR = 'logs'

# Risk Management Parameters
RISK_PER_TRADE = 0.01  # 1% del capital por trade
STOP_LOSS_PERCENT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT = 0.04  # 4% take profit
MAX_GLOBAL_EXPOSURE = 0.20  # Máximo 20% del capital expuesto

# Alert Configuration
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN', 'your_telegram_api_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_telegram_chat_id')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', 'your_email@gmail.com')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', 'your_email_password')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER', 'receiver_email@gmail.com')

7.2. Database Connection (db/db_connection.py)

Configura la conexión a PostgreSQL usando SQLAlchemy.

# db/db_connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DB_CONFIG

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

7.3. ORM Models (db/models.py)

Define los modelos ORM que corresponden a las tablas de la base de datos.

# db/models.py

from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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
    signal_type = Column(String, nullable=False)  # 'long', 'short', etc.
    strategy_name = Column(String)
    __table_args__ = (UniqueConstraint('symbol', 'timeframe', 'timestamp', 'signal_type', name='_symbol_timeframe_timestamp_signal_uc'),)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    side = Column(String, nullable=False)  # 'buy' o 'sell'
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

7.4. Data Ingestion Script (fetch_data.py)

Script para obtener datos OHLCV del exchange y almacenarlos en la base de datos.

# fetch_data.py

import ccxt
import pandas as pd
from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from db.models import Price
from config import EXCHANGE_ID, SYMBOLS, TIMEFRAMES, FETCH_LIMIT
from utils.logger import setup_logger
import time

logger = setup_logger('fetch_data')

def fetch_ohlcv(exchange, symbol, timeframe, since=None):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=FETCH_LIMIT)
        return ohlcv
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}")
        return []

def store_prices(session: Session, symbol: str, timeframe: str, data: list):
    for candle in data:
        timestamp = pd.to_datetime(candle[0], unit='ms')
        price = Price(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=candle[1],
            high=candle[2],
            low=candle[3],
            close=candle[4],
            volume=candle[5]
        )
        try:
            session.merge(price)  # Merge to avoid duplicates
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
    session.commit()

def main():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class()
    exchange.load_markets()
    session = SessionLocal()
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Fetching data for {symbol} on {timeframe}")
            # Get the latest timestamp in the database
            last_price = session.query(Price).filter_by(symbol=symbol, timeframe=timeframe).order_by(Price.timestamp.desc()).first()
            since = None
            if last_price:
                since = int(last_price.timestamp.timestamp() * 1000) + 1  # Add 1 ms to avoid duplicates
            data = fetch_ohlcv(exchange, symbol, timeframe, since=since)
            if data:
                store_prices(session, symbol, timeframe, data)
                logger.info(f"Stored {len(data)} candles for {symbol} on {timeframe}")
            else:
                logger.info(f"No new data for {symbol} on {timeframe}")
            time.sleep(exchange.rateLimit / 1000)  # Respect API rate limits
    session.close()

if __name__ == "__main__":
    main()

7.5. Logging Setup (utils/logger.py)

Módulo de utilidad para manejar logging.

# utils/logger.py

import logging
import os
from config import LOG_DIR

def setup_logger(name):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

7.6. Execute Data Ingestion

Para obtener y almacenar datos, ejecuta:

python fetch_data.py

7.7. Automate Data Ingestion with Cron

Para automatizar la ingesta de datos cada 5 minutos en Linux:

1. Abre el editor de cron:

crontab -e


2. Añade la siguiente línea:

*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/fetch_data.py >> /path/to/trading_strategy/logs/fetch_data_cron.log 2>&1



Nota: Reemplaza /path/to/trading_strategy/ con la ruta real de tu proyecto.


---

8. Indicator Calculation Module

Este módulo calcula indicadores técnicos basados en los datos de precios obtenidos y los almacena en la base de datos.

8.1. Indicator Calculation Script (calculate_indicators.py)

# calculate_indicators.py

import pandas as pd
from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from db.models import Price, Indicator
from ta import trend, momentum
from config import SYMBOLS, TIMEFRAMES
from utils.logger import setup_logger

logger = setup_logger('calculate_indicators')

def calculate_indicators_for_symbol_timeframe(session: Session, symbol: str, timeframe: str):
    logger.info(f"Calculando indicadores para {symbol} en {timeframe}")
    
    # Fetch price data
    prices = session.query(Price).filter_by(symbol=symbol, timeframe=timeframe).order_by(Price.timestamp).all()
    
    if not prices:
        logger.warning(f"No hay datos de precios para {symbol} en {timeframe}")
        return
    
    df = pd.DataFrame([{
        'timestamp': p.timestamp,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices])
    
    df.set_index('timestamp', inplace=True)
    
    try:
        # Calculate RSI
        df['rsi'] = momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # Calculate MACD
        macd = trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Calculate ADX
        df['adx'] = trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        
        # Calculate OBV
        df['obv'] = trend.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # Calculate DEMA
        df['dema'] = trend.DoubleEMAIndicator(close=df['close'], window=21).dema()
        
        # Calculate Stoch RSI
        stoch_rsi = momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
        
        # Calculate Ichimoku
        ichimoku = trend.IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_span_a'] = ichimoku.ichimoku_a()
        df['ichimoku_span_b'] = ichimoku.ichimoku_b()
        # Note: Ichimoku Chikou Span requires shifting, handled separately if needed
        
        # Calculate VWAP as a proxy for VPVR
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Calculate Pivot High and Pivot Low (Simple)
        pivot_window = 5
        df['pivot_high'] = df['high'].rolling(window=pivot_window*2 +1, center=True).max()
        df['pivot_low'] = df['low'].rolling(window=pivot_window*2 +1, center=True).min()
        
        # Calculate Fractals (Fractal Up and Fractal Down)
        fractal_window = 2
        df['fractal_up'] = df['high'].rolling(window=2*fractal_window +1, center=True).apply(lambda x: x[fractal_window] if x[fractal_window] == max(x) else pd.NA, raw=True)
        df['fractal_down'] = df['low'].rolling(window=2*fractal_window +1, center=True).apply(lambda x: x[fractal_window] if x[fractal_window] == min(x) else pd.NA, raw=True)
        
        # Calculate Correlation between Close and OBV
        df['correlation'] = df['close'].rolling(window=14).corr(df['obv'])
        
        # Placeholder for divergence detection (requires custom implementation)
        df['divergence_bull'] = False
        df['divergence_bear'] = False
        
        # Save indicators to the database
        for timestamp, row in df.iterrows():
            indicator = Indicator(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                rsi=row.get('rsi'),
                macd_line=row.get('macd_line'),
                macd_signal=row.get('macd_signal'),
                macd_hist=row.get('macd_hist'),
                ichimoku_conv=row.get('ichimoku_conv'),
                ichimoku_base=row.get('ichimoku_base'),
                ichimoku_span_a=row.get('ichimoku_span_a'),
                ichimoku_span_b=row.get('ichimoku_span_b'),
                ichimoku_chikou=row.get('ichimoku_conv'),  # Adjust if actual Chikou Span is calculated
                adx=row.get('adx'),
                obv=row.get('obv'),
                dema=row.get('dema'),
                stoch_rsi_k=row.get('stoch_rsi_k'),
                stoch_rsi_d=row.get('stoch_rsi_d'),
                vwap=row.get('vwap'),
                pivot_high=row.get('pivot_high'),
                pivot_low=row.get('pivot_low'),
                fractal_up=row.get('fractal_up'),
                fractal_down=row.get('fractal_down'),
                correlation=row.get('correlation'),
                divergence_bull=row.get('divergence_bull'),
                divergence_bear=row.get('divergence_bear')
            )
            try:
                session.merge(indicator)  # Merge to avoid duplicates
            except Exception as e:
                logger.error(f"Error storing indicator data: {e}")
        session.commit()
        logger.info(f"Indicadores calculados y almacenados para {symbol} en {timeframe}")
    except Exception as e:
        logger.error(f"Error calculando indicadores para {symbol} en {timeframe}: {e}")

def main():
    session = SessionLocal()
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            calculate_indicators_for_symbol_timeframe(session, symbol, timeframe)
    session.close()

if __name__ == "__main__":
    main()

8.2. Execute Indicator Calculation

Para calcular y almacenar indicadores, ejecuta:

python calculate_indicators.py

8.3. Automate Indicator Calculation with Cron

Para automatizar el cálculo de indicadores cada 5 minutos en Linux:

1. Abre el editor de cron:

crontab -e


2. Añade la siguiente línea:

*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/calculate_indicators.py >> /path/to/trading_strategy/logs/calculate_indicators_cron.log 2>&1



Nota: Reemplaza /path/to/trading_strategy/ con la ruta real de tu proyecto.


---

9. Strategy Implementation Module

Este módulo define la lógica de la estrategia de trading basada en los indicadores calculados y genera señales de trading.

9.1. Strategy Definition (strategies/advanced_strategy.py)

# strategies/advanced_strategy.py

from sqlalchemy.orm import Session
from db.models import Indicator, Signal
from config import SYMBOLS, TIMEFRAMES
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger('advanced_strategy')

def generate_signals_for_symbol_timeframe(session: Session, symbol: str, timeframe: str):
    logger.info(f"Generating signals for {symbol} on {timeframe}")
    
    # Fetch the latest indicator
    latest_indicator = session.query(Indicator).filter_by(symbol=symbol, timeframe=timeframe).order_by(Indicator.timestamp.desc()).first()
    
    if not latest_indicator:
        logger.warning(f"No indicators found for {symbol} on {timeframe}")
        return
    
    # Define strategy conditions
    long_conditions = [
        latest_indicator.rsi > 50,
        latest_indicator.macd_line > latest_indicator.macd_signal,
        latest_indicator.stoch_rsi_k > latest_indicator.stoch_rsi_d,
        latest_indicator.dema > latest_indicator.ichimoku_conv,
        latest_indicator.obv > latest_indicator.vwap,
        latest_indicator.adx > 25,
        latest_indicator.correlation > 0.8,
        latest_indicator.vwap > latest_indicator.ichimoku_span_a,
        pd.notna(latest_indicator.fractal_down),
        pd.notna(latest_indicator.pivot_low)
    ]
    
    short_conditions = [
        latest_indicator.rsi < 50,
        latest_indicator.macd_line < latest_indicator.macd_signal,
        latest_indicator.stoch_rsi_k < latest_indicator.stoch_rsi_d,
        latest_indicator.dema < latest_indicator.ichimoku_conv,
        latest_indicator.obv < latest_indicator.vwap,
        latest_indicator.adx > 25,
        latest_indicator.correlation < -0.8,
        latest_indicator.vwap < latest_indicator.ichimoku_span_a,
        pd.notna(latest_indicator.fractal_up),
        pd.notna(latest_indicator.pivot_high)
    ]
    
    # Count fulfilled conditions
    long_signal_count = sum(long_conditions)
    short_signal_count = sum(short_conditions)
    
    # Define thresholds
    ENTER_LONG_THRESHOLD = 7
    ENTER_SHORT_THRESHOLD = 7
    
    # Generate signals based on thresholds
    if long_signal_count >= ENTER_LONG_THRESHOLD:
        signal = Signal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=latest_indicator.timestamp,
            signal_type='long',
            strategy_name='MSLMR Advanced Strategy V6'
        )
        try:
            session.merge(signal)
            session.commit()
            logger.info(f"Long signal generated for {symbol} on {timeframe}")
        except Exception as e:
            logger.error(f"Error generating long signal for {symbol} on {timeframe}: {e}")
    
    if short_signal_count >= ENTER_SHORT_THRESHOLD:
        signal = Signal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=latest_indicator.timestamp,
            signal_type='short',
            strategy_name='MSLMR Advanced Strategy V6'
        )
        try:
            session.merge(signal)
            session.commit()
            logger.info(f"Short signal generated for {symbol} on {timeframe}")
        except Exception as e:
            logger.error(f"Error generating short signal for {symbol} on {timeframe}: {e}")

9.2. Signal Generation Script (generate_signals.py)

# generate_signals.py

from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from strategies.advanced_strategy import generate_signals_for_symbol_timeframe
from config import SYMBOLS, TIMEFRAMES
from utils.logger import setup_logger

logger = setup_logger('generate_signals')

def main():
    session = SessionLocal()
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            generate_signals_for_symbol_timeframe(session, symbol, timeframe)
    session.close()

if __name__ == "__main__":
    main()

9.3. Execute Signal Generation

Para generar señales de trading, ejecuta:

python generate_signals.py

9.4. Automate Signal Generation with Cron

Para automatizar la generación de señales cada 5 minutos en Linux:

1. Abre el editor de cron:

crontab -e


2. Añade la siguiente línea:

*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/generate_signals.py >> /path/to/trading_strategy/logs/generate_signals_cron.log 2>&1



Nota: Reemplaza /path/to/trading_strategy/ con la ruta real de tu proyecto.


---

10. Backtesting Module

Este módulo simula la estrategia de trading sobre datos históricos para evaluar su desempeño.

10.1. Backtesting Script (run_backtest.py)

# run_backtest.py

import pandas as pd
from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from db.models import Price, Indicator, Signal, Trade, Backtest
from strategies.advanced_strategy import generate_signals_for_symbol_timeframe
from config import SYMBOLS, TIMEFRAMES
from utils.logger import setup_logger
from datetime import datetime

logger = setup_logger('backtest')

def run_backtest_for_symbol_timeframe(session: Session, symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    logger.info(f"Starting backtest for {symbol} on {timeframe} from {start_date} to {end_date}")
    
    # Fetch price and indicator data within the date range
    prices = session.query(Price).filter(
        Price.symbol == symbol,
        Price.timeframe == timeframe,
        Price.timestamp >= start_date,
        Price.timestamp <= end_date
    ).order_by(Price.timestamp).all()
    
    indicators = session.query(Indicator).filter(
        Indicator.symbol == symbol,
        Indicator.timeframe == timeframe,
        Indicator.timestamp >= start_date,
        Indicator.timestamp <= end_date
    ).order_by(Indicator.timestamp).all()
    
    if not prices or not indicators:
        logger.warning(f"Insufficient data for backtest on {symbol} {timeframe}")
        return
    
    # Convert data to DataFrames
    df_prices = pd.DataFrame([{
        'timestamp': p.timestamp,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices]).set_index('timestamp')
    
    df_indicators = pd.DataFrame([{
        'timestamp': ind.timestamp,
        'rsi': ind.rsi,
        'macd_line': ind.macd_line,
        'macd_signal': ind.macd_signal,
        'macd_hist': ind.macd_hist,
        'ichimoku_conv': ind.ichimoku_conv,
        'ichimoku_base': ind.ichimoku_base,
        'ichimoku_span_a': ind.ichimoku_span_a,
        'ichimoku_span_b': ind.ichimoku_span_b,
        'ichimoku_chikou': ind.ichimoku_chikou,
        'adx': ind.adx,
        'obv': ind.obv,
        'dema': ind.dema,
        'stoch_rsi_k': ind.stoch_rsi_k,
        'stoch_rsi_d': ind.stoch_rsi_d,
        'vwap': ind.vwap,
        'pivot_high': ind.pivot_high,
        'pivot_low': ind.pivot_low,
        'fractal_up': ind.fractal_up,
        'fractal_down': ind.fractal_down,
        'correlation': ind.correlation,
        'divergence_bull': ind.divergence_bull,
        'divergence_bear': ind.divergence_bear
    } for ind in indicators]).set_index('timestamp')
    
    # Merge price and indicators data
    df = df_prices.join(df_indicators, how='inner')
    
    # Initialize backtest variables
    position = None  # 'long' or 'short'
    entry_price = 0.0
    equity = 10000.0  # Starting capital
    equity_curve = []
    trades_list = []
    
    for timestamp, row in df.iterrows():
        # Define strategy conditions based on indicators
        long_signal = (
            row['rsi'] > 50 and
            row['macd_line'] > row['macd_signal'] and
            row['stoch_rsi_k'] > row['stoch_rsi_d'] and
            row['dema'] > row['ichimoku_conv'] and
            row['obv'] > row['vwap'] and
            row['adx'] > 25 and
            row['correlation'] > 0.8 and
            row['vwap'] > row['ichimoku_span_a'] and
            pd.notna(row['fractal_down']) and
            pd.notna(row['pivot_low'])
        )
        
        short_signal = (
            row['rsi'] < 50 and
            row['macd_line'] < row['macd_signal'] and
            row['stoch_rsi_k'] < row['stoch_rsi_d'] and
            row['dema'] < row['ichimoku_conv'] and
            row['obv'] < row['vwap'] and
            row['adx'] > 25 and
            row['correlation'] < -0.8 and
            row['vwap'] < row['ichimoku_span_a'] and
            pd.notna(row['fractal_up']) and
            pd.notna(row['pivot_high'])
        )
        
        # Execute trading logic based on signals
        if long_signal and position != 'long':
            # Close short position if exists
            if position == 'short':
                profit = entry_price - row['close']
                equity += profit
                trades_list.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'side': 'cover',
                    'price': row['close'],
                    'size': 1,
                    'strategy_name': 'MSLMR Advanced Strategy V6',
                    'profit_loss': profit
                })
                logger.info(f"Cover SHORT at {row['close']} with P/L: {profit}")
            
            # Open long position
            position = 'long'
            entry_price = row['close']
            trades_list.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'side': 'buy',
                'price': row['close'],
                'size': 1,
                'strategy_name': 'MSLMR Advanced Strategy V6',
                'profit_loss': 0.0
            })
            logger.info(f"Enter LONG at {row['close']}")
        
        elif short_signal and position != 'short':
            # Close long position if exists
            if position == 'long':
                profit = row['close'] - entry_price
                equity += profit
                trades_list.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'side': 'sell',
                    'price': row['close'],
                    'size': 1,
                    'strategy_name': 'MSLMR Advanced Strategy V6',
                    'profit_loss': profit
                })
                logger.info(f"Exit LONG at {row['close']} with P/L: {profit}")
            
            # Open short position
            position = 'short'
            entry_price = row['close']
            trades_list.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'side': 'sell',
                'price': row['close'],
                'size': 1,
                'strategy_name': 'MSLMR Advanced Strategy V6',
                'profit_loss': 0.0
            })
            logger.info(f"Enter SHORT at {row['close']}")
        
        # Update equity curve
        equity_curve.append({'timestamp': timestamp, 'equity': equity})
    
    # Close any open position at the end of backtest
    if position == 'long':
        profit = df.iloc[-1]['close'] - entry_price
        equity += profit
        trades_list.append({
            'symbol': symbol,
            'timestamp': df.index[-1],
            'side': 'sell',
            'price': df.iloc[-1]['close'],
            'size': 1,
            'strategy_name': 'MSLMR Advanced Strategy V6',
            'profit_loss': profit
        })
        logger.info(f"Exit LONG at {df.iloc[-1]['close']} with P/L: {profit}")
    elif position == 'short':
        profit = entry_price - df.iloc[-1]['close']
        equity += profit
        trades_list.append({
            'symbol': symbol,
            'timestamp': df.index[-1],
            'side': 'cover',
            'price': df.iloc[-1]['close'],
            'size': 1,
            'strategy_name': 'MSLMR Advanced Strategy V6',
            'profit_loss': profit
        })
        logger.info(f"Cover SHORT at {df.iloc[-1]['close']} with P/L: {profit}")
    
    # Save trades to the database
    for trade in trades_list:
        trade_entry = Trade(
            symbol=trade['symbol'],
            timestamp=trade['timestamp'],
            side=trade['side'],
            price=trade['price'],
            size=trade['size'],
            strategy_name=trade['strategy_name'],
            profit_loss=trade['profit_loss']
        )
        try:
            session.add(trade_entry)
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    session.commit()
    
    # Calculate backtest metrics
    df_equity = pd.DataFrame(equity_curve)
    total_return = ((df_equity['equity'].iloc[-1] - df_equity['equity'].iloc[0]) / df_equity['equity'].iloc[0]) * 100
    max_drawdown = ((df_equity['equity'].cummax() - df_equity['equity']).max() / df_equity['equity'].cummax().max()) * 100
    sharpe_ratio = (df_equity['equity'].pct_change().mean() / df_equity['equity'].pct_change().std()) * (252**0.5)  # Annualized Sharpe Ratio
    trades_count = len(trades_list)
    profitable_trades = len([t for t in trades_list if t['profit_loss'] > 0])
    win_rate = (profitable_trades / trades_count) * 100 if trades_count > 0 else 0.0
    
    # Save backtest results to the database
    backtest_result = Backtest(
        strategy_name='MSLMR Advanced Strategy V6',
        start_date=start_date,
        end_date=end_date,
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        trades=trades_count,
        profitable_trades=profitable_trades
    )
    try:
        session.add(backtest_result)
        session.commit()
        logger.info(f"Backtest completed for {symbol} on {timeframe}")
    except Exception as e:
        logger.error(f"Error saving backtest result: {e}")

def main():
    session = SessionLocal()
    # Define the backtest period
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            run_backtest_for_symbol_timeframe(session, symbol, timeframe, start_date, end_date)
    session.close()

if __name__ == "__main__":
    main()

10.2. Execute Backtesting

Para realizar backtesting, ejecuta:

python run_backtest.py


---

11. Live Execution Module

Este módulo ejecuta trades en tiempo real basados en señales generadas e incorpora gestión de riesgos y alertas.

11.1. Risk Management Functions (utils/risk_management.py)

# utils/risk_management.py

import math
from config import RISK_PER_TRADE, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, MAX_GLOBAL_EXPOSURE
from db.models import Trade
from sqlalchemy.orm import Session

def calculate_position_size(account_balance, risk_per_trade, stop_loss_percent, price):
    """
    Calculate the position size based on the account balance, risk per trade, stop loss, and current price.
    """
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / (stop_loss_percent * price)
    return math.floor(position_size)

def check_max_exposure(session: Session, account_balance, max_exposure):
    """
    Check if the current exposure exceeds the maximum allowed exposure.
    """
    trades_query = session.query(Trade).filter(
        Trade.side.in_(['buy', 'sell'])
    ).with_entities(
        Trade.price, Trade.size, Trade.side
    ).all()
    
    total_exposure = 0.0
    for trade in trades_query:
        if trade.side == 'buy':
            total_exposure += trade.price * trade.size
        elif trade.side == 'sell':
            total_exposure -= trade.price * trade.size
    
    return (abs(total_exposure) / account_balance) < max_exposure

def set_stop_loss_take_profit(signal_type, price, side, position_size):
    """
    Calculate Stop Loss and Take Profit levels based on the trade side and price.
    """
    if side == 'buy':
        stop_loss = price * (1 - STOP_LOSS_PERCENT)
        take_profit = price * (1 + TAKE_PROFIT_PERCENT)
    elif side == 'sell':
        stop_loss = price * (1 + STOP_LOSS_PERCENT)
        take_profit = price * (1 - TAKE_PROFIT_PERCENT)
    else:
        stop_loss = None
        take_profit = None
    
    return stop_loss, take_profit

11.2. Alert Functions (utils/alerts.py)

# utils/alerts.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from config import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD, EMAIL_RECEIVER
from utils.logger import setup_logger

logger = setup_logger('alerts')

def send_telegram_message(message):
    """
    Send a message via Telegram bot.
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logger.error(f"Error sending Telegram message: {response.text}")
    except Exception as e:
        logger.error(f"Exception sending Telegram message: {e}")

def send_email(subject, body):
    """
    Send an email notification.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_HOST_USER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_HOST_USER, EMAIL_RECEIVER, text)
        server.quit()
    except Exception as e:
        logger.error(f"Error sending email: {e}")

def alert_trade(trade):
    """
    Send alerts upon trade execution.
    """
    message = f"""
    <b>Trade Executed</b>
    <b>Symbol:</b> {trade['symbol']}
    <b>Side:</b> {trade['side']}
    <b>Price:</b> {trade['price']}
    <b>Size:</b> {trade['size']}
    <b>Strategy:</b> {trade['strategy_name']}
    <b>Profit/Loss:</b> {trade['profit_loss']}
    """
    send_telegram_message(message)
    
    email_subject = f"Trade Executed: {trade['symbol']} - {trade['side']}"
    email_body = f"""
    Trade Executed:
    Symbol: {trade['symbol']}
    Side: {trade['side']}
    Price: {trade['price']}
    Size: {trade['size']}
    Strategy: {trade['strategy_name']}
    Profit/Loss: {trade['profit_loss']}
    """
    send_email(email_subject, email_body)

11.3. Live Execution Script (run_live.py)

# run_live.py

import ccxt
from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from db.models import Signal, Trade
from config import EXCHANGE_ID, SYMBOLS, TIMEFRAMES, RISK_PER_TRADE, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, MAX_GLOBAL_EXPOSURE
from utils.logger import setup_logger
import time
import pandas as pd
from utils.risk_management import calculate_position_size, check_max_exposure, set_stop_loss_take_profit
from utils.alerts import alert_trade
import os

logger = setup_logger('live')

# Placeholder for account balance. Ideally, fetch this from the exchange via API.
ACCOUNT_BALANCE = 10000.0

def execute_trade(exchange, symbol, side, price, size, strategy_name, signal_id, session: Session):
    try:
        # Place a market order
        order = exchange.create_order(symbol, 'market', side, size)
        logger.info(f"Executed {side} order for {symbol} at {price} for size {size}")
        
        # Calculate Stop Loss and Take Profit
        stop_loss, take_profit = set_stop_loss_take_profit(signal_type=side, price=price, side=side, position_size=size)
        
        # Record the trade
        trade = Trade(
            symbol=symbol,
            timestamp=pd.to_datetime(order['timestamp'], unit='ms'),
            side=side,
            price=price,
            size=size,
            strategy_name=strategy_name,
            signal_id=signal_id,
            profit_loss=0.0  # To be updated upon trade closure
        )
        session.add(trade)
        session.commit()
        
        # Send alerts
        alert_trade({
            'symbol': trade.symbol,
            'side': trade.side,
            'price': trade.price,
            'size': trade.size,
            'strategy_name': trade.strategy_name,
            'profit_loss': trade.profit_loss
        })
        
        # Place Stop Loss and Take Profit orders
        if side == 'buy':
            # Place Stop Loss sell order
            exchange.create_order(symbol, 'stop_market', 'sell', size, stop_loss, {'stopPrice': stop_loss})
            # Place Take Profit sell order
            exchange.create_order(symbol, 'limit', 'sell', size, take_profit)
        elif side == 'sell':
            # Place Stop Loss buy order
            exchange.create_order(symbol, 'stop_market', 'buy', size, stop_loss, {'stopPrice': stop_loss})
            # Place Take Profit buy order
            exchange.create_order(symbol, 'limit', 'buy', size, take_profit)
        
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")

def process_signal(session: Session, exchange, signal, account_balance):
    symbol = signal.symbol
    strategy_name = signal.strategy_name
    signal_type = signal.signal_type
    timestamp = signal.timestamp
    
    # Fetch current market price
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    
    # Calculate position size based on risk management
    position_size = calculate_position_size(account_balance, RISK_PER_TRADE, STOP_LOSS_PERCENT, price)
    
    # Check global exposure
    if not check_max_exposure(session, account_balance, MAX_GLOBAL_EXPOSURE):
        logger.warning(f"Maximum exposure reached. Not executing signal for {symbol}.")
        return
    
    # Define trade side based on signal type
    if signal_type == 'long':
        side = 'buy'
    elif signal_type == 'short':
        side = 'sell'
    else:
        logger.warning(f"Unknown signal type: {signal_type}")
        return
    
    # Execute the trade
    execute_trade(exchange, symbol, side, price, position_size, strategy_name, signal.id, session)

def main():
    session = SessionLocal()
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        'apiKey': os.getenv('EXCHANGE_API_KEY', ''),
        'secret': os.getenv('EXCHANGE_SECRET', ''),
        'enableRateLimit': True,
    })
    
    # Fetch unprocessed signals within the last 5 minutes
    signals_to_process = session.query(Signal).filter(
        Signal.timestamp >= pd.to_datetime('now') - pd.Timedelta(minutes=5)
    ).all()
    
    for signal in signals_to_process:
        process_signal(session, exchange, signal, ACCOUNT_BALANCE)
    
    session.close()

if __name__ == "__main__":
    main()

11.4. Execute Live Trading

Para ejecutar trading en vivo basado en señales, ejecuta:

python run_live.py

11.5. Automate Live Execution with Cron

Para automatizar la ejecución en vivo cada 5 minutos en Linux:

1. Abre el editor de cron:

crontab -e


2. Añade la siguiente línea:

*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/run_live.py >> /path/to/trading_strategy/logs/run_live_cron.log 2>&1



Nota: Reemplaza /path/to/trading_strategy/ con la ruta real de tu proyecto.


---

12. Dashboard Module

El dashboard proporciona visualización en tiempo real de actividades de trading, señales, trades y métricas de desempeño usando Flask y Plotly.

12.1. Dashboard Script (dashboard.py)

# dashboard.py

from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from sqlalchemy.orm import Session
from db.db_connection import SessionLocal, engine
from db.models import Signal, Trade, Backtest
from config import SYMBOLS, TIMEFRAMES
from utils.logger import setup_logger
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')  # Replace with a secure key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

logger = setup_logger('dashboard')

# Dummy user class for demonstration
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# User loader callback
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Replace with real authentication logic
        if username == 'admin' and password == 'password':
            user = User(id=1)
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    session = SessionLocal()
    
    # Fetch latest signals
    latest_signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(50).all()
    signals_df = pd.DataFrame([{
        'timestamp': s.timestamp,
        'symbol': s.symbol,
        'timeframe': s.timeframe,
        'signal_type': s.signal_type,
        'strategy_name': s.strategy_name
    } for s in latest_signals])
    
    # Fetch latest trades
    latest_trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(50).all()
    trades_df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'side': t.side,
        'price': t.price,
        'size': t.size,
        'strategy_name': t.strategy_name,
        'profit_loss': t.profit_loss
    } for t in latest_trades])
    
    # Fetch backtest results
    backtest_results = session.query(Backtest).order_by(Backtest.id.desc()).all()
    backtest_df = pd.DataFrame([{
        'strategy_name': b.strategy_name,
        'start_date': b.start_date,
        'end_date': b.end_date,
        'total_return': b.total_return,
        'max_drawdown': b.max_drawdown,
        'sharpe_ratio': b.sharpe_ratio,
        'trades': b.trades,
        'profitable_trades': b.profitable_trades
    } for b in backtest_results])
    
    # Fetch equity curve for a specific symbol and timeframe
    symbol = 'BTC/USDT'
    timeframe = '1h'
    trades_for_equity = session.query(Trade).filter(
        Trade.symbol == symbol,
        Trade.strategy_name == 'MSLMR Advanced Strategy V6'
    ).order_by(Trade.timestamp).all()
    
    equity = 10000.0  # Starting capital
    equity_curve = []
    
    for trade in trades_for_equity:
        if trade.side in ['buy', 'cover']:
            equity -= trade.price * trade.size
        elif trade.side in ['sell']:
            equity += trade.price * trade.size
        equity_curve.append({'timestamp': trade.timestamp, 'equity': equity})
    
    equity_df = pd.DataFrame(equity_curve)
    
    # Create Plotly figures
    fig_signals = px.scatter(signals_df, x='timestamp', y='symbol', color='signal_type',
                             title='Latest Trading Signals')
    
    fig_trades = px.scatter(trades_df, x='timestamp', y='profit_loss', color='side',
                            title='Latest Trades')
    
    if not equity_df.empty:
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'], mode='lines', name='Equity'))
        fig_equity.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity')
    else:
        fig_equity = go.Figure()
        fig_equity.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity')
    
    # Convert figures to HTML
    graph_signals = fig_signals.to_html(full_html=False)
    graph_trades = fig_trades.to_html(full_html=False)
    graph_equity = fig_equity.to_html(full_html=False)
    
    session.close()
    
    return render_template('index.html',
                           signals=signals_df.to_dict('records'),
                           trades=trades_df.to_dict('records'),
                           backtests=backtest_df.to_dict('records'),
                           graph_signals=graph_signals,
                           graph_trades=graph_trades,
                           graph_equity=graph_equity)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

12.2. HTML Templates

12.2.1. Login Template (templates/login.html)

<!-- templates/login.html -->

<!DOCTYPE html>
<html>
<head>
    <title>Login - Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f2f2f2; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .login-container { background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 5px 0 10px 0; border: 1px solid #ccc; border-radius: 4px; }
        input[type=submit] { background-color: #4CAF50; color: white; padding: 10px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        input[type=submit]:hover { background-color: #45a049; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>Login</h2>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        <form method="POST">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>

            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>

            <input type="submit" value="Login">
        </form>
    </div>
</body>
</html>

12.2.2. Main Dashboard Template (templates/index.html)

<!-- templates/index.html -->

<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
        th { background-color: #f2f2f2; }
        .graph { margin-bottom: 40px; }
    </style>
</head>
<body>
    <h1>Trading Dashboard</h1>
    
    <div class="graph">
        <h2>Latest Trading Signals</h2>
        {{ graph_signals | safe }}
    </div>
    
    <div class="graph">
        <h2>Latest Trades</h2>
        {{ graph_trades | safe }}
    </div>
    
    <div class="graph">
        <h2>Equity Curve</h2>
        {{ graph_equity | safe }}
    </div>
    
    <h2>Latest Signals in Table</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Timeframe</th>
            <th>Signal Type</th>
            <th>Strategy</th>
        </tr>
        {% for signal in signals %}
        <tr>
            <td>{{ signal.timestamp }}</td>
            <td>{{ signal.symbol }}</td>
            <td>{{ signal.timeframe }}</td>
            <td>{{ signal.signal_type }}</td>
            <td>{{ signal.strategy_name }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Latest Trades in Table</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Price</th>
            <th>Size</th>
            <th>Strategy</th>
            <th>Profit/Loss</th>
        </tr>
        {% for trade in trades %}
        <tr>
            <td>{{ trade.timestamp }}</td>
            <td>{{ trade.symbol }}</td>
            <td>{{ trade.side }}</td>
            <td>{{ trade.price }}</td>
            <td>{{ trade.size }}</td>
            <td>{{ trade.strategy_name }}</td>
            <td>{{ trade.profit_loss }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Backtest Results</h2>
    <table>
        <tr>
            <th>Strategy Name</th>
            <th>Start Date</th>
            <th>End Date</th>
            <th>Total Return (%)</th>
            <th>Max Drawdown (%)</th>
            <th>Sharpe Ratio</th>
            <th>Trades</th>
            <th>Profitable Trades</th>
        </tr>
        {% for backtest in backtests %}
        <tr>
            <td>{{ backtest.strategy_name }}</td>
            <td>{{ backtest.start_date }}</td>
            <td>{{ backtest.end_date }}</td>
            <td>{{ backtest.total_return }}</td>
            <td>{{ backtest.max_drawdown }}</td>
            <td>{{ backtest.sharpe_ratio }}</td>
            <td>{{ backtest.trades }}</td>
            <td>{{ backtest.profitable_trades }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <a href="{{ url_for('logout') }}">Logout</a>
</body>
</html>

12.3. Execute the Dashboard

Para ejecutar el dashboard, ejecuta:

python dashboard.py

Accede al dashboard navegando a http://localhost:8000/login en tu navegador y usa las credenciales definidas en la ruta login (admin / password en este ejemplo). Asegúrate de reemplazar estas credenciales con unas seguras para producción.


---

13. Risk Management Module

La gestión efectiva de riesgos es crucial para proteger el capital y asegurar un desempeño sostenible en el trading.

13.1. Risk Management Functions (utils/risk_management.py)

# utils/risk_management.py

import math
from config import RISK_PER_TRADE, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, MAX_GLOBAL_EXPOSURE
from db.models import Trade
from sqlalchemy.orm import Session

def calculate_position_size(account_balance, risk_per_trade, stop_loss_percent, price):
    """
    Calculate the position size based on account balance, risk per trade, stop loss percentage, and current price.
    """
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / (stop_loss_percent * price)
    return math.floor(position_size)

def check_max_exposure(session: Session, account_balance, max_exposure):
    """
    Check if the current exposure exceeds the maximum allowed exposure.
    """
    trades_query = session.query(Trade).filter(
        Trade.side.in_(['buy', 'sell'])
    ).with_entities(
        Trade.price, Trade.size, Trade.side
    ).all()
    
    total_exposure = 0.0
    for trade in trades_query:
        if trade.side == 'buy':
            total_exposure += trade.price * trade.size
        elif trade.side == 'sell':
            total_exposure -= trade.price * trade.size
    
    return (abs(total_exposure) / account_balance) < max_exposure

def set_stop_loss_take_profit(signal_type, price, side, position_size):
    """
    Calculate Stop Loss and Take Profit levels based on trade side and price.
    """
    if side == 'buy':
        stop_loss = price * (1 - STOP_LOSS_PERCENT)
        take_profit = price * (1 + TAKE_PROFIT_PERCENT)
    elif side == 'sell':
        stop_loss = price * (1 + STOP_LOSS_PERCENT)
        take_profit = price * (1 - TAKE_PROFIT_PERCENT)
    else:
        stop_loss = None
        take_profit = None
    
    return stop_loss, take_profit

13.2. Integrate Risk Management into Live Execution

El script run_live.py ha sido actualizado para incorporar la gestión de riesgos. Consulta la sección Live Execution Module para más detalles.


---

14. Advanced Alerts Module

Este módulo envía notificaciones vía Telegram y Email cuando se ejecutan trades o ocurren eventos significativos.

14.1. Alert Functions (utils/alerts.py)

(Consulta la sección 14.1. Alert Functions)

14.2. Configure Telegram Bot

1. Create a Telegram Bot:

Abre Telegram y busca @BotFather.

Inicia una conversación y envía /newbot.

Sigue las indicaciones para crear un nuevo bot y obtener el API Token.



2. Obtain Chat ID:

Envía un mensaje a tu nuevo bot.

Usa el siguiente script para obtener tu chat_id:


# get_chat_id.py

import requests

API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

def get_updates():
    url = f"https://api.telegram.org/bot{API_TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()
    return data

if __name__ == "__main__":
    updates = get_updates()
    print(updates)

Ejecuta el script y busca el chat_id en la respuesta.



14.3. Configure Email SMTP

Usa Gmail u otro servicio SMTP. Para Gmail:

1. Enable "Less Secure Apps":

Visita Google Account Security.

Habilita "Less secure app access" o usa una App Password si tienes 2FA habilitado.



2. Set SMTP Credentials:

Actualiza config.py con tus detalles SMTP:


EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_HOST_USER = 'your_email@gmail.com'
EMAIL_HOST_PASSWORD = 'your_email_password'
EMAIL_RECEIVER = 'receiver_email@gmail.com'




---

15. Automation and Deployment

Automatiza la ejecución de los distintos scripts y despliega el sistema usando Docker para consistencia y facilidad de gestión.

15.1. Docker and Docker Compose Setup

15.1.1. Install Docker

On Ubuntu:

sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

On Windows:

Descarga e instala Docker Desktop desde docker.com.

15.1.2. Install Docker Compose

On Ubuntu:

sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

On Windows:

Docker Desktop incluye Docker Compose.

15.1.3. Create a Dockerfile

# Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (to be overridden by docker-compose)
CMD ["python", "run_live.py"]

15.1.4. Create docker-compose.yml

# docker-compose.yml

version: '3.8'

services:
  db:
    image: postgres:13
    environment:
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: trading_db
    volumes:
      - db_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  app:
    build: .
    depends_on:
      - db
    environment:
      - DB_USER=trading_user
      - DB_PASSWORD=secure_password
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=trading_db
      - EXCHANGE_ID=binance
      - TELEGRAM_API_TOKEN=your_telegram_api_token
      - TELEGRAM_CHAT_ID=your_telegram_chat_id
      - EMAIL_HOST=smtp.gmail.com
      - EMAIL_PORT=587
      - EMAIL_HOST_USER=your_email@gmail.com
      - EMAIL_HOST_PASSWORD=your_email_password
      - EMAIL_RECEIVER=receiver_email@gmail.com
      - RISK_PER_TRADE=0.01
      - STOP_LOSS_PERCENT=0.02
      - TAKE_PROFIT_PERCENT=0.04
      - MAX_GLOBAL_EXPOSURE=0.20
    volumes:
      - .:/app
    command: python run_live.py
    restart: always
    # Optional: Map ports for the dashboard
    # ports:
    #   - "8000:8000"

  dashboard:
    build: .
    depends_on:
      - db
    environment:
      - DB_USER=trading_user
      - DB_PASSWORD=secure_password
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=trading_db
      - EXCHANGE_ID=binance
      - TELEGRAM_API_TOKEN=your_telegram_api_token
      - TELEGRAM_CHAT_ID=your_telegram_chat_id
      - EMAIL_HOST=smtp.gmail.com
      - EMAIL_PORT=587
      - EMAIL_HOST_USER=your_email@gmail.com
      - EMAIL_HOST_PASSWORD=your_email_password
      - EMAIL_RECEIVER=receiver_email@gmail.com
      - RISK_PER_TRADE=0.01
      - STOP_LOSS_PERCENT=0.02
      - TAKE_PROFIT_PERCENT=0.04
      - MAX_GLOBAL_EXPOSURE=0.20
    volumes:
      - .:/app
    command: python dashboard.py
    ports:
      - "8000:8000"
    restart: always

volumes:
  db_data:

15.1.5. Build and Run Docker Containers

docker-compose build
docker-compose up -d

Check Logs:

docker-compose logs -f app
docker-compose logs -f dashboard

15.2. Automate Script Execution with Cron (Alternative to Docker)

Si prefieres no usar Docker, puedes automatizar la ejecución de scripts usando cron en Linux.

1. Edit Crontab:

crontab -e


2. Add Cron Jobs:

# Data Ingestion every 5 minutes
*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/fetch_data.py >> /path/to/trading_strategy/logs/fetch_data_cron.log 2>&1

# Indicator Calculation every 5 minutes
*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/calculate_indicators.py >> /path/to/trading_strategy/logs/calculate_indicators_cron.log 2>&1

# Signal Generation every 5 minutes
*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/generate_signals.py >> /path/to/trading_strategy/logs/generate_signals_cron.log 2>&1

# Live Execution every 5 minutes
*/5 * * * * /path/to/trading_strategy/venv/bin/python /path/to/trading_strategy/run_live.py >> /path/to/trading_strategy/logs/run_live_cron.log 2>&1



Nota: Reemplaza /path/to/trading_strategy/ con la ruta real de tu proyecto.


---

16. Security and Robustness

16.1. Exception Handling

Asegúrate de que todos los scripts manejen excepciones de manera adecuada para prevenir fallos inesperados.

try:
    # Código que puede generar excepciones
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")

16.2. Data Validation

Valida los datos antes de procesarlos para mantener la integridad de los mismos.

if pd.isna(row['rsi']):
    logger.warning(f"RSI is NaN at {timestamp}")
    continue

16.3. Secure Credential Management

Evita hardcodear credenciales sensibles en el código. Utiliza variables de entorno o soluciones de almacenamiento seguro.

Using Environment Variables

1. Modify config.py to Use Environment Variables:

# config.py

import os

# Database Configuration
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'trading_user'),
    'password': os.getenv('DB_PASSWORD', 'secure_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'trading_db')
}

# Exchange Configuration
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance')
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Update as needed
TIMEFRAMES = ['1m', '5m', '1h', '1d']

# Data Ingestion Parameters
FETCH_LIMIT = 1000

# Logging Directory
LOG_DIR = 'logs'

# Risk Management Parameters
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 0.04))
MAX_GLOBAL_EXPOSURE = float(os.getenv('MAX_GLOBAL_EXPOSURE', 0.20))

# Alert Configuration
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN', 'your_telegram_api_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_telegram_chat_id')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', 'your_email@gmail.com')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', 'your_email_password')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER', 'receiver_email@gmail.com')


2. Set Environment Variables:

export DB_USER=trading_user
export DB_PASSWORD=secure_password
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=trading_db
export EXCHANGE_ID=binance
export TELEGRAM_API_TOKEN=your_telegram_api_token
export TELEGRAM_CHAT_ID=your_telegram_chat_id
export EMAIL_HOST=smtp.gmail.com
export EMAIL_PORT=587
export EMAIL_HOST_USER=your_email@gmail.com
export EMAIL_HOST_PASSWORD=your_email_password
export EMAIL_RECEIVER=receiver_email@gmail.com
export RISK_PER_TRADE=0.01
export STOP_LOSS_PERCENT=0.02
export TAKE_PROFIT_PERCENT=0.04
export MAX_GLOBAL_EXPOSURE=0.20



Nota: Para producción, considera usar herramientas como Docker Secrets, AWS Secrets Manager o HashiCorp Vault para manejar secretos de manera segura.

16.4. User Authentication for Dashboard

Implementa autenticación de usuarios para restringir el acceso al dashboard.

Update dashboard.py with Flask-Login

(Consulta la sección 12.1. Dashboard Script (dashboard.py))

Nota: Para producción, implementa un sistema de gestión de usuarios seguro con contraseñas hasheadas y roles de usuario.


---

17. Monitoring and Logging

Implementa logging y monitorización completa para rastrear el rendimiento del sistema y solucionar problemas.

17.1. Logging Configuration

El logging está configurado en cada módulo usando el utilitario utils/logger.py. Los logs se almacenan en el directorio logs/.

17.2. Dashboard Monitoring

El dashboard proporciona visualización en tiempo real de actividades de trading, señales, trades y resultados de backtest.

17.3. Additional Monitoring Tools

Considera integrar herramientas de monitorización como Prometheus y Grafana para métricas avanzadas y capacidades de alerting.


---

18. Additional Recommendations

1. Optimize Performance:

Utiliza consultas de base de datos eficientes.

Cachea datos de acceso frecuente.

Optimiza los cálculos de indicadores.



2. Implement More Strategies:

Desarrolla estrategias adicionales para diversificar enfoques de trading.



3. Enhance the Dashboard:

Añade más visualizaciones como heatmaps, gráficos de rendimiento, etc.

Implementa gestión de usuarios para múltiples usuarios.



4. Advanced Risk Management:

Incorpora sizing de posición basado en volatilidad.

Implementa ajustes dinámicos de stop-loss y take-profit.



5. Security Enhancements:

Usa HTTPS para el dashboard.

Implementa autenticación de dos factores.



6. Scalability:

Utiliza colas de mensajes como RabbitMQ o Kafka para manejar datos de alta frecuencia.

Distribuye componentes en múltiples servidores si es necesario.



7. Implementación de Estrategias Adicionales:

Añade nuevas estrategias y módulos para diversificar tus operaciones.



8. Documentación y Mantenimiento:

Documenta todo el sistema para facilitar el mantenimiento y futuras mejoras.

Implementa pruebas unitarias y de integración para asegurar la estabilidad del sistema.



9. Automatización y Orquestación Avanzada:

Considera el uso de herramientas como Airflow o Prefect para orquestar flujos de trabajo más complejos.





---

19. Appendix: Complete Code Listings

19.1. config.py

# config.py

import os

# Database Configuration
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'trading_user'),
    'password': os.getenv('DB_PASSWORD', 'secure_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'trading_db')
}

# Exchange Configuration
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance')
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Update as needed
TIMEFRAMES = ['1m', '5m', '1h', '1d']

# Data Ingestion Parameters
FETCH_LIMIT = 1000

# Logging Directory
LOG_DIR = 'logs'

# Risk Management Parameters
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 0.04))
MAX_GLOBAL_EXPOSURE = float(os.getenv('MAX_GLOBAL_EXPOSURE', 0.20))

# Alert Configuration
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN', 'your_telegram_api_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_telegram_chat_id')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', 'your_email@gmail.com')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', 'your_email_password')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER', 'receiver_email@gmail.com')

19.2. db/db_connection.py

# db/db_connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DB_CONFIG

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

19.3. db/models.py

# db/models.py

from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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
    signal_type = Column(String, nullable=False)  # 'long', 'short', etc.
    strategy_name = Column(String)
    __table_args__ = (UniqueConstraint('symbol', 'timeframe', 'timestamp', 'signal_type', name='_symbol_timeframe_timestamp_signal_uc'),)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    side = Column(String, nullable=False)  # 'buy' o 'sell'
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

19.4. utils/logger.py

# utils/logger.py

import logging
import os
from config import LOG_DIR

def setup_logger(name):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

19.5. fetch_data.py

(Consulta la sección 7.4. Data Ingestion Script (fetch_data.py))

19.6. calculate_indicators.py

(Consulta la sección 8.1. Indicator Calculation Script (calculate_indicators.py))

19.7. strategies/advanced_strategy.py

(Consulta la sección 9.1. Strategy Definition (strategies/advanced_strategy.py))

19.8. generate_signals.py

(Consulta la sección 9.2. Signal Generation Script (generate_signals.py))

19.9. run_backtest.py

(Consulta la sección 10.1. Backtesting Script (run_backtest.py))

19.10. run_live.py

(Consulta la sección 11.3. Live Execution Script (run_live.py))

19.11. dashboard.py

(Consulta la sección 12.1. Dashboard Script (dashboard.py))

19.12. utils/risk_management.py

(Consulta la sección 13.1. Risk Management Functions (utils/risk_management.py))

19.13. utils/alerts.py

(Consulta la sección 14.1. Alert Functions (utils/alerts.py))


---

Fin de la Documentación


---
