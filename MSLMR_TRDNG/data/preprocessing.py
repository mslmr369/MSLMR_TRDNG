import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class DataPreprocessor:
    def __init__(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]],
        indicators: Optional[List[str]] = None
    ):
        """
        Preprocesador de datos de mercado
        
        :param data: Datos de mercado por símbolo y timeframe
        :param indicators: Lista de indicadores a calcular
        """
        self.original_data = data
        self.processed_data = {}
        
        # Indicadores por defecto
        self.indicators = indicators or [
            'rsi', 'macd', 'ema', 'bollinger', 'atr'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el Relative Strength Index (RSI)
        
        :param data: DataFrame de precios
        :param period: Período de cálculo
        :return: Serie con valores de RSI
        """
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self, 
        data: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calcula Moving Average Convergence Divergence (MACD)
        
        :param data: DataFrame de precios
        :return: Diccionario con líneas MACD
        """
        exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def preprocess_data(
        self, 
        drop_na: bool = True, 
        normalize: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Preprocesa datos de mercado
        
        :param drop_na: Eliminar filas con valores nulos
        :param normalize: Normalizar valores
        :return: Datos procesados
        """
        for symbol, timeframe_data in self.original_data.items():
            symbol_processed = {}
            
            for timeframe, df in timeframe_data.items():
                # Copiar datos originales
                processed_df = df.copy()
                
                # Calcular indicadores
                if 'rsi' in self.indicators:
                    processed_df['rsi'] = self._calculate_rsi(df)
                
                if 'macd' in self.indicators:
                    macd_data = self._calculate_macd(df)
                    processed_df.update(macd_data)
                
                # Eliminar valores nulos
                if drop_na:
                    processed_df.dropna(inplace=True)
                
                # Normalización
                if normalize:
                    columns_to_normalize = [
                        'open', 'high', 'low', 'close', 'volume'
                    ]
                    processed_df[columns_to_normalize] = (
                        processed_df[columns_to_normalize] - 
                        processed_df[columns_to_normalize].mean()
                    ) / processed_df[columns_to_normalize].std()
                
                symbol_processed[timeframe] = processed_df
            
            self.processed_data[symbol] = symbol_processed
        
        return self.processed_data

# Ejemplo de uso
def main():
    # Crear instancia de ingesta de datos
    data_ingestion = MultiSymbolDataIngestion()
    
    # Obtener datos
    market_data = data_ingestion.concurrent_data_fetch()
    
    # Preprocesar datos
    preprocessor = DataPreprocessor(market_data)
    processed_data = preprocessor.preprocess_data(
        drop_na=True, 
        normalize=True
    )

if __name__ == "__main__":
    main()
