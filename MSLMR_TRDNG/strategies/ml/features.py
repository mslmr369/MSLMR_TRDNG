# strategies/ml/features.py
import pandas as pd
from typing import List

class FeatureExtractor:
    """
    Extracts features from market data for use in machine learning models.
    """

    def __init__(self, features: List[str]):
        """
        Initializes the feature extractor.

        :param features: A list of feature names to be extracted.
        """
        self.features = features

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the specified features from the input data.

        :param data: DataFrame containing market data.
        :return: DataFrame with extracted features.
        """
        extracted_features = pd.DataFrame()

        if 'rsi' in self.features:
            # Assume a function calculate_rsi exists (can be from ta library or custom)
            extracted_features['rsi'] = self.calculate_rsi(data['close'])

        if 'macd' in self.features:
            # Assume a function calculate_macd exists
            macd_line, signal_line, _ = self.calculate_macd(data['close'])
            extracted_features['macd_line'] = macd_line
            extracted_features['macd_signal'] = signal_line

        # Add more feature extraction logic here as needed

        return extracted_features

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI) for a given series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        exp1 = series.ewm(span=fast_period, adjust=False).mean()
        exp2 = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

