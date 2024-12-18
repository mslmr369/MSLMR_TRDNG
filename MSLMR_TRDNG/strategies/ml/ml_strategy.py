# strategies/ml/ml_strategy.py
from typing import Dict, Any, List, Optional
import pandas as pd

from core.logging_system import LoggingMixin
from strategies.base import BaseStrategy
from .models import BaseModel
from .features import FeatureExtractor

class MLStrategy(BaseStrategy, LoggingMixin):
    """
    Base class for trading strategies using machine learning models.
    """

    def __init__(
        self,
        model: BaseModel,
        feature_extractor: FeatureExtractor,
        name: str = "MLStrategy",
        description: str = "Trading strategy using a machine learning model",
        timeframes: List[str] = None
    ):
        """
        Initializes the ML strategy.

        :param model: The machine learning model to use.
        :param feature_extractor: The feature extractor to use.
        :param name: The name of the strategy.
        :param description: A brief description of the strategy.
        :param timeframes: The timeframes on which the strategy should operate.
        """
        super().__init__(name, description, timeframes)
        self.model = model
        self.feature_extractor = feature_extractor

    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates trading signals based on market data and ML predictions.

        :param data: DataFrame with market data.
        :return: Dictionary with trading signals or None.
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(data)

            # Make prediction
            prediction = self.model.predict(features)

            # Interpret prediction and generate signal
            signal = self.interpret_prediction(prediction)
            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal with {self.name}: {e}")
            return None

    def interpret_prediction(self, prediction: Any) -> Optional[Dict]:
        """
        Interprets the model's prediction and generates a trading signal.

        :param prediction: The prediction made by the ML model.
        :return: A trading signal or None.
        """
        # Placeholder for converting model output to a trading signal
        # This will depend on the specifics of your ML model and strategy logic
        raise NotImplementedError("Subclasses must implement this method.")

    def get_required_indicators(self) -> List[str]:
        """
        Returns a list of indicators required by this strategy.
        For ML strategies, this might be an empty list or determined by the feature extractor.
        """
        return self.feature_extractor.features
