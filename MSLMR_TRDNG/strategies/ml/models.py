# strategies/ml/models.py
from typing import List
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Makes predictions on the given data.

        :param X: Input data for prediction.
        :return: Model predictions.
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Saves the model to a file.

        :param filepath: Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, filepath: str):
        """
        Loads the model from a file.

        :param filepath: Path to load the model from.
        """
        pass
