from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union, Tuple
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def train(self, data: pd.DataFrame, target: str, **kwargs):
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Makes predictions using the trained model.
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Saves the model to disk.
        """
        pass

    @abstractmethod
    def load(self, filepath: str):
        """
        Loads a saved model from disk.
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, target: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the model's performance.
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Gets the model's hyperparameters.
        """
        pass
    
    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        """
        Sets the model's hyperparameters.
        """
        pass

