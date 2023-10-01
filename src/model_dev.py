import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, Y_train):
        """
        Trains the model

        Args:
            X_train: training data
            Y_train: training labels

        Returns:
            None

        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, X_train, Y_train, **kwargs):
        """
        Trains the model

        Args:
            X_train: Training data
            Y_train: Training labels

        Returns: 
            None
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X=X_train, y=Y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error while trainig the model: {e}")
            raise e