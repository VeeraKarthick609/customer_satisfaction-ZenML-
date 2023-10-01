import logging
import pandas as pd
from zenml import step
import mlflow

from src.model_dev import LinearRegressionModel
from .config import ModuleNameConfig

from sklearn.base import RegressorMixin

from zenml.client  import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    config: ModuleNameConfig,
) -> RegressorMixin:
    """
    Trains the  model on the ingested data

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame

    Returns:
        RegressorMixin
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train=X_train, Y_train=Y_train)
            return trained_model
        else:
            raise ValueError(f"Mode {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e