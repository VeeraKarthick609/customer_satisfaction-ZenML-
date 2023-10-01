import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
import mlflow
from zenml import step
from zenml.client import Client

from src.evaluation import MSE, RMSE, R2 

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, Y_test: pd.DataFrame
) -> Tuple[Annotated[float, "mse"], Annotated[float, "rmse"], Annotated[float, "r2"]]:
    """
    Evaluates the model on the ingested data
    Args:
        model: Regression model
        X_test: Test input dataframe
        Y_test: Test output dataframe

    Returns:
        Tuple:
            mse  -> Mean sqaured error
            rmse -> Root Mean squared error
            r2   -> R2 score

    """
    try:
        y_pred = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_pred=y_pred, y_true=Y_test)
        mlflow.log_metric("mse",mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_pred=y_pred, y_true=Y_test)
        mlflow.log_metric("r2",r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_pred=y_pred, y_true=Y_test)
        mlflow.log_metric("rmse",rmse)

        return mse, rmse, r2
    except Exception as e:
        logging.error("Error while calculating score: {}".format(e))
        raise e

