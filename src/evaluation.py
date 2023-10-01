import logging 
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining stratergy for evaluation of models
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred: np.ndarray):
        """
        Calculates the score of the model

        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            None
        """

        pass

class MSE(Evaluation):
    """
    Evaluation Stratergy that uses Mean Sqaured Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation stratergy that uses R2 score
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_pred=y_pred, y_true=y_true)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation stratergy that uses Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RSME")
            rsme = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)
            logging.info(f"RSME: {rsme}")
            return rsme
        except Exception as e:
            logging.error("Error while calculaitng RSME: {}".format(e))
            raise e
        
    
