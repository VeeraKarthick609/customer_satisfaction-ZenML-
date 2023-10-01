import logging as lg
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining stratergy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStartergy(DataStrategy):
    """
    Stratergy for processing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(columns=cols_to_drop, axis=1)
            return data
        except Exception as e:
            lg.error(f"Preprocessing error: {e}")
            raise e


class DataDivideStratergy(DataStrategy):
    """
    Stratergy for dividing data into train and test split
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """

        try:
            """
            review_score is the target variable
            """
            x = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, Y_train, Y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            lg.error(f"Error while train test split: {e}")
            raise e


class DataCleaning:
    """
    Process the data and divide the data into train and test split
    """

    def __init__(self, data: pd.DataFrame, stratergy: DataStrategy):
        self.data = data
        self.stratergy = stratergy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle Data
        """

        try:
            return self.stratergy.handle_data(self.data)
        except Exception as e:
            lg.error(e)
            raise e

