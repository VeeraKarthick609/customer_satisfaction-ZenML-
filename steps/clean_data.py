import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

from src.data_cleaning import DataCleaning, DataDivideStratergy, DataPreprocessStartergy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"]
]:
    
    """
    Cleans the data and divides into train and test

    Args:
        df -> Raw Data
    
    Returns:
        X_train: Training data
        X_test:  Testing data
        Y_train: Training labels
        Y_test:  Testing labels
    """

    try:
        process_stratergy = DataPreprocessStartergy()
        data_cleaning = DataCleaning(df, stratergy=process_stratergy)
        processed_data = data_cleaning.handle_data()

        divide_stratergy = DataDivideStratergy()
        data_divide = DataCleaning(processed_data,stratergy=divide_stratergy)
        X_train, X_test, Y_train, Y_test = data_divide.handle_data()
        logging.info("Data cleaning complleted")
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error(f"Error while cleaning the data: {e}")
        raise e
    
        
