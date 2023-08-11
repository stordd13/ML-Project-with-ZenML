import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df:pd.DataFrame) ->  Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:

    """ 
    Cleans the data and divides it into train and test
    Arg:
        df: Raw data
    Returns:
        X_train: training data
        X_test: Testing data
        y_train: training labels
        y_test: testing labels
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")

    except Exception as e:
        logging.info(f"Error in cleaning data : {e}")
        raise e