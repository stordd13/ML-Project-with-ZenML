import logging
from zenml import step
import numpy as np
import pandas as pd
from typing import Tuple, Annotated
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin

@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame) -> Tuple[
        Annotated[float, 'mse'],
        Annotated[float, "r2"],
        Annotated[float, "rmse"]
    ]:
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
    
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)

        return mse, r2, rmse
        
    except Exception as e:
        logging.error(f"Error in calculating scores : {e}")
        raise e
