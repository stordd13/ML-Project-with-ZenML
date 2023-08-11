import logging
from zenml import step
import numpy as np
import pandas as pd

@step
def evaluate_model(df: pd.DataFrame) -> None:
    pass