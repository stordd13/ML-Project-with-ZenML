from zenml import step
import logging
import pandas as pd

@step
def train_model(df: pd.DataFrame) -> None:
    pass
