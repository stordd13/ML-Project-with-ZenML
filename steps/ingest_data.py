import logging
import pandas as pd
from zenml import step


class IngestData:
    """ 
    Ingesting the data from the data path
    """
    def __init__(self, data_path:str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingesting the data from the data path

    Args: 
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data 
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        print(df.shape)
        return df
    except Exception as e:
        logging.error(f"Error while ingestion data : {e}")
        raise e
    