from pipeline.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/Users/brunostordeur/Docs/GitHub/ML-Project-with-ZenML/data/olist_customers_dataset.csv")
    #mlflow ui --backend-store-uri "file:/Users/brunostordeur/Library/Application Support/zenml/local_stores/fbadd7e2-7343-44c2-8c41-72e9e362de4e/mlruns"