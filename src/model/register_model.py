# register model

import json
import mlflow
import logging
import os
import pickle
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from mlflow.models import infer_signature

load_dotenv()

def _configure_mlflow():
    """
    Use DagsHub if creds exist; else fallback to local.
    DagsHub expects username=<dagshub username>, password=<token>.
    """
    dagshub_username = os.getenv("DAGSHUB_USERNAME", "sohamfcb")  # <-- your username
    dagshub_token = os.getenv("DAGSHUB_PAT")

    dagshub_uri = os.getenv("MLFLOW_URL")

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username   # <-- username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token      # <-- token
        mlflow.set_tracking_uri(dagshub_uri)
    else:
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def register_model(model_name: str, model_path: str, vectorizer_path: str):
    """Register the model to the MLflow Model Registry."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Start a new run to log and register the model
        with mlflow.start_run() as run:
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load the vectorizer to create example data
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Create a dummy input example as a DataFrame with proper feature names
            feature_names = vectorizer.get_feature_names_out()
            input_array = np.zeros((1, len(feature_names)))
            input_example = pd.DataFrame(input_array, columns=feature_names)
            
            # Get output example
            output_example = model.predict(input_example)
            
            # Infer signature
            signature = infer_signature(input_example, output_example)
            
            # Log the model with signature
            # mlflow.lightgbm.log_model(
            #     model, 
            #     "lgbm_model",
            #     signature=signature,
            #     input_example=input_example
            # )

            mlflow.lightgbm.log_model(
                model,
                artifact_path="lgbm_model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )

            
            # # Register the model
            # model_uri = f"runs:/{run.info.run_id}/lgbm_model"
            # model_version = mlflow.register_model(model_uri, model_name)
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

            # Transition the model to "Staging" stage
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
            
            logger.debug(f'Model {model_name} version {latest_version.version} registered and transitioned to Staging.')
            print(f'Model {model_name} version {latest_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    _configure_mlflow()
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        model_name = "yt_chrome_plugin_model"
        model_path = os.path.join(root_dir, 'lgbm_model.pkl')
        vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
        
        register_model(model_name, model_path, vectorizer_path)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()