import os
from dotenv import load_dotenv
import mlflow

def _configure_mlflow():
    """
    Use DagsHub if creds exist; else fallback to local.
    DagsHub expects username=<dagshub username>, password=<token>.
    """
    load_dotenv()
    dagshub_username = os.getenv("DAGSHUB_USERNAME", "sohamfcb")  # <-- your username
    dagshub_token = os.getenv("DAGSHUB_PAT")

    # dagshub_url = "https://dagshub.com"
    # repo_owner = "sohamfcb"
    # repo_name = "mlops-mini-project"
    # dagshub_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

    dagshub_uri = os.getenv("MLFLOW_URL")


    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username   # <-- username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token      # <-- token
        mlflow.set_tracking_uri(dagshub_uri)
    else:
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
