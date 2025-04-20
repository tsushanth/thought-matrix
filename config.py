import os
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

# Path to your credentials file (should be in your project root)
GCP_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials.json')

class Config:
    GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
    N_PARTITIONS = int(os.getenv("N_PARTITIONS", 3))
    HF_TOKEN = os.getenv("HF_TOKEN")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    
    @property
    def gcp_auth(self):
        return {
            'credentials': service_account.Credentials.from_service_account_file(
                GCP_CREDENTIALS_PATH
            ) if os.path.exists(GCP_CREDENTIALS_PATH) else None
        }