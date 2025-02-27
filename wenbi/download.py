import os
import sys
import subprocess

def check_and_download_spacy_model(model_name: str):
    """Download spaCy language model if not already installed."""
    try:
        __import__(model_name)
        print(f"{model_name} is already installed.")
    except ImportError:
        print(f"{model_name} not found. Downloading via spacy CLI...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

def download_all():
    """Download all required models."""
    check_and_download_spacy_model("zh_core_web_sm")
    check_and_download_spacy_model("en_core_web_sm")

if __name__ == "__main__":
    download_all()
