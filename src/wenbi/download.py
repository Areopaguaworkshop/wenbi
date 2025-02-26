import os
import sys
import subprocess
import urllib.request

def check_and_download_spacy_model(model_name: str):
    try:
        __import__(model_name)
        print(f"{model_name} is already installed.")
    except ImportError:
        print(f"{model_name} not found. Downloading via spacy CLI...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

def check_and_download_fasttext_model(model_path: str, url: str):
    if os.path.exists(model_path):
        print(f"FastText model found at {model_path}.")
    else:
        print(f"FastText model not found at {model_path}. Downloading from {url} ...")
        urllib.request.urlretrieve(url, model_path)
        print("FastText model download complete.")

def download_all():
    # Check and download spaCy models
    check_and_download_spacy_model("zh_core_web_sm")
    check_and_download_spacy_model("en_core_web_sm")
    
    # Use the default FastText model location in the user home directory (~/.fasttext)
    default_dir = os.path.join(os.path.expanduser("~"), ".fasttext")
    os.makedirs(default_dir, exist_ok=True)
    fasttext_model_path = os.path.join(default_dir, "lid.176.bin")
    fasttext_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    check_and_download_fasttext_model(fasttext_model_path, fasttext_model_url)

if __name__ == "__main__":
    download_all()
