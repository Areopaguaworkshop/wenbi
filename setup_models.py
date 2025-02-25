import os
import subprocess
import shutil
import sys

def setup_spacy_models():
    """Download and setup spaCy models in the project's model directory"""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    
    models = ["zh_core_web_sm", "en_core_web_sm"]
    for model in models:
        try:
            print(f"\nInstalling {model}...")
            # Download and install the model using spacy download
            subprocess.check_call([
                sys.executable,
                "-m",
                "spacy",
                "download",
                model
            ])
            
            # Use spacy validate to get the model path
            result = subprocess.check_output([
                sys.executable,
                "-m",
                "spacy",
                "validate",
                model
            ], stderr=subprocess.STDOUT)
            
            # Get the model directory from site-packages
            site_packages = subprocess.check_output([
                sys.executable,
                "-c",
                "import site; print(site.getsitepackages()[0])"
            ]).decode('utf-8').strip()
            
            model_path = os.path.join(site_packages, model)
            target_path = os.path.join(model_dir, model)
            
            # Copy the model files
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(model_path, target_path)
            
            # Copy the model-specific config.cfg
            cfg_src = os.path.join(site_packages, model, "config.cfg")
            cfg_dst = os.path.join(target_path, "config.cfg")
            shutil.copy2(cfg_src, cfg_dst)
            
            print(f"Successfully installed {model} to {target_path}")
            
        except Exception as e:
            print(f"Error installing {model}: {e}")
            print("Try installing manually with:")
            print(f"python -m spacy download {model}")
            print(f"Then copy from site-packages/{model} to {os.path.join(model_dir, model)}")

if __name__ == "__main__":
    setup_spacy_models()
