import subprocess
import sys
import os


def set_tokenizers_parallelism(value):
    """Set the TOKENIZERS_PARALLELISM environment variable."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'true' if value else 'false'
    print(f"TOKENIZERS_PARALLELISM set to {os.environ['TOKENIZERS_PARALLELISM']}")


def install_requirements():
    """Install packages listed in requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages from requirements.txt installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages from requirements.txt: {e}")
        sys.exit(1)


def install_spacy_model(model_name):
    """Install a specific spaCy model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"spaCy model '{model_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install spaCy model '{model_name}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_requirements()
    install_spacy_model("de_core_news_lg")
    set_tokenizers_parallelism(True)
