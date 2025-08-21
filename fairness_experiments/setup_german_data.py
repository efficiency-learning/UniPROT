import os
import urllib.request
from pathlib import Path

def setup_german_credit_dataset():
    """Download and setup German Credit dataset for AIF360"""
    
    # Define paths
    base_path = Path.home() / '.local/lib/python3.10/site-packages/aif360/data/raw/german'
    base_path.mkdir(parents=True, exist_ok=True)
    
    # URLs for the dataset files
    urls = {
        'german.data': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
        'german.doc': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc'
    }
    
    # Download files
    for filename, url in urls.items():
        filepath = base_path / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Saved to {filepath}")
        else:
            print(f"{filename} already exists at {filepath}")

if __name__ == "__main__":
    setup_german_credit_dataset()
