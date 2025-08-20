import os
import urllib.request
from pathlib import Path

def setup_dataset(dataset_name):
    """Download and setup dataset files"""
    base_path = Path.home() / '.local/lib/python3.10/site-packages/aif360/data/raw'
    
    datasets = {
        'adult': {
            'path': base_path / 'adult',
            'files': {
                'adult.data': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                'adult.test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                'adult.names': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
            }
        },
        'german': {
            'path': base_path / 'german',
            'files': {
                'german.data': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                'german.doc': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc'
            }
        },
        'compas': {
            'path': base_path / 'compas',
            'files': {
                'compas-scores-two-years.csv': 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
            }
        }
    }
    
    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not configured for download")
        return
        
    dataset = datasets[dataset_name]
    dataset['path'].mkdir(parents=True, exist_ok=True)
    
    for filename, url in dataset['files'].items():
        filepath = dataset['path'] / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists at {filepath}")

def setup_all_datasets():
    """Setup all required datasets"""
    for dataset in ['adult', 'german', 'compas']:
        print(f"\nSetting up {dataset} dataset...")
        setup_dataset(dataset)

if __name__ == "__main__":
    setup_all_datasets()
