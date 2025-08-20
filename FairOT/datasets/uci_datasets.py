import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_communities_crime(root_dir=None, split_ratio=0.7, random_state=42):
    """Load Communities and Crime dataset from UCI ML Repository."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading Communities and Crime dataset...")
        
        # Fetch dataset
        crime = fetch_openml("communities-and-crime", version=1, as_frame=True, parser='auto')
        if crime is None or crime.data is None or crime.target is None:
            print("Error: Failed to load Communities and Crime dataset")
            return None
        
        # Select numeric features and handle missing values
        X = crime.data.select_dtypes(include=[np.number])
        if len(X.columns) == 0:
            print("Error: No numeric features found")
            return None
            
        X = X.fillna(X.mean())
        
        # Convert target to numeric and handle missing values
        try:
            y = pd.to_numeric(crime.target, errors='coerce')
            if y.isna().any():
                print("Warning: Missing values in target, using mean imputation")
                y = y.fillna(y.mean())
        except Exception as e:
            print(f"Error converting target to numeric: {e}")
            return None
            
        # Convert to binary based on median
        median_crime_rate = np.median(y)
        y = (y > median_crime_rate).astype(int)
        
        # Print dataset statistics
        print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
        print(f"Class distribution:")
        print(f"- Low crime rate (0): {np.sum(y == 0)}")
        print(f"- High crime rate (1): {np.sum(y == 1)}")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=split_ratio,
            random_state=random_state,
            stratify=y
        )
        
        return (X_train, y_train), (X_test, y_test)
        
    except Exception as e:
        print(f"Error loading Communities and Crime dataset: {e}")
        return None

def get_feature_names():
    """Return list of feature names for the Communities and Crime dataset"""
    try:
        from sklearn.datasets import fetch_openml
        crime = fetch_openml("communities-and-crime", version=1, as_frame=True)
        numeric_features = crime.data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_features
    except:
        return None
