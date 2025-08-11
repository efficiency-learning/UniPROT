#!/usr/bin/env python3
"""
Download and prepare Flickr dataset for SPOT evaluation.
This script downloads a sample Flickr dataset and converts it to the expected format.
"""

import numpy as np
import requests
import os
import zipfile
import urllib.request
from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.image import extract_patches_2d
import random

def download_flickr_substitute():
    """
    Since the original Flickr dataset from the SPOT paper may not be publicly available,
    we'll create a substitute using a combination of available image datasets
    that can serve as a proxy for evaluation purposes.
    """
    print("Creating Flickr substitute dataset...")
    
    # Option 1: Use Olivetti faces as a substitute (40 classes, 400 samples)
    print("Loading Olivetti faces dataset as Flickr substitute...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X_faces = faces.data
    y_faces = faces.target
    
    # Add some noise and variations to make it more realistic
    np.random.seed(42)
    
    # Create multiple variations of each face
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X_faces)):
        # Original
        X_augmented.append(X_faces[i])
        y_augmented.append(y_faces[i])
        
        # Add noise variation
        noise = np.random.normal(0, 0.05, X_faces[i].shape)
        X_noisy = np.clip(X_faces[i] + noise, 0, 1)
        X_augmented.append(X_noisy)
        y_augmented.append(y_faces[i])
        
        # Add brightness variation
        brightness = np.random.uniform(0.8, 1.2)
        X_bright = np.clip(X_faces[i] * brightness, 0, 1)
        X_augmented.append(X_bright)
        y_augmented.append(y_faces[i])
    
    X_final = np.array(X_augmented)
    y_final = np.array(y_augmented)
    
    print(f"Created dataset with {len(X_final)} samples and {len(np.unique(y_final))} classes")
    
    return X_final, y_final

def download_real_flickr_dataset():
    """
    Attempt to download a real Flickr-based dataset.
    This tries to get a publicly available Flickr dataset.
    """
    print("Attempting to download real Flickr dataset...")
    
    # Try to download from a known source (this is a placeholder)
    # In practice, you would need the actual URL or dataset source
    urls = [
        # Add actual URLs here if available
        "https://example.com/flickr_features.npz",  # Placeholder
    ]
    
    for url in urls:
        try:
            print(f"Trying to download from {url}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open('flickr_features.npz', 'wb') as f:
                    f.write(response.content)
                print("Successfully downloaded Flickr dataset!")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    
    return False

def create_synthetic_flickr_dataset():
    """
    Create a synthetic dataset that mimics what a Flickr feature dataset might look like.
    """
    print("Creating synthetic Flickr-like dataset...")
    
    np.random.seed(42)
    
    # Create synthetic image features (e.g., as if extracted from CNN)
    n_samples = 2000
    n_features = 512  # Common CNN feature dimension
    n_classes = 20    # Number of different "scenes" or "categories"
    
    # Generate cluster centers for each class
    cluster_centers = np.random.randn(n_classes, n_features) * 2
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Generate sample around cluster center with some noise
            sample = cluster_centers[class_idx] + np.random.randn(n_features) * 0.5
            X.append(sample)
            y.append(class_idx)
    
    # Add some additional random samples
    remaining = n_samples - len(X)
    for _ in range(remaining):
        class_idx = np.random.randint(0, n_classes)
        sample = cluster_centers[class_idx] + np.random.randn(n_features) * 0.5
        X.append(sample)
        y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Created synthetic dataset with {len(X)} samples, {n_features} features, and {n_classes} classes")
    
    return X, y

def main():
    print("Flickr Dataset Downloader for SPOT Evaluation")
    print("=" * 50)
    
    # Check if file already exists
    if os.path.exists('flickr_features.npz'):
        print("flickr_features.npz already exists!")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Try different approaches in order of preference
    success = False
    
    # 1. Try to download real dataset
    if download_real_flickr_dataset():
        success = True
    
    if not success:
        print("\nReal Flickr dataset not available. Creating substitute...")
        
        # Ask user preference
        print("\nChoose dataset type:")
        print("1. Face-based substitute (Olivetti faces)")
        print("2. Synthetic CNN-like features")
        
        choice = input("Enter choice (1 or 2, default=2): ").strip()
        
        if choice == "1":
            X, y = download_flickr_substitute()
        else:
            X, y = create_synthetic_flickr_dataset()
        
        # Save to npz file
        print(f"\nSaving dataset to flickr_features.npz...")
        np.savez('flickr_features.npz', X=X, y=y)
        success = True
    
    if success:
        print("\n✓ Flickr dataset is now available!")
        
        # Verify the file
        data = np.load('flickr_features.npz', allow_pickle=True)
        X, y = data['X'], data['y']
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Feature dimension: {X.shape[1]}")
    else:
        print("\n✗ Failed to create Flickr dataset")

if __name__ == "__main__":
    main()
