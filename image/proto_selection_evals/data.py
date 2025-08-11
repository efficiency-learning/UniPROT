import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import random
import os
import sys
import torch

# Add the parent python directory to the path to import SPOTgreedy and MMD-critic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMD-critic'))
from loader import get_tinyimagenet_loader
from features import load_feature_extractor, extract_features

def load_dataset(name):
    if name == 'Letter':
        from sklearn.datasets import fetch_openml
        letter = fetch_openml("letter", version=1)
        X, y = letter.data.to_numpy(), letter.target
        y = np.array([ord(c) - ord('A') for c in y])  # convert 'A'-'Z' to 0-25
        
        # Split into source and target pools
        from sklearn.model_selection import train_test_split
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, train_size=0.7, random_state=42, stratify=y
        )
        return (source_X, source_y), (target_pool_X, target_pool_y)

    elif name == 'Digits':
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Split into source and target pools
        from sklearn.model_selection import train_test_split
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, train_size=0.7, random_state=42, stratify=y
        )
        return (source_X, source_y), (target_pool_X, target_pool_y)

    elif name == 'Wine':
        from sklearn.datasets import load_wine
        wine = load_wine()
        X, y = wine.data, wine.target
        
        # Split into source and target pools
        from sklearn.model_selection import train_test_split
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, train_size=0.7, random_state=42, stratify=y
        )
        return (source_X, source_y), (target_pool_X, target_pool_y)

    elif name == 'mnist':
        try:
            from sklearn.datasets import fetch_openml
            print("Loading MNIST dataset...")
            mnist = fetch_openml('mnist_784', version=1)
            X, y = mnist.data.to_numpy(), mnist.target.astype(int)
            
            print(f"MNIST dataset: {len(X)} samples, {len(np.unique(y))} classes")
            print(f"Feature shape: {X.shape}")
            
            # Following the protocol: use standard MNIST train/test split
            # Standard MNIST has 60k train + 10k test
            # We'll use 70% for source, 30% for target pool (similar to train/test ratio)
            total_samples = len(X)
            source_size = int(0.7 * total_samples)  # 70% for source
            
            print(f"Planning to select {source_size} samples for source from {total_samples} total")
            
            # Stratified split to maintain class distribution
            from sklearn.model_selection import train_test_split
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=source_size, random_state=42, stratify=y
            )
            
            print(f"Source set: {len(source_X)} samples")
            print(f"Target pool: {len(target_pool_X)} samples")
            print(f"Classes in source: {np.unique(source_y)}")
            print(f"Classes in target pool: {np.unique(target_pool_y)}")
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
        except Exception as e:
            print(f"Error loading MNIST: {e}")
            # Fallback to sklearn digits if MNIST fails
            print("Falling back to sklearn digits dataset...")
            from sklearn.datasets import load_digits
            digits = load_digits()
            X, y = digits.data, digits.target
            
            # Split digits dataset similarly
            from sklearn.model_selection import train_test_split
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)


    elif name == 'Letter':
        from sklearn.datasets import fetch_openml
        print("Loading Letter dataset...")
        letter = fetch_openml("letter", version=1)
        X, y = letter.data.to_numpy(), letter.target
        y = np.array([ord(c) - ord('A') for c in y])  # convert 'A'-'Z' to 0-25
        
        # Following the protocol: 20K data points, sample 4K as source, rest for target
        print(f"Letter dataset: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Create source set (4K samples)
        from sklearn.model_selection import train_test_split
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, train_size=4000, random_state=42, stratify=y
        )
        
        return (source_X, source_y), (target_pool_X, target_pool_y)

    elif name == 'USPS':
        try:
            from sklearn.datasets import fetch_openml
            print("Loading USPS dataset...")
            usps = fetch_openml('usps', version=2)
            X, y = usps.data.to_numpy(), usps.target.astype(int)
            
            print(f"USPS dataset: {len(X)} samples, {len(np.unique(y))} classes")
            print(f"Index range: 0 to {len(X)-1}")
            
            # Following the protocol: source set 7291 points, target from remaining 2007
            # We'll split proportionally if we don't have exact numbers
            total_samples = len(X)
            source_size = min(7291, int(0.78 * total_samples))  # ~78% for source
            
            print(f"Planning to select {source_size} samples for source from {total_samples} total")
            
            source_indices = np.random.choice(len(X), source_size, replace=False)
            source_X, source_y = X[source_indices], y[source_indices]
            
            # Remaining for target construction
            remaining_mask = np.ones(len(X), dtype=bool)
            remaining_mask[source_indices] = False
            target_pool_X = X[remaining_mask]
            target_pool_y = y[remaining_mask]
            
            print(f"Source set: {len(source_X)} samples")
            print(f"Target pool: {len(target_pool_X)} samples")
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
        except Exception as e:
            print(f"Error loading USPS: {e}")
            return None, None

    elif name == 'ImageNet':
        raise NotImplementedError

    elif name == 'tinyimagenet':
        # For ImageNet, we'll use a substitute since the full dataset is massive
        # We'll simulate 2048-dimensional features as mentioned in the paper
        print("Loading tinyimagenet dataset...")
        processor, model = load_feature_extractor("microsoft/resnet-50")  # or e.g. "google/vit-base-patch16-224"

        # Load TinyImageNet train loader
        loader, dataset = get_tinyimagenet_loader(
            root="/home/ganesh/AAAI26/spot/SPOT/baselines/tiny/tiny-imagenet-200",
            image_processor=processor,
            batch_size=128,
            split="train",
            num_workers=4
        )

        features, labels = extract_features(loader, model, device="cuda" if torch.cuda.is_available() else "cpu")
        features = features.view(features.size(0), -1)

        X = features
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        y = labels
        
        print(f"TinyImagenet: {len(X)} samples")
        
        # Following protocol: source set is 50% of points, target from remaining 50%
        source_indices = np.random.choice(len(X), len(X)//2, replace=False)
        source_X, source_y = X[source_indices].numpy(), y[source_indices].numpy()
        
        remaining_mask = np.ones(len(X), dtype=bool)
        remaining_mask[source_indices] = False
        target_pool_X = X[remaining_mask].numpy()
        target_pool_y = y[remaining_mask].numpy()
        
        return (source_X, source_y), (target_pool_X, target_pool_y)
    elif name == 'Flickr':
        # Check if the flickr file exists, if not skip this dataset
        flickr_path = os.path.join(os.path.dirname(__file__), 'flickr_features.npz')
        if not os.path.exists(flickr_path):
            print(f"Warning: flickr_features.npz not found at {flickr_path}. Skipping Flickr dataset.")
            return None
        
        print("Loading Flickr dataset...")
        data = np.load(flickr_path, allow_pickle=True)
        X, y = data['X'], data['y']
        
        print(f"Flickr dataset: {len(X)} samples, features shape: {X.shape}")
        
        # Following the protocol: source 9836, target 9885 points
        # We'll split approximately if we don't have exact numbers
        total_samples = len(X)
        source_size = min(9836, total_samples // 2)
        
        source_indices = np.random.choice(len(X), source_size, replace=False)
        source_X, source_y = X[source_indices], y[source_indices]
        
        remaining_mask = np.ones(len(X), dtype=bool)
        remaining_mask[source_indices] = False
        target_pool_X = X[remaining_mask]
        target_pool_y = y[remaining_mask]
        
        return (source_X, source_y), (target_pool_X, target_pool_y)

    else:
        raise ValueError(f"Unknown dataset: {name}")