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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize

# Add the parent python directory to the path to import SPOTgreedy and MMD-critic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMD-critic'))
from loader import get_tinyimagenet_loader
from features import load_feature_extractor, extract_features

def plot_dataset_distribution(X, y, dataset_name):
    """Plot distribution of features and classes for a dataset."""
    # Create output directory
    plot_dir = os.path.join(os.path.dirname(__file__), 'dataset_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Class distribution
    unique_classes = np.unique(y)
    class_counts = [np.sum(y == cls) for cls in unique_classes]
    ax1.bar(unique_classes, class_counts)
    ax1.set_title(f'{dataset_name} Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    for i, count in enumerate(class_counts):
        ax1.text(unique_classes[i], count, str(count), ha='center', va='bottom')
    
    # Plot 2: Feature box plot
    ax2.boxplot(X)
    ax2.set_title(f'{dataset_name} Feature Distribution')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{dataset_name.lower()}_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved dataset distribution plot: {plot_path}")

def load_dataset(name):
    """Load datasets with proper preprocessing:
    - Remove protected attributes
    - Convert categorical to one-hot
    - Normalize columns to unit L2-norm
    - Handle group membership (sex/race)
    """
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

    elif name == 'adult':
        # UCI Adult dataset
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split  
        print("Loading Adult dataset...")
        try:
            adult = fetch_openml("adult", version=2, as_frame=True, parser='auto')
            if adult is None or adult.data is None:
                print("Error: Failed to load Adult dataset")
                return None
                
            # Handle categorical variables - select only numeric columns
            X = adult.data.select_dtypes(include=[np.number])
            if len(X.columns) == 0:
                print("Error: No numeric features found in Adult dataset")
                return None
                
            # Convert target to binary ('>50K' as 1, '<=50K' as 0)
            y = adult.target
            if y is None:
                print("Error: Target variable is None")
                return None
                
            y = (y == '>50K').astype(int)
            
            # Add visualization
            plot_dataset_distribution(X.to_numpy(), y, 'Adult')
            
            # Split into source and target pools
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
        except Exception as e:
            print(f"Error loading Adult dataset: {e}")
            return None

    elif name == 'crime':
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split
            print("Loading Crime dataset...")
            
            crime = fetch_openml("communities-and-crime", version=1, as_frame=True)
            if crime is None or crime.data is None:
                print("Error: Failed to load Crime dataset")
                return None
                
            # Handle numeric features
            X = crime.data.select_dtypes(include=[np.number])
            if len(X.columns) == 0:
                print("Error: No numeric features found in Crime dataset")
                return None
            
            X = X.to_numpy()
            
            # Handle target variable
            y = crime.target
            if y is None:
                print("Error: Target variable is None")
                return None
            
            # Convert target to binary based on the median value
            target_values = y.astype(float)
            median_value = np.median(target_values)
            y = (target_values > median_value).astype(int)
            print(f"Crime dataset: Converting to binary classification using median {median_value:.3f}")
            print(f"Class distribution - 0: {np.sum(y==0)}, 1: {np.sum(y==1)}")
            
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            print(f"Crime dataset loaded - Features shape: {X.shape}, Classes: {np.unique(y)}")
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Crime dataset: {e}")
            return None

    elif name == 'drug':
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        print("Loading Illicit Drugs dataset...")
        try:
            # Use "illicit-drugs" dataset instead
            drug = fetch_openml("illicit-drugs", version=1, as_frame=True, parser='auto')
            if drug is None or drug.data is None:
                print("Error: Failed to load Illicit Drugs dataset")
                return None
                
            # Handle numeric features
            X = drug.data.select_dtypes(include=[np.number])
            if len(X.columns) == 0:
                print("Error: No numeric features found in Illicit Drugs dataset")
                return None
            
            print(f"Available features: {X.columns.tolist()}")
            X = X.to_numpy()
            
            # Handle target variable - binary classification based on usage frequency
            y = drug.target
            if y is None:
                print("Error: Target variable is None")
                return None
            
            # Convert to binary: Never=0, Any usage=1
            y = (y != "Never Used").astype(int)
            print(f"Illicit Drugs dataset: Converting to binary classification (Never Used vs Any Usage)")
            print(f"Class distribution - Never Used (0): {np.sum(y==0)}, Used (1): {np.sum(y==1)}")
            
            # Add visualization
            plot_dataset_distribution(X, y, 'Drug')
            
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            print(f"Illicit Drugs dataset loaded - Features shape: {X.shape}, Classes: {np.unique(y)}")
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Illicit Drugs dataset: {e}")
            # Helpful debug info
            print("Available OpenML datasets can be found at: https://www.openml.org/search?type=data")
            return None

    elif name == 'credit':
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split  # <-- Add this import
        print("Loading Credit dataset...")
        credit = fetch_openml("credit-g", version=1, as_frame=True)
        X = credit.data.select_dtypes(include=[np.number]).to_numpy()
        y = credit.target
        y = (y == 'good').astype(int)
        
        # Add visualization
        plot_dataset_distribution(X, y, 'Credit')
        
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, train_size=0.7, random_state=42, stratify=y
        )
        return (source_X, source_y), (target_pool_X, target_pool_y)

    elif name == 'recidivism':
        print("Loading Juvenile Recidivism dataset...")
        try:
            # Assuming dataset is stored in a CSV
            data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/recidivism.csv'))
            protected_attr = ['sex']  # Remove protected attributes
            X = data.drop(protected_attr + ['target'], axis=1)
            y = data['target']
            sex = data['sex']  # Store for group membership
            
            # Convert categorical variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_cat = enc.fit_transform(X[cat_cols])
                X_num = X.select_dtypes(include=[np.number])
                X = np.hstack([X_num, X_cat])
            
            # Normalize columns
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Recidivism dataset: {e}")
            return None

    elif name == 'meps':
        print("Loading Medical Expenditure Survey 2015 dataset...")
        try:
            data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/meps_2015.csv'))
            protected_attr = ['sex']
            X = data.drop(protected_attr + ['target'], axis=1)
            y = data['target']
            sex = data['sex']
            
            # Convert categorical variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_cat = enc.fit_transform(X[cat_cols])
                X_num = X.select_dtypes(include=[np.number])
                X = np.hstack([X_num, X_cat])
            
            # Normalize columns
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading MEPS dataset: {e}")
            return None

    elif name == 'heart':
        print("Loading Heart Cleveland dataset...")
        try:
            data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/heart_cleveland.csv'))
            protected_attr = ['sex']
            X = data.drop(protected_attr + ['target'], axis=1)
            y = data['target']
            sex = data['sex']
            
            # Convert categorical variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_cat = enc.fit_transform(X[cat_cols])
                X_num = X.select_dtypes(include=[np.number])
                X = np.hstack([X_num, X_cat])
            
            # Normalize columns
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Heart dataset: {e}")
            return None

    elif name == 'communities':
        print("Loading Communities dataset...")
        try:
            crime = fetch_openml("communities-and-crime", version=1, as_frame=True)
            X = crime.data.select_dtypes(include=[np.number])
            y = (crime.target.astype(float) > np.median(crime.target.astype(float))).astype(int)
            
            # Group membership based on majority white vs non-white community
            race_cols = [col for col in X.columns if 'race' in col.lower()]
            white_pct = X[race_cols[0]] if race_cols else None  # Assuming first race column is white %
            group = (white_pct > 0.5).astype(int) if white_pct is not None else None
            
            # Remove race-related columns and normalize
            X = X[[col for col in X.columns if 'race' not in col.lower()]]
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Communities dataset: {e}")
            return None

    elif name == 'compas':
        print("Loading COMPAS Recidivism dataset...")
        try:
            data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/compas.csv'))
            protected_attr = ['sex']
            X = data.drop(protected_attr + ['target'], axis=1)
            y = data['target']
            sex = data['sex']
            
            # Convert categorical variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_cat = enc.fit_transform(X[cat_cols])
                X_num = X.select_dtypes(include=[np.number])
                X = np.hstack([X_num, X_cat])
            
            # Normalize columns
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading COMPAS dataset: {e}")
            return None

    elif name == 'student':
        print("Loading Student Performance dataset...")
        try:
            data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/student.csv'))
            protected_attr = ['sex']
            X = data.drop(protected_attr + ['target'], axis=1)
            y = data['target']
            sex = data['sex']
            
            # Convert categorical variables
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_cat = enc.fit_transform(X[cat_cols])
                X_num = X.select_dtypes(include=[np.number])
                X = np.hstack([X_num, X_cat])
            
            # Normalize columns
            X = normalize(X, norm='l2', axis=0)
            
            # Split data
            source_X, target_pool_X, source_y, target_pool_y = train_test_split(
                X, y, train_size=0.7, random_state=42, stratify=y
            )
            
            return (source_X, source_y), (target_pool_X, target_pool_y)
            
        except Exception as e:
            print(f"Error loading Student dataset: {e}")
            return None

    else:
        raise ValueError(f"Unknown dataset: {name}")