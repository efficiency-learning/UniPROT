import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import random
import os
import sys
import json
import matplotlib.pyplot as plt

# Add the parent python directory to the path to import SPOTgreedy and MMD-critic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMDcritic'))
from baselines.MMDcritic.mmd_critic import Dataset, select_prototypes
from SPOTgreedy import SPOT_GreedySubsetSelection
from image.proto_selection_evals import data
from UniPROT.UniPROT import UniPROT
from baselines import evaluation

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def identify_minority_classes(y, minority_threshold=0.1):
    """
    Identify minority classes as classes with population below the threshold.
    
    Args:
        y: Labels array
        minority_threshold: Threshold for minority classes (default 0.1 for 10% of total population)
    
    Returns:
        minority_classes: List of classes that have population below the threshold
    """
    classes = np.unique(y)
    class_counts = {}
    total_population = len(y)
    
    # Count samples per class
    for cls in classes:
        class_counts[cls] = np.sum(y == cls)
    
    # Calculate threshold based on total population
    population_threshold = total_population * minority_threshold
    
    # Identify classes below the population threshold
    minority_classes = []
    for cls, count in class_counts.items():
        if count < population_threshold:
            minority_classes.append(cls)
    
    # Sort by population size (ascending) to get the smallest classes first
    minority_classes = sorted(minority_classes, key=lambda x: class_counts[x])
    
    print(f"Identified {len(minority_classes)} minority classes out of {len(classes)} total classes:")
    print(f"Population threshold: {population_threshold:.1f} samples ({minority_threshold*100:.1f}% of {total_population} total)")
    for cls in minority_classes:
        percentage = (class_counts[cls] / total_population) * 100
        print(f"  Class {cls}: {class_counts[cls]} samples ({percentage:.1f}%)")
    
    # If no classes are below threshold, select the smallest class as minority
    if len(minority_classes) == 0:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        minority_classes = [sorted_classes[0][0]]
        print(f"No classes below {minority_threshold*100:.1f}% threshold, selecting smallest class {minority_classes[0]} as minority")
    
    return minority_classes

def generate_ablation_target_set_other(pool_X, pool_y, minority_classes=None, total_size=2000):
    """
    Generate ablation target set for non-MNIST datasets:
    - minority_classes: 1% of target set split among all minority classes
    - other classes: remaining samples split evenly

    Args:
        pool_X: Pool of target features
        pool_y: Pool of target labels
        minority_classes: List of classes to be assigned 1% of the target set (if None, auto-identify)
        total_size: Total size of target set

    Returns:
        target_X, target_y: Generated target set
    """
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values

    # Auto-identify minority classes if not provided
    if minority_classes is None:
        minority_classes = identify_minority_classes(pool_y)
    elif not isinstance(minority_classes, list):
        minority_classes = [minority_classes]

    classes = np.unique(pool_y)
    target_indices = []

    # Minority classes: 1% total split among all minority classes
    minority_samples_total = int(total_size * 0.01)
    minority_samples_per_class = max(1, minority_samples_total // len(minority_classes))
    
    for minority_class in minority_classes:
        min_idx = np.where(pool_y == minority_class)[0]
        if len(min_idx) >= minority_samples_per_class:
            selected_min = np.random.choice(min_idx, size=minority_samples_per_class, replace=False)
            target_indices.extend(selected_min)
        else:
            target_indices.extend(min_idx)
            print(f"Warning: Only {len(min_idx)} samples available for minority class {minority_class}, needed {minority_samples_per_class}")

    # Majority classes: evenly split remaining samples
    majority_classes = [cls for cls in classes if cls not in minority_classes]
    if len(majority_classes) > 0:
        majority_samples_per_class = int((total_size - len(target_indices)) / len(majority_classes))
        for cls in majority_classes:
            cls_idx = np.where(pool_y == cls)[0]
            if len(cls_idx) >= majority_samples_per_class:
                selected_cls = np.random.choice(cls_idx, size=majority_samples_per_class, replace=False)
                target_indices.extend(selected_cls)
            else:
                target_indices.extend(cls_idx)
                print(f"Warning: Only {len(cls_idx)} samples available for class {cls}, needed {majority_samples_per_class}")

    target_indices = np.array(target_indices)
    np.random.shuffle(target_indices)

    if len(target_indices) == 0:
        raise ValueError("No valid target indices generated for ablation")

    # Print actual distribution
    final_target_y = pool_y[target_indices]
    print(f"Generated ablation target set: {len(target_indices)} samples")
    
    # Show minority vs majority distribution
    minority_count = sum(np.sum(final_target_y == cls) for cls in minority_classes)
    minority_percentage = (minority_count / len(target_indices)) * 100
    print(f"  Minority classes ({minority_classes}): {minority_count} samples ({minority_percentage:.1f}%)")
    
    for cls in classes:
        count = np.sum(final_target_y == cls)
        percentage = (count / len(target_indices)) * 100
        class_type = "minority" if cls in minority_classes else "majority"
        print(f"    Class {cls} ({class_type}): {count} samples ({percentage:.1f}%)")

    return pool_X[target_indices], pool_y[target_indices]

def plot_source_target_histograms(source_y, target_y, dataset_name, experiment_type='ablation'):
    """
    Plot histograms showing class distribution in source and target sets.
    
    Args:
        source_y: Source labels
        target_y: Target labels  
        dataset_name: Name of the dataset
        experiment_type: Type of experiment (for file naming)
    """
    classes = np.unique(np.concatenate([source_y, target_y]))
    
    # Identify minority classes in target set
    minority_classes = identify_minority_classes(target_y)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Source set histogram
    source_counts = [np.sum(source_y == cls) for cls in classes]
    ax1.bar(classes, source_counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.set_title(f'{dataset_name} Source Set Distribution\n(Balanced: ~{np.mean(source_counts):.0f} samples per class)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(source_counts):
        ax1.text(classes[i], count + max(source_counts)*0.01, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    # Target set histogram with minority classes highlighted
    target_counts = [np.sum(target_y == cls) for cls in classes]
    colors = ['red' if cls in minority_classes else 'lightgreen' for cls in classes]
    ax2.bar(classes, target_counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Calculate minority percentage
    minority_total = sum(np.sum(target_y == cls) for cls in minority_classes)
    minority_percentage = (minority_total / len(target_y)) * 100
    
    ax2.set_title(f'{dataset_name} Target Set Distribution\n(Minority classes: {minority_percentage:.1f}%, Majority: {100-minority_percentage:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add count labels and percentages on bars
    total_target = len(target_y)
    for i, count in enumerate(target_counts):
        percentage = (count / total_target) * 100
        class_type = "M" if classes[i] in minority_classes else "Maj"
        ax2.text(classes[i], count + max(target_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)\n{class_type}', 
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{experiment_type}_source_target_distributions.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved source/target distribution plot: {filepath}")

def load_cifar_lt():
    """
    Load CIFAR-LT (long-tailed) dataset and return balanced source and naturally long-tailed target splits.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    X = np.concatenate([trainset.data, testset.data], axis=0)
    y = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)
    
    # Convert to float32 and normalize
    X = X.astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], -1)
    
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Balanced source: equal samples per class
    samples_per_class = 100  # 5000 total for 10 classes
    source_indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        selected = np.random.choice(cls_idx, size=samples_per_class, replace=False)
        source_indices.extend(selected)
    source_indices = np.array(source_indices)
    source_X, source_y = X[source_indices], y[source_indices]
    
    # Create long-tailed target pool with exponential decay
    target_indices = []
    max_target = 5000
    imbalance_factor = 50
    img_per_class = []
    for i in range(num_classes):
        n = int(max_target * (imbalance_factor ** (-i / (num_classes - 1))))
        img_per_class.append(n)
    
    for cls, n in zip(classes, img_per_class):
        cls_idx = np.where(y == cls)[0]
        cls_idx = np.setdiff1d(cls_idx, source_indices)  # Exclude source samples
        if len(cls_idx) >= n:
            selected = np.random.choice(cls_idx, size=n, replace=False)
        else:
            selected = cls_idx
        target_indices.extend(selected)
    
    target_indices = np.array(target_indices)
    target_pool_X, target_pool_y = X[target_indices], y[target_indices]
    
    # Print the actual long-tailed distribution
    print(f"CIFAR-LT distribution created:")
    for cls in classes:
        cls_count = np.sum(target_pool_y == cls)
        percentage = (cls_count / len(target_pool_y)) * 100
        print(f"  Class {cls}: {cls_count} samples ({percentage:.1f}%)")
    
    return (source_X, source_y), (target_pool_X, target_pool_y)

def load_cifar100_lt(imbalance_factor=100, max_samples=500):
    """
    Load long-tailed CIFAR-100 using exponential profile (Cao et al. 2019).
    Returns balanced source and long-tailed target splits.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    X = np.concatenate([trainset.data, testset.data], axis=0)
    y = np.concatenate([np.array(trainset.targets), np.array(testset.targets)], axis=0)
    
    # Convert to float32 and normalize
    X = X.astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], -1)
    
    classes = np.unique(y)
    num_classes = len(classes)
    # Balanced source
    samples_per_class = min(max_samples, int(len(y) / num_classes / 2))
    source_indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        selected = np.random.choice(cls_idx, size=samples_per_class, replace=False)
        source_indices.extend(selected)
    # Long-tailed target with remaining data
    target_indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        cls_idx = np.setdiff1d(cls_idx, source_indices)
        target_indices.extend(cls_idx)
    source_X, source_y = X[source_indices], y[source_indices]
    target_pool_X, target_pool_y = X[target_indices], y[target_indices]
    return (source_X, source_y), (target_pool_X, target_pool_y)


def load_imagenet_lt():
    """
    Load ImageNet-LT features and labels with enhanced download capabilities.
    """
    # Common paths where ImageNet-LT data might be stored
    possible_paths = [
        './data/imagenet_lt/',
        '../data/imagenet_lt/',
        '../../data/imagenet_lt/',
        '/data/imagenet_lt/',
        './imagenet_lt/'
    ]
    
    # Try to load from existing paths
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"Checking for ImageNet-LT data in {path}")
                
                # Check for different file naming conventions
                possible_files = [
                    # Pre-extracted features
                    ('features.npy', 'labels.npy'),
                    ('imagenet_lt_features.npy', 'imagenet_lt_labels.npy'),
                    ('resnet50_features.npy', 'resnet50_labels.npy'),
                    # Train/test splits
                    ('train_features.npy', 'train_labels.npy', 'test_features.npy', 'test_labels.npy'),
                    # HDF5 format
                    ('imagenet_lt.h5',),
                    # PyTorch format
                    ('imagenet_lt_features.pt', 'imagenet_lt_labels.pt')
                ]
                
                for file_combo in possible_files:
                    if len(file_combo) == 2:  # Single file pair
                        features_path = os.path.join(path, file_combo[0])
                        labels_path = os.path.join(path, file_combo[1])
                        
                        if os.path.exists(features_path) and os.path.exists(labels_path):
                            print(f"Loading ImageNet-LT from {path}")
                            
                            # Load based on file extension
                            if features_path.endswith('.npy'):
                                X = np.load(features_path)
                                y = np.load(labels_path)
                            elif features_path.endswith('.pt'):
                                X = torch.load(features_path).numpy()
                                y = torch.load(labels_path).numpy()
                            
                            # Split into source and target
                            source_X, target_X, source_y, target_y = train_test_split(
                                X, y, test_size=0.3, random_state=SEED, stratify=y
                            )
                            return (source_X, source_y), (target_X, target_y)
                    
                    elif len(file_combo) == 4:  # Train/test split
                        train_features_path = os.path.join(path, file_combo[0])
                        train_labels_path = os.path.join(path, file_combo[1])
                        test_features_path = os.path.join(path, file_combo[2])
                        test_labels_path = os.path.join(path, file_combo[3])
                        
                        if all(os.path.exists(p) for p in [train_features_path, train_labels_path, 
                                                          test_features_path, test_labels_path]):
                            print(f"Loading ImageNet-LT train/test split from {path}")
                            source_X = np.load(train_features_path)
                            source_y = np.load(train_labels_path) 
                            target_X = np.load(test_features_path)
                            target_y = np.load(test_labels_path)
                            return (source_X, source_y), (target_X, target_y)
                    
                    elif file_combo[0].endswith('.h5'):  # HDF5 format
                        h5_path = os.path.join(path, file_combo[0])
                        if os.path.exists(h5_path):
                            try:
                                import h5py
                                print(f"Loading ImageNet-LT from HDF5: {h5_path}")
                                with h5py.File(h5_path, 'r') as f:
                                    if 'train_features' in f and 'train_labels' in f:
                                        source_X = f['train_features'][:]
                                        source_y = f['train_labels'][:]
                                        target_X = f['test_features'][:]
                                        target_y = f['test_labels'][:]
                                        return (source_X, source_y), (target_X, target_y)
                                    else:
                                        X = f['features'][:]
                                        y = f['labels'][:]
                                        source_X, target_X, source_y, target_y = train_test_split(
                                            X, y, test_size=0.3, random_state=SEED, stratify=y
                                        )
                                        return (source_X, source_y), (target_X, target_y)
                            except ImportError:
                                print("h5py not installed, skipping HDF5 files")
                                continue
                            except Exception as e:
                                print(f"Error reading HDF5 file: {e}")
                                continue
                                
        except Exception as e:
            print(f"Failed to load ImageNet-LT from {path}: {e}")
            continue
    
    # If no data found, attempt to download or provide instructions
    print("ImageNet-LT data not found in standard paths.")
    print("\nTo obtain ImageNet-LT dataset, you have several options:")
    print("\n1. Download from official sources:")
    print("   - Original paper: https://arxiv.org/abs/1906.07413")
    print("   - GitHub repo: https://github.com/zhmiao/OpenLongTailRecognition-OLTR")
    print("\n2. Use Hugging Face datasets:")
    print("   pip install datasets")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('imagenet-1k')  # Then create long-tailed version")
    print("\n3. Extract from full ImageNet (requires ImageNet access)")
    print("\n4. Use pre-extracted features from research groups")
    
    # Attempt automatic download of splits
    try:
        download_imagenet_lt_info()
    except:
        print("\nGenerating synthetic ImageNet-LT-like data for prototyping...")
    
    # Generate synthetic data as fallback
    return generate_synthetic_imagenet_lt()

def download_imagenet_lt_info():
    """
    Download ImageNet-LT split information and class lists.
    """
    os.makedirs('./data/imagenet_lt/', exist_ok=True)
    
    base_urls = [
        "https://raw.githubusercontent.com/AkonLau/LTC/master/datasets/ImageNet_LT/",
        "https://github.com/AkonLau/LTC/raw/master/datasets/ImageNet_LT/"
    ]
    
    files_to_download = [
        "ImageNet_LT_train.txt",
        "ImageNet_LT_test.txt",
    ]
    
    for base_url in base_urls:
        success = True
        for filename in files_to_download:
            url = base_url + filename
            local_path = f'./data/imagenet_lt/{filename}'
            
            if not os.path.exists(local_path):
                try:
                    print(f"Downloading {filename}...")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
                    success = False
                    break
        
        if success:
            print("Successfully downloaded ImageNet-LT split files")
            print("You can now use these to create the dataset from full ImageNet")
            break

def generate_synthetic_imagenet_lt():
    """
    Generate synthetic ImageNet-LT-like data with realistic long-tailed distribution.
    """
    np.random.seed(SEED)
    num_classes = 1000  # ImageNet has 1000 classes
    feature_dim = 2048  # ResNet-50 feature dimension
    
    # Create realistic long-tailed distribution based on ImageNet-LT statistics
    # Head classes: 100-1280 samples, Medium: 20-100, Tail: 5-20
    head_classes = 390  # ~39% of classes
    medium_classes = 390  # ~39% of classes  
    tail_classes = 220   # ~22% of classes
    
    source_X_list, source_y_list = [], []
    target_X_list, target_y_list = [], []
    
    cls_idx = 0
    
    # Head classes (many samples)
    for i in range(head_classes):
        n_samples = np.random.randint(500, 1280)
        class_center = np.random.randn(feature_dim) * 0.3
        class_features = np.random.randn(n_samples, feature_dim) * 0.1 + class_center
        class_labels = np.full(n_samples, cls_idx)
        
        # Split 70-30 for train-test
        n_source = int(n_samples * 0.7)
        n_target = n_samples - n_source
        
        source_X_list.append(class_features[:n_source])
        source_y_list.append(class_labels[:n_source])
        target_X_list.append(class_features[n_source:n_source+n_target])
        target_y_list.append(class_labels[n_source:n_source+n_target])
        
        cls_idx += 1
    
    # Medium classes
    for i in range(medium_classes):
        n_samples = np.random.randint(20, 100)
        class_center = np.random.randn(feature_dim) * 0.3
        class_features = np.random.randn(n_samples, feature_dim) * 0.1 + class_center
        class_labels = np.full(n_samples, cls_idx)
        
        n_source = max(1, int(n_samples * 0.7))
        n_target = n_samples - n_source
        
        source_X_list.append(class_features[:n_source])
        source_y_list.append(class_labels[:n_source])
        if n_target > 0:
            target_X_list.append(class_features[n_source:n_source+n_target])
            target_y_list.append(class_labels[n_source:n_source+n_target])
        
        cls_idx += 1
    
    # Tail classes (few samples)
    for i in range(tail_classes):
        n_samples = np.random.randint(5, 20)
        class_center = np.random.randn(feature_dim) * 0.3
        class_features = np.random.randn(n_samples, feature_dim) * 0.1 + class_center
        class_labels = np.full(n_samples, cls_idx)
        
        n_source = max(1, int(n_samples * 0.7))
        n_target = n_samples - n_source
        
        source_X_list.append(class_features[:n_source])
        source_y_list.append(class_labels[:n_source])
        if n_target > 0:
            target_X_list.append(class_features[n_source:n_source+n_target])
            target_y_list.append(class_labels[n_source:n_source+n_target])
        
        cls_idx += 1
    
    source_X = np.vstack(source_X_list)
    source_y = np.hstack(source_y_list)
    target_X = np.vstack(target_X_list)
    target_y = np.hstack(target_y_list)
    
    print(f"Generated synthetic ImageNet-LT: Source {source_X.shape}, Target {target_X.shape}")
    print(f"Classes: {len(np.unique(source_y))} source, {len(np.unique(target_y))} target")
    
    # Print distribution statistics
    target_class_counts = np.bincount(target_y)
    head_count = np.sum(target_class_counts >= 100)
    medium_count = np.sum((target_class_counts >= 20) & (target_class_counts < 100))
    tail_count = np.sum(target_class_counts < 20)
    
    print(f"Target distribution: {head_count} head, {medium_count} medium, {tail_count} tail classes")
    
    return (source_X, source_y), (target_X, target_y)



def load_inaturalist_lt():
    """
    Load iNaturalist-LT features and labels. 
    First tries to load from common paths, then falls back to synthetic data.
    """
    # Common paths where iNaturalist-LT data might be stored
    possible_paths = [
        './data/inaturalist_lt/',
        '../data/inaturalist_lt/',
        '../../data/inaturalist_lt/',
        '/data/inaturalist_lt/',
        './inaturalist_lt/'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                # Try to load features and labels
                features_path = os.path.join(path, 'features.npy')
                labels_path = os.path.join(path, 'labels.npy')
                train_features_path = os.path.join(path, 'train_features.npy')
                train_labels_path = os.path.join(path, 'train_labels.npy')
                test_features_path = os.path.join(path, 'test_features.npy')
                test_labels_path = os.path.join(path, 'test_labels.npy')
                
                if os.path.exists(features_path) and os.path.exists(labels_path):
                    print(f"Loading iNaturalist-LT from {path}")
                    X = np.load(features_path)
                    y = np.load(labels_path)
                    # Split into source and target
                    source_X, target_X, source_y, target_y = train_test_split(
                        X, y, test_size=0.3, random_state=SEED, stratify=y
                    )
                    return (source_X, source_y), (target_X, target_y)
                
                elif os.path.exists(train_features_path) and os.path.exists(train_labels_path) and \
                     os.path.exists(test_features_path) and os.path.exists(test_labels_path):
                    print(f"Loading iNaturalist-LT train/test split from {path}")
                    source_X = np.load(train_features_path)
                    source_y = np.load(train_labels_path)
                    target_X = np.load(test_features_path)
                    target_y = np.load(test_labels_path)
                    return (source_X, source_y), (target_X, target_y)
        except Exception as e:
            print(f"Failed to load iNaturalist-LT from {path}: {e}")
            continue
    
    print("iNaturalist-LT data not found in standard paths. Generating synthetic long-tailed data...")
    
    # Generate synthetic iNaturalist-LT-like data
    np.random.seed(SEED)
    num_classes = 8142  # iNaturalist has 8142 classes
    feature_dim = 2048  # ResNet-50 feature dimension
    
    # Create long-tailed distribution
    max_samples = 1000  # Max samples for head classes
    min_samples = 2     # Min samples for tail classes
    
    source_X_list, source_y_list = [], []
    target_X_list, target_y_list = [], []
    
    for cls in range(num_classes):
        # Exponential decay for class sizes
        n_samples = int(max_samples * ((min_samples/max_samples) ** (cls/(num_classes-1))))
        n_samples = max(n_samples, min_samples)
        
        # Generate synthetic features for this class
        class_center = np.random.randn(feature_dim) * 0.5
        class_features = np.random.randn(n_samples, feature_dim) * 0.1 + class_center
        class_labels = np.full(n_samples, cls)
        
        # Split into source (30%) and target (70%)
        n_source = max(1, n_samples // 3)
        n_target = n_samples - n_source
        
        source_X_list.append(class_features[:n_source])
        source_y_list.append(class_labels[:n_source])
        target_X_list.append(class_features[n_source:n_source+n_target])
        target_y_list.append(class_labels[n_source:n_source+n_target])
    
    source_X = np.vstack(source_X_list)
    source_y = np.hstack(source_y_list)
    target_X = np.vstack(target_X_list)
    target_y = np.hstack(target_y_list)
    
    print(f"Generated synthetic iNaturalist-LT: Source {source_X.shape}, Target {target_X.shape}")
    return (source_X, source_y), (target_X, target_y)

def train_model(model, train_loader, val_loader=None, epochs=5, lr=0.01, device="cuda"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    return model

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def make_loader(X, y, batch_size=128):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def generate_mnist_target_set_ablation(pool_X, pool_y, total_size=2000):
    """
    Generate MNIST target set for ablation study:
    - Class 0: 1% of target set
    - Classes 1-9: 11% each of target set
    """
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values
    
    classes = np.unique(pool_y)
    target_indices = []
    
    # Class 0: 1%
    class_0_samples = int(total_size * 0.01)
    class_0_indices = np.where(pool_y == 0)[0]
    if len(class_0_indices) > 0:
        selected_0 = np.random.choice(class_0_indices, size=min(class_0_samples, len(class_0_indices)), replace=False)
        target_indices.extend(selected_0)
    
    # Classes 1-9: 11% each
    other_class_samples = int(total_size * 0.11)
    for cls in classes[1:]:
        cls_indices = np.where(pool_y == cls)[0]
        if len(cls_indices) > 0:
            selected_cls = np.random.choice(cls_indices, size=min(other_class_samples, len(cls_indices)), replace=False)
            target_indices.extend(selected_cls)
    
    target_indices = np.array(target_indices)
    np.random.shuffle(target_indices)
    
    if len(target_indices) == 0:
        raise ValueError("No valid target indices generated")
    
    actual_class_0_percentage = np.sum(pool_y[target_indices] == 0) / len(target_indices) * 100
    print(f"Generated MNIST ablation target set: {len(target_indices)} samples, class 0: {actual_class_0_percentage:.1f}%")
    
    return pool_X[target_indices], pool_y[target_indices]

# -------------------------
# Prototype Selection Functions
# -------------------------
def select_prototypes_mmd_critic(X, y, k_per_class=10, gamma=None):
    """
    Select prototypes using the existing MMD-critic implementation.
    """
    classes = np.unique(y)
    total_prototypes = min(len(classes) * k_per_class, len(X))
    
    print(f"Using MMD-critic to select {total_prototypes} prototypes...")
    
    # Convert to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).long()
    
    # Create Dataset object
    dataset = Dataset(X_torch, y_torch)
    
    # Compute RBF kernel
    dataset.compute_rbf_kernel(gamma=gamma)
    
    # Select prototypes using MMD-critic
    selected_indices = select_prototypes(dataset.K, total_prototypes)
    
    # Convert back to numpy indices and get the original indices (before sorting)
    selected_indices_np = selected_indices.cpu().numpy()
    original_indices = dataset.sort_indices[selected_indices_np].cpu().numpy()
    
    prototypes_X = X[original_indices]
    prototypes_y = y[original_indices]
    
    return prototypes_X, prototypes_y

def select_prototypes_fair_ot(X, y, sims, k_per_class=10, method='approx', regularization=0.01, epsilon=None):
    """
    Select prototypes using Fair Optimal Transport method.
    """
    classes = np.unique(y)
    n_source = sims.shape[0]
    total_prototypes = min(len(classes) * k_per_class, n_source)
    
    print(f"Using UniPROT ({method}) to select {total_prototypes} prototypes...")
    
    # Initialize UniPROT selector
    uniprot = UniPROT(regularization=regularization)
    
    # Select prototypes using UniPROT
    sims = torch.from_numpy(sims).to("cuda")
    if epsilon is not None:
        selected_indices, objectives = uniprot.prototype_selection(sims, total_prototypes, method=method, epsilon=epsilon)
    else:
        selected_indices, objectives = uniprot.prototype_selection(sims, total_prototypes, method=method)
    
    # Handle different return types
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.cpu().numpy()
    elif isinstance(selected_indices, list):
        selected_indices = np.array(selected_indices)
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    return prototypes_X, prototypes_y
        
def prototype_selection_uniform(X, y, k_per_class=10):
    """Original uniform prototype selection for comparison."""
    classes = np.unique(y)
    prototypes_X, prototypes_y = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        selected = np.random.choice(idx, size=min(k_per_class, len(idx)), replace=False)
        prototypes_X.append(X[selected])
        prototypes_y.append(y[selected])
    return np.vstack(prototypes_X), np.hstack(prototypes_y)

def prototype_selection(X, y, target_X, target_y, k_per_class=10, method='spotgreedy'):
    """
    Main prototype selection function that dispatches to different methods.
    """
    # Ensure inputs are float32
    X = X.astype(np.float32)
    target_X = target_X.astype(np.float32)
    
    # Convert to torch tensors with proper device handling
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        topt = lambda x: torch.from_numpy(x).float().to(device)
        
        print(f"Computing cost matrix on {device}...")
        dist, sims = evaluation.compute_cost_matrix(topt(X), topt(target_X), metric="cosine", return_sims=True)
        dist, sims = dist.cpu().numpy(), sims.cpu().numpy()
        
    except Exception as e:
        print(f"Error computing cost matrix: {e}")
        print("Falling back to uniform selection")
        return prototype_selection_uniform(X, y, k_per_class)
    
    if method == 'uniform':
        return prototype_selection_uniform(X, y, k_per_class)
    elif method == 'mmd_critic':
        return select_prototypes_mmd_critic(X, y, k_per_class)
    elif method == 'uniprot_approx':
        return select_prototypes_fair_ot(X, y, sims, k_per_class, method='approx')
    elif method == 'uniprot_stochastic':
        return select_prototypes_fair_ot(X, y, sims, k_per_class, method='approx', epsilon=0.01)
    elif method == 'uniprot_exact':
        with torch.no_grad():
            return select_prototypes_fair_ot(X, y, sims, k_per_class, method='exact')
    
    # Use SPOTgreedy for prototype selection
    classes = np.unique(y)
    total_prototypes = min(len(classes) * k_per_class, len(X))
    
    # Create target distribution (uniform across all classes) - keep as numpy array
    target_marginal = np.ones(len(target_X)) / len(target_X)
    
    print(f"Using SPOTgreedy to select {total_prototypes} prototypes...")
    # Use SPOTgreedy to select prototypes - pass numpy arrays only
    selected_indices = SPOT_GreedySubsetSelection(dist, total_prototypes)
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.cpu().numpy()
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    return prototypes_X, prototypes_y

def select_prototypes_mmd_critic_count(X, y, total_prototypes=50, gamma=None):
    """
    Select specific total number of prototypes using MMD-critic.
    """
    total_prototypes = min(total_prototypes, len(X))
    
    print(f"Using MMD-critic to select {total_prototypes} prototypes...")
    
    # Convert to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).long()
    
    # Create Dataset object
    dataset = Dataset(X_torch, y_torch)
    
    # Compute RBF kernel
    dataset.compute_rbf_kernel(gamma=gamma)
    
    # Select prototypes using MMD-critic
    selected_indices = select_prototypes(dataset.K, total_prototypes)
    
    # Convert back to numpy indices and get the original indices (before sorting)
    selected_indices_np = selected_indices.cpu().numpy()
    original_indices = dataset.sort_indices[selected_indices_np].cpu().numpy()
    
    prototypes_X = X[original_indices]
    prototypes_y = y[original_indices]
    
    return prototypes_X, prototypes_y

def select_prototypes_fair_ot_count(X, y, sims, total_prototypes=50, method='approx', regularization=0.01, epsilon=None):
    """
    Select specific total number of prototypes using Fair OT.
    """
    total_prototypes = min(total_prototypes, len(X))
    
    print(f"Using UniPROT ({method}) to select {total_prototypes} prototypes...")
    print(f"DEBUG: Input sims shape: {sims.shape}, X shape: {X.shape}")
    
    # Add some randomness to break potential caching/determinism
    np.random.seed(None)  # Reset seed to ensure randomness
    
    # Initialize UniPROT selector with varied regularization
    uniprot = UniPROT(regularization=regularization)
    
    # Convert sims to torch tensor if it's not already
    if isinstance(sims, np.ndarray):
        sims_torch = torch.from_numpy(sims).float().to("cuda")
    else:
        sims_torch = sims.to("cuda")
    
    print(f"DEBUG: sims_torch shape: {sims_torch.shape}")
    
    try:
        # Select prototypes using UniPROT
        selected_indices, objectives = uniprot.prototype_selection(sims_torch, total_prototypes, method=method, epsilon=epsilon)
        print(f"DEBUG: UniPROT returned indices type: {type(selected_indices)}")
        
        # Handle different return types and ensure we get numpy indices
        if isinstance(selected_indices, torch.Tensor):
            selected_indices = selected_indices.cpu().numpy()
        elif isinstance(selected_indices, list):
            selected_indices = np.array(selected_indices)
        
        print(f"DEBUG: Selected indices before filtering: {len(selected_indices)} indices")
        print(f"DEBUG: First few indices: {selected_indices[:min(10, len(selected_indices))]}")
        
        # Ensure indices are valid and unique
        selected_indices = np.unique(selected_indices)  # Remove duplicates
        selected_indices = selected_indices[selected_indices < len(X)]
        selected_indices = selected_indices[selected_indices >= 0]
        
        print(f"DEBUG: Selected indices after filtering: {len(selected_indices)} indices")
        
        # If we don't have enough unique indices, supplement with random selection
        if len(selected_indices) < total_prototypes:
            print(f"WARNING: UniPROT only selected {len(selected_indices)} unique prototypes, need {total_prototypes}")
            remaining_needed = total_prototypes - len(selected_indices)
            available_indices = np.setdiff1d(np.arange(len(X)), selected_indices)
            if len(available_indices) >= remaining_needed:
                additional_indices = np.random.choice(available_indices, remaining_needed, replace=False)
                selected_indices = np.concatenate([selected_indices, additional_indices])
                print(f"Added {remaining_needed} random indices to reach target count")
        
        # If we have too many, randomly subsample to exact count
        if len(selected_indices) > total_prototypes:
            selected_indices = np.random.choice(selected_indices, total_prototypes, replace=False)
            print(f"Subsampled to exactly {total_prototypes} prototypes")
        
        print(f"DEBUG: Final selected indices count: {len(selected_indices)}")
        
    except Exception as e:
        print(f"ERROR in UniPROT selection: {e}")
        print("Falling back to random selection")
        selected_indices = np.random.choice(len(X), total_prototypes, replace=False)
    
    if len(selected_indices) != total_prototypes:
        print(f"Warning: UniPROT selected {len(selected_indices)} prototypes instead of requested {total_prototypes}")
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    print(f"DEBUG: Returning {len(prototypes_X)} prototypes")
    return prototypes_X, prototypes_y

def balance_source_set(source_X, source_y, samples_per_class=None, target_total_size=5000):
    """
    Ensure the source set has uniform class representation and specific total size.
    Handles index validation and ensures no out of bounds access.
    """
    # Convert inputs to numpy arrays if they're not already
    if hasattr(source_X, 'values'):
        source_X = source_X.values
    if hasattr(source_y, 'values'):
        source_y = source_y.values
    
    # Convert to numpy array if they are lists
    source_X = np.array(source_X)
    source_y = np.array(source_y)
    
    classes = np.unique(source_y)
    class_counts = [np.sum(source_y == cls) for cls in classes]
    
    if samples_per_class is None:
        # Calculate samples per class to reach target total size
        samples_per_class = target_total_size // len(classes)
        # Ensure we don't exceed available samples for any class
        samples_per_class = min(samples_per_class, min(class_counts))
    
    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(source_y == cls)[0]
        if len(cls_indices) > 0:  # Only proceed if we have samples for this class
            if len(cls_indices) >= samples_per_class:
                selected = np.random.choice(cls_indices, size=samples_per_class, replace=False)
            else:
                # If we don't have enough samples, use all available with replacement
                selected = np.random.choice(cls_indices, size=samples_per_class, replace=True)
            balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    
    # Ensure all indices are valid
    balanced_indices = balanced_indices[balanced_indices < len(source_X)]
    if len(balanced_indices) == 0:
        raise ValueError("No valid indices found after balancing")
    
    np.random.shuffle(balanced_indices)
    
    samples_per_class_actual = len(balanced_indices) // len(classes)
    print(f"Balanced source set: {len(balanced_indices)} samples ({samples_per_class_actual} per class)")
    
    return source_X[balanced_indices], source_y[balanced_indices]

def plot_class_histogram(target_y, classes, skew_percent, dataset_name, method, run_idx):
    """
    Plot class histogram for target distribution.
    """
    plt.figure(figsize=(8, 4))
    counts = [np.sum(target_y == cls) for cls in classes]
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - {method} - Skew {skew_percent}% (Run {run_idx+1})')
    plt.tight_layout()
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots_ablation')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{method}_skew{skew_percent}_run{run_idx+1}_histogram.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram plot: {filepath}")

def run_experiment(dataset_name, minority_class=None):
    print(f"\n{'='*80}")
    print(f"Running ablation experiment for {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    try:
        if dataset_name == "cifarlt":
            (source_X, source_y), (target_pool_X, target_pool_y) = load_cifar_lt()
            target_X, target_y = target_pool_X, target_pool_y
            print(f"Using natural CIFAR-LT distribution: {len(target_X)} samples")
        elif dataset_name == "cifar100lt":
            (source_X, source_y), (target_pool_X, target_pool_y) = load_cifar100_lt()
            target_X, target_y = target_pool_X, target_pool_y
            print(f"Using natural CIFAR-100-LT distribution: {len(target_X)} samples")
        elif dataset_name == "imagenetlt":
            result = load_imagenet_lt()
            if result is None:
                print("ImageNet-LT data not available.")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = result
            target_X, target_y = target_pool_X, target_pool_y
            print(f"Using natural ImageNet-LT distribution: {len(target_X)} samples")
        elif dataset_name == "inaturalistlt":
            result = load_inaturalist_lt()
            if result is None:
                print("iNaturalist-LT data not available.")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = result
            target_X, target_y = target_pool_X, target_pool_y
            print(f"Using natural iNaturalist-LT distribution: {len(target_X)} samples")
        else:
            dataset_result = data.load_dataset(dataset_name)
            if dataset_result is None:
                print(f"Failed to load {dataset_name} dataset")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
            scaler = StandardScaler()
            source_X = scaler.fit_transform(source_X.astype(np.float32))
            target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
            source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
        
        # Ensure all data is float32
        source_X = source_X.astype(np.float32)
        target_X = target_X.astype(np.float32)
        source_y = source_y.astype(np.int64)
        target_y = target_y.astype(np.int64)
        
        print(f"Loaded {dataset_name}: Source {source_X.shape} {source_X.dtype}, Target {target_X.shape} {target_X.dtype}")
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # For large synthetic datasets, subsample for efficiency
    if dataset_name in ["imagenetlt", "inaturalistlt"]:
        if len(source_X) > 10000:
            print(f"Subsampling {dataset_name} source set from {len(source_X)} to 10000 for efficiency...")
            source_indices = np.random.choice(len(source_X), 10000, replace=False)
            source_X = source_X[source_indices]
            source_y = source_y[source_indices]
        
        if len(target_X) > 20000:
            print(f"Subsampling {dataset_name} target set from {len(target_X)} to 20000 for efficiency...")
            target_indices = np.random.choice(len(target_X), 20000, replace=False)
            target_X = target_X[target_indices]
            target_y = target_y[target_indices]
    
    plot_source_target_histograms(source_y, target_y, dataset_name, experiment_type='natural_distribution')
    
    # Identify minority classes
    minority_classes = identify_minority_classes(target_y, minority_threshold=0.1)
    
    # Run comprehensive prototype selection evaluation
    print(f"[INFO] Running comprehensive prototype selection evaluation for {dataset_name}")
    methods = ['uniprot_stochastic','spotgreedy', 'mmd_critic', 'uniform','uniprot_approx']
    prototype_counts = [10, 25, 50, 75, 100, 150, 200]
    runs = 1
    
    # Store results for plotting
    all_results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} Method ---")
        method_results = {
            'overall_accuracies': [],
            'minority_class_accuracies': {cls: [] for cls in minority_classes},
            'minority_class_maps': {cls: [] for cls in minority_classes}
        }
        
        for run in range(runs):
            print(f"Run {run+1}/{runs} for {method}")
            run_overall = []
            run_minority_accs = {cls: [] for cls in minority_classes}
            run_minority_maps = {cls: [] for cls in minority_classes}
            
            for proto_count in prototype_counts:
                try:
                    print(f"  Testing with {proto_count} prototypes...")
                    
                    # Select prototypes
                    prototypes_X, prototypes_y = prototype_selection_with_count(
                        source_X, source_y, target_X, target_y, 
                        total_prototypes=proto_count, method=method
                    )
                    
                    if len(prototypes_X) == 0:
                        run_overall.append(0.0)
                        for cls in minority_classes:
                            run_minority_accs[cls].append(0.0)
                            run_minority_maps[cls].append(0.0)
                        continue
                    
                    # Evaluate with detailed results
                    overall_acc, class_accs, class_counts, class_maps = evaluate_1nn_with_map(
                        prototypes_X, prototypes_y, target_X, target_y
                    )
                    
                    run_overall.append(overall_acc)
                    
                    # Store minority class results
                    for cls in minority_classes:
                        run_minority_accs[cls].append(class_accs.get(cls, 0.0))
                        run_minority_maps[cls].append(class_maps.get(cls, 0.0))
                    
                    print(f"    Overall accuracy: {overall_acc:.4f}")
                    for cls in minority_classes:
                        print(f"    Class {cls}: Acc={class_accs.get(cls, 0.0):.4f}, mAP={class_maps.get(cls, 0.0):.4f}")
                    
                except Exception as e:
                    print(f"    Error with {proto_count} prototypes: {e}")
                    run_overall.append(0.0)
                    for cls in minority_classes:
                        run_minority_accs[cls].append(0.0)
                        run_minority_maps[cls].append(0.0)
            
            method_results['overall_accuracies'].append(run_overall)
            for cls in minority_classes:
                method_results['minority_class_accuracies'][cls].append(run_minority_accs[cls])
                method_results['minority_class_maps'][cls].append(run_minority_maps[cls])
        
        # Calculate averages
        all_results[method] = {
            'overall_mean': np.mean(method_results['overall_accuracies'], axis=0),
            'overall_std': np.std(method_results['overall_accuracies'], axis=0),
            'minority_acc_mean': {cls: np.mean(method_results['minority_class_accuracies'][cls], axis=0) 
                                  for cls in minority_classes},
            'minority_acc_std': {cls: np.std(method_results['minority_class_accuracies'][cls], axis=0) 
                                 for cls in minority_classes},
            'minority_map_mean': {cls: np.mean(method_results['minority_class_maps'][cls], axis=0) 
                                  for cls in minority_classes},
            'minority_map_std': {cls: np.std(method_results['minority_class_maps'][cls], axis=0) 
                                 for cls in minority_classes}
        }
    
    # Generate comprehensive plots
    plot_comprehensive_results(all_results, prototype_counts, minority_classes, dataset_name)
    
    # Save detailed results
    results = {
        'dataset': dataset_name,
        'minority_classes': minority_classes,
        'prototype_counts': prototype_counts,
        'methods': methods,
        'detailed_results': all_results
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results_ablation')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{dataset_name}_comprehensive_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == 'detailed_results':
            json_results[key] = {}
            for method, method_data in value.items():
                json_results[key][method] = {}
                for metric, metric_data in method_data.items():
                    if isinstance(metric_data, dict):
                        json_results[key][method][metric] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                           for k, v in metric_data.items()}
                    else:
                        json_results[key][method][metric] = metric_data.tolist() if isinstance(metric_data, np.ndarray) else metric_data
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"[INFO] Comprehensive results saved to {results_path}")

def evaluate_1nn(P_X, P_y, target_X, target_y):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(P_X, P_y)
    pred = clf.predict(target_X)
    acc = accuracy_score(target_y, pred)
    return acc

def evaluate_1nn_detailed(P_X, P_y, target_X, target_y):
    """
    Evaluate 1-NN with detailed per-class results.
    
    Returns:
        overall_accuracy: Overall accuracy
        class_accuracies: Dictionary with per-class accuracies
        class_counts: Dictionary with per-class sample counts
    """
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(P_X, P_y)
    pred = clf.predict(target_X)
    
    overall_acc = accuracy_score(target_y, pred)
    
    classes = np.unique(target_y)
    class_accs = {}
    class_counts = {}
    
    for cls in classes:
        cls_mask = target_y == cls
        cls_pred = pred[cls_mask]
        cls_true = target_y[cls_mask]
        class_counts[cls] = len(cls_true)
        if len(cls_true) > 0:
            class_accs[cls] = accuracy_score(cls_true, cls_pred)
        else:
            class_accs[cls] = 0.0
    
    return overall_acc, class_accs, class_counts

def evaluate_1nn_with_map(P_X, P_y, target_X, target_y):
    """
    Evaluate 1-NN with detailed per-class results including mean Average Precision.
    
    Returns:
        overall_accuracy: Overall accuracy
        class_accuracies: Dictionary with per-class accuracies
        class_counts: Dictionary with per-class sample counts
        class_maps: Dictionary with per-class mean Average Precision
    """
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(P_X, P_y)
    pred = clf.predict(target_X)
    pred_proba = clf.predict_proba(target_X)
    
    overall_acc = accuracy_score(target_y, pred)
    
    classes = np.unique(target_y)
    class_accs = {}
    class_counts = {}
    class_maps = {}
    
    # Binarize labels for mAP calculation
    y_bin = label_binarize(target_y, classes=classes)
    
    for i, cls in enumerate(classes):
        cls_mask = target_y == cls
        cls_pred = pred[cls_mask]
        cls_true = target_y[cls_mask]
        class_counts[cls] = len(cls_true)
        
        if len(cls_true) > 0:
            class_accs[cls] = accuracy_score(cls_true, cls_pred)
            
            # Calculate mAP for this class
            if len(classes) > 2:  # Multi-class case
                cls_proba = pred_proba[:, i]
                cls_true_bin = y_bin[:, i]
                class_maps[cls] = average_precision_score(cls_true_bin, cls_proba)
            else:  # Binary case
                if cls == classes[1]:  # Positive class
                    cls_proba = pred_proba[:, 1]
                    class_maps[cls] = average_precision_score(target_y == cls, cls_proba)
                else:
                    class_maps[cls] = 0.0
        else:
            class_accs[cls] = 0.0
            class_maps[cls] = 0.0
    
    return overall_acc, class_accs, class_counts, class_maps

def prototype_selection_with_count(X, y, target_X, target_y, total_prototypes=50, method='spotgreedy'):
    """
    Select a specific total number of prototypes regardless of class distribution.
    """
    print(f"DEBUG: prototype_selection_with_count called with {total_prototypes} prototypes, method={method}")
    
    # Ensure inputs are float32
    X = X.astype(np.float32)
    target_X = target_X.astype(np.float32)
    
    if method == 'uniform':
        indices = np.random.choice(len(X), size=min(total_prototypes, len(X)), replace=False)
        return X[indices], y[indices]
    elif method == 'mmd_critic':
        return select_prototypes_mmd_critic_count(X, y, total_prototypes)
    
    # For methods that need similarity/distance matrices
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        topt = lambda x: torch.from_numpy(x).float().to(device)
        
        dist, sims = evaluation.compute_cost_matrix(topt(X), topt(target_X), metric="cosine", return_sims=True)
        dist, sims = dist.cpu().numpy(), sims.cpu().numpy()
        
    except Exception as e:
        print(f"Error computing cost matrix: {e}")
        indices = np.random.choice(len(X), size=min(total_prototypes, len(X)), replace=False)
        return X[indices], y[indices]
    
    if method == 'uniprot_approx':
        return select_prototypes_fair_ot_count(X, y, sims, total_prototypes, method='approx')
    elif method == 'uniprot_exact':
        with torch.no_grad():
            return select_prototypes_fair_ot_count(X, y, sims, total_prototypes, method='exact')
    elif method == 'uniprot_stochastic':
        return select_prototypes_fair_ot_count(X, y, sims, total_prototypes, method='approx', epsilon=0.01)
    
    # Use SPOTgreedy for prototype selection
    total_prototypes = min(total_prototypes, len(X))
    target_marginal = np.ones(len(target_X)) / len(target_X)
    
    print(f"Using SPOTgreedy to select {total_prototypes} prototypes...")
    selected_indices = SPOT_GreedySubsetSelection(dist, total_prototypes)
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.cpu().numpy()
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    return prototypes_X, prototypes_y

def plot_comprehensive_results(all_results, prototype_counts, minority_classes, dataset_name):
    """
    Plot comprehensive results including overall accuracy and per-minority-class accuracy/mAP.
    """
    methods = list(all_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
    
    # 1. Overall accuracy plot
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        mean_acc = all_results[method]['overall_mean']
        std_acc = all_results[method]['overall_std']
        
        plt.errorbar(prototype_counts, mean_acc, yerr=std_acc,
                    color=color, marker=marker, linewidth=2, markersize=6,
                    label=method.replace('_', ' ').title(), capsize=3)
    
    plt.title(f'{dataset_name} - Overall Accuracy vs Number of Prototypes', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Prototypes', fontsize=14)
    plt.ylabel('1-NN Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save overall accuracy plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_overall_accuracy_comparison.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved overall accuracy plot: {filepath}")
    
    # 2. Per-minority-class plots (accuracy and mAP side by side)
    for cls in minority_classes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy subplot
        for i, method in enumerate(methods):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            mean_acc = all_results[method]['minority_acc_mean'][cls]
            std_acc = all_results[method]['minority_acc_std'][cls]
            
            ax1.errorbar(prototype_counts, mean_acc, yerr=std_acc,
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=method.replace('_', ' ').title(), capsize=3)
        
        ax1.set_title(f'Class {cls} - Accuracy vs Number of Prototypes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Prototypes', fontsize=12)
        ax1.set_ylabel('1-NN Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # mAP subplot
        for i, method in enumerate(methods):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            mean_map = all_results[method]['minority_map_mean'][cls]
            std_map = all_results[method]['minority_map_std'][cls]
            
            ax2.errorbar(prototype_counts, mean_map, yerr=std_map,
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=method.replace('_', ' ').title(), capsize=3)
        
        ax2.set_title(f'Class {cls} - mAP vs Number of Prototypes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Prototypes', fontsize=12)
        ax2.set_ylabel('Mean Average Precision', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save minority class plot
        filename = f'{dataset_name}_class_{cls}_accuracy_map_comparison.png'
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved class {cls} comparison plot: {filepath}")

def main():
    """
    Main function to run comprehensive ablation experiments across multiple datasets.
    """
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDY ON PROTOTYPE SELECTION METHODS")
    print("=" * 80)
    
    # Configuration
    experiment_datasets = [
        #"cifarlt",
        # "cifar100lt",     # Uncomment to include CIFAR-100-LT
        #"imagenetlt",     # Uncomment to include ImageNet-LT  
        "inaturalistlt",  # Uncomment to include iNaturalist-LT
    ]
    
    # Run comprehensive experiments on long-tailed datasets
    print("\n[PHASE 1] Running experiments on long-tailed datasets...")
    for dataset_name in experiment_datasets:
        try:
            print(f"\n{'*' * 60}")
            print(f"Processing {dataset_name.upper()}")
            print(f"{'*' * 60}")
            run_experiment(dataset_name)
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results_ablation')
    plots_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    
    print(f"\nResults saved in: {results_dir}")
    print(f"Plots saved in: {plots_dir}")
    
    # Check if results files exist and print summary
    for dataset_name in experiment_datasets:
        results_file = os.path.join(results_dir, f'{dataset_name}_comprehensive_results.json')
        if os.path.exists(results_file):
            print(f"✓ {dataset_name}: Comprehensive results available")
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    methods = results.get('methods', [])
                    minority_classes = results.get('minority_classes', [])
                    print(f"  - Methods tested: {', '.join(methods)}")
                    print(f"  - Minority classes identified: {minority_classes}")
            except Exception as e:
                print(f"  - Error reading results: {e}")
        else:
            print(f"✗ {dataset_name}: No results found")
    
    print(f"\nGenerated plots:")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file}")
    else:
        print("  - No plots directory found")
    
    print(f"\nExperiment completed successfully!")
    print(f"Total datasets processed: {len(experiment_datasets)}")
    print(f"Methods compared: SPOTgreedy, MMD-critic, Uniform, UniPROT-Approx, UniPROT-Stochastic")
    print(f"Metrics evaluated: Overall Accuracy, Per-class Accuracy, Mean Average Precision")









# ...existing code...

import torch.nn.functional as F
from torchvision.models import vit_b_16
import torch.backends.cudnn as cudnn

def create_vit_model(num_classes, pretrained=True):
    """
    Create a Vision Transformer model for classification.
    """
    model = vit_b_16(pretrained=pretrained)
    # Replace the classifier head
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def train_vit_model(prototypes_X, prototypes_y, num_classes, epochs=20, lr=0.001, batch_size=32, device="cuda"):
    """
    Train a ViT model on the selected prototypes.
    """
    # Prepare data
    if len(prototypes_X.shape) == 2:
        # If flattened, try to reshape back to image format
        if prototypes_X.shape[1] == 3072:  # CIFAR-10/100 flattened
            prototypes_X = prototypes_X.reshape(-1, 3, 32, 32)
        elif prototypes_X.shape[1] == 784:  # MNIST flattened
            prototypes_X = prototypes_X.reshape(-1, 1, 28, 28)
            # Convert grayscale to RGB for ViT
            prototypes_X = np.repeat(prototypes_X, 3, axis=1)
        else:
            # For other datasets, use the features as-is but reshape for ViT input
            # Create dummy image-like data from features
            feat_dim = int(np.sqrt(prototypes_X.shape[1] / 3)) if prototypes_X.shape[1] % 3 == 0 else 32
            if feat_dim * feat_dim * 3 != prototypes_X.shape[1]:
                feat_dim = 32
                # Pad or truncate features to fit 32x32x3
                if prototypes_X.shape[1] < 3072:
                    padded = np.zeros((prototypes_X.shape[0], 3072))
                    padded[:, :prototypes_X.shape[1]] = prototypes_X
                    prototypes_X = padded
                else:
                    prototypes_X = prototypes_X[:, :3072]
            prototypes_X = prototypes_X.reshape(-1, 3, feat_dim, feat_dim)
    
    # Resize to 224x224 for ViT (standard input size)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform the prototype data
    transformed_data = []
    for i in range(len(prototypes_X)):
        img = prototypes_X[i]
        if img.min() < 0:  # Normalize to [0,1] if needed
            img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        transformed_img = transform(img)
        transformed_data.append(transformed_img)
    
    prototypes_X_tensor = torch.stack(transformed_data)
    prototypes_y_tensor = torch.tensor(prototypes_y, dtype=torch.long)
    
    # Create data loader
    train_dataset = TensorDataset(prototypes_X_tensor, prototypes_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = create_vit_model(num_classes, pretrained=True)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train Acc={100.*correct/total:.2f}%")
    
    return model

def evaluate_vit_model(model, target_X, target_y, device="cuda"):
    """
    Evaluate ViT model on target data and return detailed results.
    """
    model.eval()
    
    # Prepare target data similar to training
    if len(target_X.shape) == 2:
        if target_X.shape[1] == 3072:  # CIFAR-10/100 flattened
            target_X = target_X.reshape(-1, 3, 32, 32)
        elif target_X.shape[1] == 784:  # MNIST flattened
            target_X = target_X.reshape(-1, 1, 28, 28)
            target_X = np.repeat(target_X, 3, axis=1)
        else:
            feat_dim = int(np.sqrt(target_X.shape[1] / 3)) if target_X.shape[1] % 3 == 0 else 32
            if feat_dim * feat_dim * 3 != target_X.shape[1]:
                feat_dim = 32
                if target_X.shape[1] < 3072:
                    padded = np.zeros((target_X.shape[0], 3072))
                    padded[:, :target_X.shape[1]] = target_X
                    target_X = padded
                else:
                    target_X = target_X[:, :3072]
            target_X = target_X.reshape(-1, 3, feat_dim, feat_dim)
    
    # Transform target data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed_data = []
    for i in range(len(target_X)):
        img = target_X[i]
        if img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        transformed_img = transform(img)
        transformed_data.append(transformed_img)
    
    target_X_tensor = torch.stack(transformed_data)
    target_y_tensor = torch.tensor(target_y, dtype=torch.long)
    
    # Create data loader
    test_dataset = TensorDataset(target_X_tensor, target_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(prob.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Calculate overall accuracy
    overall_acc = accuracy_score(all_targets, all_predictions)
    
    # Calculate per-class accuracy and mAP
    classes = np.unique(all_targets)
    class_accs = {}
    class_counts = {}
    class_maps = {}
    
    # Binarize labels for mAP calculation
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import average_precision_score
    
    y_bin = label_binarize(all_targets, classes=classes)
    
    for i, cls in enumerate(classes):
        cls_mask = all_targets == cls
        cls_pred = all_predictions[cls_mask]
        cls_true = all_targets[cls_mask]
        class_counts[cls] = len(cls_true)
        
        if len(cls_true) > 0:
            class_accs[cls] = accuracy_score(cls_true, cls_pred)
            
            # Calculate mAP for this class
            if len(classes) > 2:  # Multi-class case
                cls_proba = all_probabilities[:, i]
                cls_true_bin = y_bin[:, i]
                class_maps[cls] = average_precision_score(cls_true_bin, cls_proba)
            else:  # Binary case
                if cls == classes[1]:  # Positive class
                    cls_proba = all_probabilities[:, 1]
                    class_maps[cls] = average_precision_score(all_targets == cls, cls_proba)
                else:
                    class_maps[cls] = 0.0
        else:
            class_accs[cls] = 0.0
            class_maps[cls] = 0.0
    
    return overall_acc, class_accs, class_counts, class_maps

def run_experiment_with_vit(dataset_name, minority_class=None):
    """
    Extended experiment function that includes ViT training and evaluation.
    This is a new function that supplements the existing run_experiment function.
    """
    print(f"\n{'='*80}")
    print(f"Running ViT experiment for {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset (same as original function)
    try:
        if dataset_name == "cifarlt":
            (source_X, source_y), (target_pool_X, target_pool_y) = load_cifar_lt()
            target_X, target_y = target_pool_X, target_pool_y
        elif dataset_name == "cifar100lt":
            (source_X, source_y), (target_pool_X, target_pool_y) = load_cifar100_lt()
            target_X, target_y = target_pool_X, target_pool_y
        elif dataset_name == "imagenetlt":
            result = load_imagenet_lt()
            if result is None:
                print("ImageNet-LT data not available.")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = result
            target_X, target_y = target_pool_X, target_pool_y
        elif dataset_name == "inaturalistlt":
            result = load_inaturalist_lt()
            if result is None:
                print("iNaturalist-LT data not available.")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = result
            target_X, target_y = target_pool_X, target_pool_y
        else:
            dataset_result = data.load_dataset(dataset_name)
            if dataset_result is None:
                print(f"Failed to load {dataset_name} dataset")
                return
            (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
            scaler = StandardScaler()
            source_X = scaler.fit_transform(source_X.astype(np.float32))
            target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
            source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
        
        # Ensure all data is float32
        source_X = source_X.astype(np.float32)
        target_X = target_X.astype(np.float32)
        source_y = source_y.astype(np.int64)
        target_y = target_y.astype(np.int64)
        
        print(f"Loaded {dataset_name} for ViT: Source {source_X.shape}, Target {target_X.shape}")
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return
    
    # For efficiency, subsample for ViT experiments
    if len(source_X) > 5000:
        print(f"Subsampling source set from {len(source_X)} to 5000 for ViT efficiency...")
        source_indices = np.random.choice(len(source_X), 5000, replace=False)
        source_X = source_X[source_indices]
        source_y = source_y[source_indices]
    
    if len(target_X) > 10000:
        print(f"Subsampling target set from {len(target_X)} to 10000 for ViT efficiency...")
        target_indices = np.random.choice(len(target_X), 10000, replace=False)
        target_X = target_X[target_indices]
        target_y = target_y[target_indices]
    
    # Identify minority classes
    minority_classes = identify_minority_classes(target_y, minority_threshold=0.1)
    num_classes = len(np.unique(np.concatenate([source_y, target_y])))
    
    # ViT experiment configuration
    methods = ['uniprot_stochastic', 'spotgreedy', 'mmd_critic', 'uniform', 'uniprot_approx']
    prototype_counts = [50, 100, 150]  # Reduced for ViT experiments due to computational cost
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Store ViT results
    vit_results = {}
    
    print(f"[INFO] Running ViT experiments for {dataset_name}")
    print(f"Device: {device}, Classes: {num_classes}, Minority classes: {minority_classes}")
    
    for method in methods:
        print(f"\n--- ViT Training with {method.upper()} Method ---")
        method_results = {
            'overall_accuracies': [],
            'minority_class_accuracies': {cls: [] for cls in minority_classes},
            'minority_class_maps': {cls: [] for cls in minority_classes}
        }
        
        for proto_count in prototype_counts:
            try:
                print(f"  Training ViT with {proto_count} prototypes...")
                
                # Select prototypes
                prototypes_X, prototypes_y = prototype_selection_with_count(
                    source_X, source_y, target_X, target_y, 
                    total_prototypes=proto_count, method=method
                )
                
                if len(prototypes_X) == 0:
                    print(f"    No prototypes selected for {method} with {proto_count} prototypes")
                    method_results['overall_accuracies'].append(0.0)
                    for cls in minority_classes:
                        method_results['minority_class_accuracies'][cls].append(0.0)
                        method_results['minority_class_maps'][cls].append(0.0)
                    continue
                
                print(f"    Selected {len(prototypes_X)} prototypes, training ViT...")
                
                # Train ViT model
                model = train_vit_model(prototypes_X, prototypes_y, num_classes, 
                                      epochs=15, lr=0.001, device=device)
                
                # Evaluate ViT model
                overall_acc, class_accs, class_counts, class_maps = evaluate_vit_model(
                    model, target_X, target_y, device=device
                )
                
                method_results['overall_accuracies'].append(overall_acc)
                
                # Store minority class results
                for cls in minority_classes:
                    method_results['minority_class_accuracies'][cls].append(class_accs.get(cls, 0.0))
                    method_results['minority_class_maps'][cls].append(class_maps.get(cls, 0.0))
                
                print(f"    ViT Overall accuracy: {overall_acc:.4f}")
                for cls in minority_classes:
                    print(f"    ViT Class {cls}: Acc={class_accs.get(cls, 0.0):.4f}, mAP={class_maps.get(cls, 0.0):.4f}")
                
                # Clean up model to save memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error training ViT with {proto_count} prototypes: {e}")
                method_results['overall_accuracies'].append(0.0)
                for cls in minority_classes:
                    method_results['minority_class_accuracies'][cls].append(0.0)
                    method_results['minority_class_maps'][cls].append(0.0)
        
        vit_results[method] = method_results
    
    # Plot ViT results
    plot_vit_results(vit_results, prototype_counts, minority_classes, dataset_name)
    
    # Save ViT results
    results = {
        'dataset': dataset_name,
        'minority_classes': minority_classes,
        'prototype_counts': prototype_counts,
        'methods': methods,
        'vit_results': vit_results
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results_ablation')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{dataset_name}_vit_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] ViT results saved to {results_path}")

def plot_vit_results(vit_results, prototype_counts, minority_classes, dataset_name):
    """
    Plot ViT-specific results.
    """
    methods = list(vit_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
    
    # 1. ViT Overall accuracy plot
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        accuracies = vit_results[method]['overall_accuracies']
        
        plt.plot(prototype_counts, accuracies, color=color, marker=marker, 
                linewidth=2, markersize=8, label=method.replace('_', ' ').title())
    
    plt.title(f'{dataset_name} - ViT Overall Accuracy vs Number of Prototypes', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Number of Prototypes', fontsize=14)
    plt.ylabel('ViT Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save ViT overall accuracy plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_vit_overall_accuracy_comparison.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ViT overall accuracy plot: {filepath}")
    
    # 2. ViT Per-minority-class plots
    for cls in minority_classes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy subplot
        for i, method in enumerate(methods):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            accuracies = vit_results[method]['minority_class_accuracies'][cls]
            
            ax1.plot(prototype_counts, accuracies, color=color, marker=marker,
                    linewidth=2, markersize=8, label=method.replace('_', ' ').title())
        
        ax1.set_title(f'ViT Class {cls} - Accuracy vs Number of Prototypes', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Prototypes', fontsize=12)
        ax1.set_ylabel('ViT Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # mAP subplot
        for i, method in enumerate(methods):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            maps = vit_results[method]['minority_class_maps'][cls]
            
            ax2.plot(prototype_counts, maps, color=color, marker=marker,
                    linewidth=2, markersize=8, label=method.replace('_', ' ').title())
        
        ax2.set_title(f'ViT Class {cls} - mAP vs Number of Prototypes', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Prototypes', fontsize=12)
        ax2.set_ylabel('ViT Mean Average Precision', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save ViT minority class plot
        filename = f'{dataset_name}_vit_class_{cls}_accuracy_map_comparison.png'
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved ViT class {cls} comparison plot: {filepath}")

def main_with_vit():
    """
    Extended main function that runs both 1-NN and ViT experiments.
    This supplements the existing main() function.
    """
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDY WITH ViT TRAINING")
    print("=" * 80)
    
    # Configuration
    experiment_datasets = [
        "cifarlt",
        # "cifar100lt",     # Uncomment to include CIFAR-100-LT
    ]
    
    # Run original 1-NN experiments
    print("\n[PHASE 1] Running 1-NN experiments...")
    for dataset_name in experiment_datasets:
        try:
            run_experiment(dataset_name)
        except Exception as e:
            print(f"ERROR in 1-NN experiment for {dataset_name}: {e}")
    
    # Run ViT experiments
    print("\n[PHASE 2] Running ViT experiments...")
    for dataset_name in experiment_datasets:
        try:
            run_experiment_with_vit(dataset_name)
        except Exception as e:
            print(f"ERROR in ViT experiment for {dataset_name}: {e}")
    
    print("\n" + "=" * 80)
    print("COMPLETE EXPERIMENT SUMMARY")
    print("=" * 80)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results_ablation')
    plots_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    
    print(f"\nAll results saved in: {results_dir}")
    print(f"All plots saved in: {plots_dir}")
    
    for dataset_name in experiment_datasets:
        print(f"\n{dataset_name.upper()}:")
        
        # Check 1-NN results
        results_file = os.path.join(results_dir, f'{dataset_name}_comprehensive_results.json')
        if os.path.exists(results_file):
            print(f"  ✓ 1-NN results available")
        else:
            print(f"  ✗ 1-NN results missing")
        
        # Check ViT results
        vit_results_file = os.path.join(results_dir, f'{dataset_name}_vit_results.json')
        if os.path.exists(vit_results_file):
            print(f"  ✓ ViT results available")
        else:
            print(f"  ✗ ViT results missing")

# Uncomment the line below to run experiments with ViT
# if __name__ == "__main__":
#     main_with_vit()



if __name__ == "__main__":
    main()