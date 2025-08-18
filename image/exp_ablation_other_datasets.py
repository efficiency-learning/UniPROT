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

# Add the parent python directory to the path to import SPOTgreedy and MMD-critic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMDcritic'))
from baselines.MMDcritic.mmd_critic import Dataset, select_prototypes
from SPOTgreedy import SPOT_GreedySubsetSelection
from image.proto_selection_evals import data
from FairOT.FairOptimalTransport import FairOptimalTransport
from baselines import evaluation

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# Create Prototype Set using SPOTgreedy
# -------------------------
def select_prototypes_mmd_critic(X, y, k_per_class=10, gamma=None):
    """
    Select prototypes using the existing MMD-critic implementation.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        k_per_class: Number of prototypes per class
        gamma: Kernel bandwidth parameter
    
    Returns:
        prototypes_X: Selected prototype features
        prototypes_y: Selected prototype labels
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

def select_prototypes_fair_ot(X, y, sims, k_per_class=10, method='approx', regularization=0.01):
    """
    Select prototypes using Fair Optimal Transport method.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        k_per_class: Number of prototypes per class
        method: 'approx' or 'exact' for fairOT algorithm
        regularization: Regularization parameter for entropic OT
    
    Returns:
        prototypes_X: Selected prototype features
        prototypes_y: Selected prototype labels
    """
    classes = np.unique(y)
    n_source = sims.shape[0]
    total_prototypes = min(len(classes) * k_per_class, n_source)
    
    print(f"Using Fair OT ({method}) to select {total_prototypes} prototypes...")
    
    # Initialize Fair OT selector
    fair_ot = FairOptimalTransport(regularization=regularization)
    
    # Select prototypes using Fair OT
    sims = torch.from_numpy(sims).to("cuda")
    selected_indices, objectives = fair_ot.prototype_selection(sims, total_prototypes, method=method)
    
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
    topt = lambda x: torch.from_numpy(x).to("cuda")
    dist, sims = evaluation.compute_cost_matrix(topt(X), topt(target_X), metric = "cosine", return_sims=True)
    dist, sims = dist.cpu().numpy(), sims.cpu().numpy()
    if method == 'uniform':
        return prototype_selection_uniform(X, y, k_per_class)
    elif method == 'mmd_critic':
        return select_prototypes_mmd_critic(X, y, k_per_class)
    elif method == 'fairot_approx':
        return select_prototypes_fair_ot(X, y, sims, k_per_class, method='approx')
    elif method == 'fairot_exact':
        with torch.no_grad():
            return select_prototypes_fair_ot(X, y, sims, k_per_class, method='exact')
    
    # Use SPOTgreedy for prototype selection
    classes = np.unique(y)
    total_prototypes = min(len(classes) * k_per_class, len(X))
    
    # Create target distribution (uniform across all classes)
    target_marginal = np.ones(len(target_X)) / len(target_X)
    
    device = "cuda"
    target_marginal_torch = torch.from_numpy(target_marginal).float().to(device)
    # Convert distance matrix to torch tensor for SPOTgreedy
    dist_torch = torch.from_numpy(dist).float().to(device)
    
    print(f"Using SPOTgreedy to select {total_prototypes} prototypes...")
    # Use SPOTgreedy to select prototypes
    selected_indices = SPOT_GreedySubsetSelection(dist_torch, target_marginal_torch, total_prototypes)
    selected_indices = selected_indices.cpu().numpy()
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    return prototypes_X, prototypes_y

# Add new function to handle total prototype count selection
def prototype_selection_with_count(X, y, target_X, target_y, total_prototypes=50, method='spotgreedy'):
    """
    Select a specific total number of prototypes regardless of class distribution.
    
    Args:
        X: Source feature matrix (numpy array)
        y: Source labels (numpy array)
        target_X: Target feature matrix (numpy array)
        target_y: Target labels (numpy array)
        total_prototypes: Total number of prototypes to select
        method: Selection method ('spotgreedy', 'mmd_critic', 'uniform', etc.)
    
    Returns:
        prototypes_X: Selected prototype features
        prototypes_y: Selected prototype labels
    """
    print(f"DEBUG: prototype_selection_with_count called with {total_prototypes} prototypes, method={method}")
    
    if method == 'uniform':
        # Random selection of total_prototypes
        indices = np.random.choice(len(X), size=min(total_prototypes, len(X)), replace=False)
        print(f"DEBUG: Uniform selection returned {len(indices)} prototypes")
        return X[indices], y[indices]
    elif method == 'mmd_critic':
        return select_prototypes_mmd_critic_count(X, y, total_prototypes)
    
    # For methods that need similarity/distance matrices
    print(f"DEBUG: Computing similarity matrix for {len(X)} source samples and {len(target_X)} target samples")
    topt = lambda x: torch.from_numpy(x).to("cuda")
    dist, sims = evaluation.compute_cost_matrix(topt(X), topt(target_X), metric = "cosine", return_sims=True)
    dist, sims = dist.cpu().numpy(), sims.cpu().numpy()
    print(f"DEBUG: Computed similarity matrix shape: {sims.shape}")
    
    if method == 'fairot_approx':
        return select_prototypes_fair_ot_count(X, y, sims, total_prototypes, method='approx')
    elif method == 'fairot_exact':
        with torch.no_grad():
            return select_prototypes_fair_ot_count(X, y, sims, total_prototypes, method='exact')
    
    # Use SPOTgreedy for prototype selection
    total_prototypes = min(total_prototypes, len(X))
    

    # Create target distribution (uniform across all classes)
    target_marginal = np.ones(len(target_X)) / len(target_X)
    
    device = "cuda"
    target_marginal_torch = torch.from_numpy(target_marginal).float().to(device)
    # Convert distance matrix to torch tensor for SPOTgreedy
    dist_torch = torch.from_numpy(dist).float().to(device)
    
    print(f"Using SPOTgreedy to select {total_prototypes} prototypes...")
    # Use SPOTgreedy to select prototypes
    selected_indices = SPOT_GreedySubsetSelection(dist_torch, target_marginal_torch, total_prototypes)
    selected_indices = selected_indices.cpu().numpy()
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    print(f"DEBUG: SPOTgreedy returned {len(prototypes_X)} prototypes")
    return prototypes_X, prototypes_y

def select_prototypes_mmd_critic_count(X, y, total_prototypes=50, gamma=None):
    """
    Select specific total number of prototypes using MMD-critic.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        total_prototypes: Total number of prototypes to select
        gamma: Kernel bandwidth parameter
    
    Returns:
        prototypes_X: Selected prototype features
        prototypes_y: Selected prototype labels
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

def select_prototypes_fair_ot_count(X, y, sims, total_prototypes=50, method='approx', regularization=0.01):
    """
    Select specific total number of prototypes using Fair OT.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        sims: Similarity matrix (numpy array)
        total_prototypes: Total number of prototypes to select
        method: 'approx' or 'exact' for fairOT algorithm
        regularization: Regularization parameter for entropic OT
    
    Returns:
        prototypes_X: Selected prototype features
        prototypes_y: Selected prototype labels
    """
    total_prototypes = min(total_prototypes, len(X))
    
    print(f"Using Fair OT ({method}) to select {total_prototypes} prototypes...")
    print(f"DEBUG: Input sims shape: {sims.shape}, X shape: {X.shape}")
    
    # Add some randomness to break potential caching/determinism
    np.random.seed(None)  # Reset seed to ensure randomness
    
    # Initialize Fair OT selector with varied regularization
    fair_ot = FairOptimalTransport(regularization=regularization)
    
    # Convert sims to torch tensor if it's not already
    if isinstance(sims, np.ndarray):
        sims_torch = torch.from_numpy(sims).float().to("cuda")
    else:
        sims_torch = sims.to("cuda")
    
    print(f"DEBUG: sims_torch shape: {sims_torch.shape}")
    
    try:
        # Select prototypes using Fair OT
        selected_indices, objectives = fair_ot.prototype_selection(sims_torch, total_prototypes, method=method)
        print(f"DEBUG: Fair OT returned indices type: {type(selected_indices)}")
        
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
            print(f"WARNING: Fair OT only selected {len(selected_indices)} unique prototypes, need {total_prototypes}")
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
        print(f"ERROR in Fair OT selection: {e}")
        print("Falling back to random selection")
        selected_indices = np.random.choice(len(X), total_prototypes, replace=False)
    
    if len(selected_indices) != total_prototypes:
        print(f"Warning: Fair OT selected {len(selected_indices)} prototypes instead of requested {total_prototypes}")
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    print(f"DEBUG: Returning {len(prototypes_X)} prototypes")
    return prototypes_X, prototypes_y

def evaluate_1nn(P_X, P_y, target_X, target_y):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(P_X, P_y)
    pred = clf.predict(target_X)
    acc = accuracy_score(target_y, pred)
    return acc

def plot_class_histogram(target_y, classes, skew_percent, dataset_name, method, run_idx):
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

def plot_accuracy_curve_comparison(prototype_counts, all_method_accuracies, dataset_name, skew_percent):
    """
    Plot accuracy curves for all methods on the same plot for comparison.
    
    Args:
        prototype_counts: List of prototype counts tested
        all_method_accuracies: Dict with method names as keys and accuracy lists as values
        dataset_name: Name of the dataset
        skew_percent: Skew percentage
    """
    print(f"DEBUG: Plotting accuracy curves for {dataset_name} - {skew_percent}% skew")
    print(f"DEBUG: Prototype counts: {prototype_counts}")
    print(f"DEBUG: All method accuracies: {all_method_accuracies}")
    
    if len(prototype_counts) == 0:
        print("WARNING: No prototype counts to plot!")
        return
    
    if len(all_method_accuracies) == 0:
        print("WARNING: No method accuracies to plot!")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (method, accuracies) in enumerate(all_method_accuracies.items()):
        print(f"DEBUG: Method {method}: {len(accuracies)} accuracies: {accuracies}")
        if len(accuracies) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            x_vals = prototype_counts[:len(accuracies)]
            print(f"DEBUG: Plotting {method} with x_vals: {x_vals}, y_vals: {accuracies}")
            plt.plot(x_vals, accuracies, 
                    marker=marker, linewidth=2, markersize=8, 
                    label=method.upper(), color=color)
    
    plt.xlabel('Number of Prototypes', fontsize=12)
    plt.ylabel('1-NN Accuracy', fontsize=12)
    plt.title(f'{dataset_name} - Skew {skew_percent}% - Method Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Create directory for plots if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots_ablation')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    filename = f'{dataset_name}_skew{skew_percent}_method_comparison_accuracy_curve.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved method comparison plot: {filepath}")

def run_split_dataset_experiments(dataset_name, k_proto=10, skew_percent_list=[10, 30, 50, 70, 100], runs=5, methods=['spotgreedy', 'mmd_critic', 'uniform'], subsample=True):
    """
    Run experiments on datasets that have pre-split source and target pools.
    This includes Letter, USPS, ImageNet, and Flickr datasets.
    
    Args:
        dataset_name: Name of the dataset
        k_proto: Number of prototypes per class
        skew_percent_list: List of skew percentages to test
        runs: Number of experimental runs
        methods: List of prototype selection methods to compare
    
    Returns:
        results_summary: Dictionary with results for each method
    """
    print(f"Loading {dataset_name} dataset with pre-split protocol...")
    dataset_result = data.load_dataset(dataset_name)
    
    if dataset_result is None:
        print(f"Failed to load {dataset_name} dataset")
        return None
    
    # Check if dataset_result has the expected format
    try:
        (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
    except (TypeError, ValueError) as e:
        print(f"Error unpacking {dataset_name} dataset: {e}")
        return None
    
    # Convert to numpy arrays if they are pandas Series/DataFrames
    if hasattr(source_X, 'values'):
        source_X = source_X.values
    if hasattr(source_y, 'values'):
        source_y = source_y.values
    if hasattr(target_pool_X, 'values'):
        target_pool_X = target_pool_X.values
    if hasattr(target_pool_y, 'values'):
        target_pool_y = target_pool_y.values
    
    # Normalize the data
    scaler = StandardScaler()
    source_X = scaler.fit_transform(source_X.astype(np.float32))
    target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
    # Balance the source set to ensure uniform class representation
    source_X, source_y = balance_source_set(source_X, source_y)
    
    # For computational efficiency, subsample large datasets
    if subsample and len(source_X) > 5000:
        print(f"Subsampling {dataset_name} source set from {len(source_X)} to 5000 for efficiency...")
        source_indices = np.random.choice(len(source_X), 5000, replace=False)
        source_X = source_X[source_indices]
        source_y = source_y[source_indices]
    
    if subsample and len(target_pool_X) > 10000:
        print(f"Subsampling {dataset_name} target pool from {len(target_pool_X)} to 10000 for efficiency...")
        target_indices = np.random.choice(len(target_pool_X), 10000, replace=False)
        target_pool_X = target_pool_X[target_indices]
        target_pool_y = target_pool_y[target_indices]
    
    print(f"Final dataset sizes - Source: {len(source_X)}, Target pool: {len(target_pool_X)}")
    
    results_summary = {}
    
    for skew_percent in skew_percent_list:
        print(f"\n{'='*70}")
        print(f"Testing with {skew_percent}% skew across all methods")
        print(f"{'='*70}")
        
        # Collect accuracy curves for all methods for this skew
        all_method_accuracy_curves = {}
        
        for method in methods:
            print(f"\n--- {method.upper()} Method ---")
            
            skew_results = []
            # Fix prototype counts range - use proper min/max/step
            num_classes = len(np.unique(source_y))
            min_protos = num_classes  # At least 1 per class
            max_protos = min(100, len(source_X) // 10)  # Reasonable upper bound
            #prototype_counts = list(range(min_protos, max_protos + 1, max(1, (max_protos - min_protos) // 4)))
            prototype_counts = [10,50,100,200,500,1000]  # Fixed counts for simplicity
            #if len(prototype_counts) < 5:  # Ensure we have at least 5 points
             #   prototype_counts = [min_protos + i * (max_protos - min_protos) // 4 for i in range(5)]
            print(f"DEBUG: Testing prototype counts: {prototype_counts}")
            method_accuracy_curves = []
            
            for run in range(runs):
                print(f"Run {run+1}/{runs} for {method} with {skew_percent}% skew")
                
                # Generate target set with specified skew from target pool
                target_X, target_y = generate_target_set_from_pool(target_pool_X, target_pool_y, 
                                                                    skew_percent, total_size=2000)
                
                # Plot histogram for class distribution in target set (only for first run and first method)
                if run == 0 and method == methods[0]:
                    classes = np.unique(target_y)
                    plot_class_histogram(target_y, classes, skew_percent, dataset_name, 'all_methods', run)
                
                # Test different prototype counts for accuracy curve
                run_accuracies = []
                for proto_count in prototype_counts:
                    try:
                        print(f"  Testing with {proto_count} prototypes...")
                        # Use total prototype count directly instead of k_per_class
                        prototypes_X, prototypes_y = prototype_selection_with_count(source_X, source_y, 
                                                                                   target_X, target_y, 
                                                                                   total_prototypes=proto_count, 
                                                                                   method=method)
                        
                        if len(prototypes_X) == 0:
                            print(f"    Warning: No prototypes selected for {proto_count} count. Skipping.")
                            run_accuracies.append(0.0)
                            continue
                            
                        accuracy = evaluate_1nn(prototypes_X, prototypes_y, target_X, target_y)
                        run_accuracies.append(accuracy)
                        print(f"    Selected {len(prototypes_X)} prototypes, accuracy: {accuracy:.4f}")
                    except Exception as e:
                        print(f"    Error with {proto_count} prototypes: {e}")
                        run_accuracies.append(0.0)
                
                method_accuracy_curves.append(run_accuracies)
                
                # Also run with original k_proto for main results
                print("Selecting prototypes from source set...")
                prototypes_X, prototypes_y = prototype_selection(source_X, source_y, 
                                                                target_X, target_y, 
                                                                k_per_class=k_proto, method=method)
                
                # Evaluate 1-NN accuracy
                accuracy = evaluate_1nn(prototypes_X, prototypes_y, target_X, target_y)
                skew_results.append(accuracy)
                
                print(f"  Accuracy: {accuracy:.4f}")
            
            # Store average accuracy curve for this method
            if method_accuracy_curves:
                avg_accuracies = np.mean(method_accuracy_curves, axis=0)
                all_method_accuracy_curves[method] = avg_accuracies
            
            results_summary[method] = results_summary.get(method, {})
            if skew_results:
                mean_acc = np.mean(skew_results)
                std_acc = np.std(skew_results)
                print(f"\n{skew_percent}% skew results: {mean_acc:.4f} ± {std_acc:.4f}")
                results_summary[method][f'skew_{skew_percent}'] = {'mean': mean_acc, 'std': std_acc}
        
        # Plot comparison of all methods for this skew level
        if all_method_accuracy_curves:
            plot_accuracy_curve_comparison(prototype_counts, all_method_accuracy_curves, dataset_name, skew_percent)
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"{dataset_name} COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for skew_percent in skew_percent_list:
        print(f"\n{skew_percent}% SKEW:")
        for method in methods:
            if method in results_summary and f'skew_{skew_percent}' in results_summary[method]:
                result = results_summary[method][f'skew_{skew_percent}']
                print(f"  {method:12}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    return results_summary

def generate_target_set_from_pool(pool_X, pool_y, skew_percent, total_size=2000):
    """
    Generate target set from a pool following the skew protocol.
    Similar to MNIST protocol but works with any dataset.
    
    Args:
        pool_X: Pool of target features
        pool_y: Pool of target labels
        skew_percent: Percentage of target set from the skewed class
        total_size: Total size of target set
    
    Returns:
        target_X, target_y: Generated target set
    """
    # Convert to numpy arrays if they are pandas Series/DataFrames
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values
    
    classes = np.unique(pool_y)
    skew_class = np.random.choice(classes)
    
    # Count instances per class in the pool
    class_counts = {}
    for cls in classes:
        class_counts[cls] = np.sum(pool_y == cls)
    
    min_class_count = min(class_counts.values())
    
    if skew_percent == 10:
        # For 10% skew, use balanced approach
        target_size = min(total_size, min_class_count * len(classes))
        samples_per_class = target_size // len(classes)
        samples_skew_class = samples_per_class
        samples_per_other_class = samples_per_class
    else:
        # For higher skew percentages
        skew_class_count = class_counts[skew_class]
        max_skew_samples = min(skew_class_count, int(total_size * skew_percent / 100))
        
        samples_skew_class = max_skew_samples
        remaining_samples = total_size - samples_skew_class
        samples_per_other_class = remaining_samples // (len(classes) - 1)
    
    target_indices = []
    
    # Add samples from skewed class
    skew_idx = np.where(pool_y == skew_class)[0]
    if len(skew_idx) > 0:
        selected_skew = np.random.choice(skew_idx, size=min(samples_skew_class, len(skew_idx)), replace=False)
        target_indices.extend(selected_skew)
    
    # Add samples from other classes
    for cls in classes:
        if cls == skew_class:
            continue
        cls_idx = np.where(pool_y == cls)[0]
        if len(cls_idx) > 0:
            selected_cls = np.random.choice(cls_idx, size=min(samples_per_other_class, len(cls_idx)), replace=False)
            target_indices.extend(selected_cls)
    
    target_indices = np.array(target_indices)
    
    # Ensure all indices are valid
    valid_indices = target_indices[target_indices < len(pool_X)]
    if len(valid_indices) < len(target_indices):
        print(f"Warning: Removed {len(target_indices) - len(valid_indices)} invalid indices")
        target_indices = valid_indices
    
    np.random.shuffle(target_indices)
    
    if len(target_indices) == 0:
        raise ValueError("No valid target indices generated")
    
    actual_skew_percentage = np.sum(pool_y[target_indices] == skew_class) / len(target_indices) * 100
    print(f"Generated target set: {len(target_indices)} samples, actual skew: {actual_skew_percentage:.1f}% towards class {skew_class}")
    
    return pool_X[target_indices], pool_y[target_indices]


def run_mnist_experiments(k_proto=10, skew_percent_list=[10, 30, 50, 70, 100], runs=5, methods=['spotgreedy', 'mmd_critic', 'uniform']):
    """
    Run MNIST experiments following the paper's protocol.
    
    Args:
        k_proto: Number of prototypes per class
        skew_percent_list: List of skew percentages to test
        runs: Number of experimental runs
        methods: List of prototype selection methods to compare
    
    Returns:
        results_summary: Dictionary with results for each method
    """
    print(f"Loading MNIST dataset with paper-specific protocol...")
    dataset_result = data.load_dataset('mnist')
    
    if dataset_result is None:
        print(f"Failed to load MNIST dataset")
        return None
    
    # For MNIST, we expect the full dataset
    if len(dataset_result) == 2:
        # Standard split dataset format
        (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
    else:
        # Single dataset format - split it ourselves
        X, y = dataset_result
        # Use standard MNIST train/test split proportions
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
    
    # Convert to numpy arrays if they are pandas Series/DataFrames
    if hasattr(source_X, 'values'):
        source_X = source_X.values
    if hasattr(source_y, 'values'):
        source_y = source_y.values
    if hasattr(target_pool_X, 'values'):
        target_pool_X = target_pool_X.values
    if hasattr(target_pool_y, 'values'):
        target_pool_y = target_pool_y.values
    
    # Normalize the data
    scaler = StandardScaler()
    source_X = scaler.fit_transform(source_X.astype(np.float32))
    target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
    # Balance the source set to ensure uniform class representation
    source_X, source_y = balance_source_set(source_X, source_y)
    
    print(f"MNIST dataset sizes - Source: {len(source_X)}, Target pool: {len(target_pool_X)}")
    
    results_summary = {}
    
    for skew_percent in skew_percent_list:
        print(f"\n{'='*70}")
        print(f"Testing with {skew_percent}% skew across all methods")
        print(f"{'='*70}")
        
        # Collect accuracy curves for all methods for this skew
        all_method_accuracy_curves = {}
        
        for method in methods:
            print(f"\n--- {method.upper()} Method ---")
            
            skew_results = []
            # Fix prototype counts range for MNIST (10 classes)
            prototype_counts = [10,50,100,200,500,1000]  # Fixed counts for simplicity
            print(f"DEBUG: Testing prototype counts: {prototype_counts}")
            method_accuracy_curves = []
            
            for run in range(runs):
                print(f"Run {run+1}/{runs} for {method} with {skew_percent}% skew")
                
                # Generate target set with specified skew from target pool
                target_X, target_y = generate_mnist_target_set(target_pool_X, target_pool_y, 
                                                             skew_percent, total_size=2000)
                
                # Plot histogram for class distribution in target set (only for first run and first method)
                if run == 0 and method == methods[0]:
                    classes = np.unique(target_y)
                    plot_class_histogram(target_y, classes, skew_percent, 'MNIST', 'all_methods', run)
                
                # Test different prototype counts for accuracy curve
                run_accuracies = []
                for proto_count in prototype_counts:
                    try:
                        print(f"  Testing with {proto_count} prototypes...")
                        prototypes_X, prototypes_y = prototype_selection_with_count(source_X, source_y, 
                                                                                   target_X, target_y, 
                                                                                   total_prototypes=proto_count,
                                                                                   method=method)
                        
                        if len(prototypes_X) == 0:
                            run_accuracies.append(0.0)
                            continue
                            
                        accuracy = evaluate_1nn(prototypes_X, prototypes_y, target_X, target_y)
                        run_accuracies.append(accuracy)
                        print(f"    Selected {len(prototypes_X)} prototypes, accuracy: {accuracy:.4f}")
                    except Exception as e:
                        print(f"    Error with {proto_count} prototypes: {e}")
                        run_accuracies.append(0.0)
                
                method_accuracy_curves.append(run_accuracies)
                
                # Also run with original k_proto for main results
                print("Selecting prototypes from source set...")
                prototypes_X, prototypes_y = prototype_selection(source_X, source_y, 
                                                                target_X, target_y, 
                                                                k_per_class=k_proto, method=method)
                
                # Evaluate 1-NN accuracy
                accuracy = evaluate_1nn(prototypes_X, prototypes_y, target_X, target_y)
                skew_results.append(accuracy)
                
                print(f"  Accuracy: {accuracy:.4f}")
            
            # Store average accuracy curve for this method
            if method_accuracy_curves:
                avg_accuracies = np.mean(method_accuracy_curves, axis=0)
                all_method_accuracy_curves[method] = avg_accuracies
            
            results_summary[method] = results_summary.get(method, {})
            if skew_results:
                mean_acc = np.mean(skew_results)
                std_acc = np.std(skew_results)
                print(f"\n{skew_percent}% skew results: {mean_acc:.4f} ± {std_acc:.4f}")
                results_summary[method][f'skew_{skew_percent}'] = {'mean': mean_acc, 'std': std_acc}
        
        # Plot comparison of all methods for this skew level
        if all_method_accuracy_curves:
            plot_accuracy_curve_comparison(prototype_counts, all_method_accuracy_curves, 'MNIST', skew_percent)
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"MNIST COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for skew_percent in skew_percent_list:
        print(f"\n{skew_percent}% SKEW:")
        for method in methods:
            if method in results_summary and f'skew_{skew_percent}' in results_summary[method]:
                result = results_summary[method][f'skew_{skew_percent}']
                print(f"  {method:12}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    return results_summary

def generate_mnist_target_set(pool_X, pool_y, skew_percent, total_size=2000):
    """
    Generate MNIST target set following the skew protocol:
    - Class 0: skew_percent% of target set
    - Classes 1-9: (100 - skew_percent)% / 9 each of target set
    
    Args:
        pool_X: Pool of target features
        pool_y: Pool of target labels
        skew_percent: Percentage of target set from the skewed class (class 0)
        total_size: Total size of target set
    
    Returns:
        target_X, target_y: Generated target set
    """
    # Convert to numpy arrays if they are pandas Series/DataFrames
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values
    
    classes = np.unique(pool_y)
    
    # Calculate number of samples for each class in the target set
    skew_class_samples = int(total_size * skew_percent / 100)
    other_class_samples = (total_size - skew_class_samples) // (len(classes) - 1)
    
    target_indices = []
    
    # Add samples from class 0 (skewed class)
    class_0_indices = np.where(pool_y == 0)[0]
    if len(class_0_indices) > 0:
        selected_0 = np.random.choice(class_0_indices, size=min(skew_class_samples, len(class_0_indices)), replace=False)
        target_indices.extend(selected_0)
    
    # Add samples from other classes
    for cls in classes[1:]:
        cls_indices = np.where(pool_y == cls)[0]
        if len(cls_indices) > 0:
            selected_cls = np.random.choice(cls_indices, size=min(other_class_samples, len(cls_indices)), replace=False)
            target_indices.extend(selected_cls)
    
    target_indices = np.array(target_indices)
    
    # Ensure all indices are valid
    valid_indices = target_indices[target_indices < len(pool_X)]
    if len(valid_indices) < len(target_indices):
        print(f"Warning: Removed {len(target_indices) - len(valid_indices)} invalid indices")
        target_indices = valid_indices
    
    np.random.shuffle(target_indices)
    
    if len(target_indices) == 0:
        raise ValueError("No valid target indices generated")
    
    actual_skew_percentage = np.sum(pool_y[target_indices] == 0) / len(target_indices) * 100
    print(f"Generated MNIST target set: {len(target_indices)} samples, actual skew: {actual_skew_percentage:.1f}% towards class 0")
    
    return pool_X[target_indices], pool_y[target_indices]

def run_mnist_ablation_experiments(k_proto=10, runs=3, methods=['spotgreedy', 'mmd_critic', 'fairot_approx']):
    """
    Run MNIST ablation experiments with specific class distribution:
    - Source: 10% per class (balanced)
    - Target: Class 0 = 1%, Classes 1-9 = 11% each
    
    Args:
        k_proto: Number of prototypes per class
        runs: Number of experimental runs
        methods: List of prototype selection methods to compare
    
    Returns:
        results_summary: Dictionary with detailed results for each method
    """
    print(f"Loading MNIST dataset for ablation study...")
    dataset_result = data.load_dataset('mnist')
    
    if dataset_result is None:
        print(f"Failed to load MNIST dataset")
        return None
    
    # For MNIST, we expect the full dataset
    if len(dataset_result) == 2:
        # Standard split dataset format
        (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
    else:
        # Single dataset format - split it ourselves
        X, y = dataset_result
        # Use standard MNIST train/test split proportions
        source_X, target_pool_X, source_y, target_pool_y = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
    
    # Convert to numpy arrays if they are pandas Series/DataFrames
    if hasattr(source_X, 'values'):
        source_X = source_X.values
    if hasattr(source_y, 'values'):
        source_y = source_y.values
    if hasattr(target_pool_X, 'values'):
        target_pool_X = target_pool_X.values
    if hasattr(target_pool_y, 'values'):
        target_pool_y = target_pool_y.values
    
    # Normalize the data
    scaler = StandardScaler()
    source_X = scaler.fit_transform(source_X.astype(np.float32))
    target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
    
    # Balance the source set to ensure uniform class representation and fixed size of 5000
    source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
    
    print(f"MNIST ablation dataset sizes - Source: {len(source_X)}, Target pool: {len(target_pool_X)}")
    print("Source distribution (balanced): 10% per class")
    print("Target distribution: Class 0 = 1%, Classes 1-9 = 11% each")
    
    # Prototype counts to test
    prototype_counts = [10,50,100,150,200,250,300]
    
    results_summary = {}
    all_method_results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing {method.upper()} method")
        print(f"{'='*70}")
        
        method_overall_accuracies = []
        method_class_0_accuracies = []
        method_detailed_results = []
        
        # Generate one target set for plotting distributions
        sample_target_X, sample_target_y = generate_mnist_target_set_ablation(target_pool_X, target_pool_y, 
                                                                          total_size=2000)
        
        # Plot source and target distributions (once at the beginning)
        print("Plotting source and target set distributions...")
        plot_source_target_histograms(source_y, sample_target_y, 'MNIST', 'ablation')
        
        for run in range(runs):
            print(f"\nRun {run+1}/{runs} for {method}")
            
            # Generate target set with ablation distribution
            target_X, target_y = generate_mnist_target_set_ablation(target_pool_X, target_pool_y, 
                                                                       total_size=2000)
            
            run_overall_accs = []
            run_class_0_accs = []
            run_detailed = []
            
            # Test different prototype counts
            for proto_count in prototype_counts:
                try:
                    print(f"  Testing with {proto_count} prototypes...")
                    
                    # Select prototypes
                    prototypes_X, prototypes_y = prototype_selection_with_count(source_X, source_y, 
                                                                               target_X, target_y, 
                                                                               total_prototypes=proto_count,
                                                                               method=method)
                    
                    if len(prototypes_X) == 0:
                        print(f"    Warning: No prototypes selected for {proto_count} count. Skipping.")
                        run_overall_accs.append(0.0)
                        run_class_0_accs.append(0.0)
                        continue
                    
                    # Plot prototype distribution for first run and specific prototype counts
                    if run == 0 and proto_count in [50, 150, 300]:
                        plot_prototype_distribution_histogram(prototypes_y, 'MNIST', method, proto_count)
                    
                    # Evaluate with detailed results
                    overall_acc, class_accs, class_counts = evaluate_1nn_detailed(prototypes_X, prototypes_y, 
                                                                                      target_X, target_y)
                    
                    run_overall_accs.append(overall_acc)
                    run_class_0_accs.append(class_accs.get(0, 0.0))
                    run_detailed.append({
                        'proto_count': proto_count,
                        'overall_acc': overall_acc,
                        'class_accs': class_accs,
                        'class_counts': class_counts
                    })
                    
                    print(f"    Selected {len(prototypes_X)} prototypes")
                    print(f"    Overall accuracy: {overall_acc:.4f}")
                    print(f"    Class 0 accuracy: {class_accs.get(0, 0.0):.4f} ({class_counts.get(0, 0)} samples)")
                    
                except Exception as e:
                    print(f"    Error with {proto_count} prototypes: {e}")
                    run_overall_accs.append(0.0)
                    run_class_0_accs.append(0.0)
            
            method_overall_accuracies.append(run_overall_accs)
            method_class_0_accuracies.append(run_class_0_accs)
            method_detailed_results.append(run_detailed)
        
        # Calculate averages across runs
        if method_overall_accuracies:
            avg_overall = np.mean(method_overall_accuracies, axis=0)
            avg_class_0 = np.mean(method_class_0_accuracies, axis=0)
            
            all_method_results[method] = {
                'overall_accuracies': avg_overall,
                'class_0_accuracies': avg_class_0,
                'detailed_results': method_detailed_results
            }
            
            results_summary[method] = {
                'avg_overall_accuracies': avg_overall.tolist(),
                'avg_class_0_accuracies': avg_class_0.tolist(),
                'prototype_counts': prototype_counts
            }
            
            print(f"\n{method.upper()} Summary:")
            for i, proto_count in enumerate(prototype_counts):
                if i < len(avg_overall):
                    print(f"  {proto_count} prototypes: Overall={avg_overall[i]:.4f}, Class 0={avg_class_0[i]:.4f}")
    
    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING ABLATION PLOTS")
    print(f"{'='*70}")
    
    plot_ablation_results(prototype_counts, all_method_results, 'MNIST')
    
    return results_summary, all_method_results

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
    
    # Target set histogram
    target_counts = [np.sum(target_y == cls) for cls in classes]
    colors = ['red' if cls == 0 else 'lightgreen' for cls in classes]  # Highlight class 0
    ax2.bar(classes, target_counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title(f'{dataset_name} Target Set Distribution\n(Class 0: 1%, Classes 1-9: 11% each)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add count labels and percentages on bars
    total_target = len(target_y)
    for i, count in enumerate(target_counts):
        percentage = (count / total_target) * 100
        ax2.text(classes[i], count + max(target_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{experiment_type}_source_target_distributions.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved source/target distribution plot: {filepath}")

def plot_prototype_distribution_histogram(prototypes_y, dataset_name, method, proto_count):
    """
    Plot histogram showing class distribution in selected prototypes.
    
    Args:
        prototypes_y: Prototype labels
        dataset_name: Name of the dataset
        method: Prototype selection method
        proto_count: Number of prototypes
    """
    classes = np.unique(prototypes_y)
    prototype_counts = [np.sum(prototypes_y == cls) for cls in classes]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if cls == 0 else 'lightblue' for cls in classes]  # Highlight class 0
    bars = plt.bar(classes, prototype_counts, color=colors, alpha=0.7, edgecolor='black')
    
    plt.title(f'{dataset_name} - {method.upper()}\nPrototype Distribution ({proto_count} total prototypes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Prototypes Selected', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(prototype_counts):
        percentage = (count / len(prototypes_y)) * 100
        plt.text(classes[i], count + max(prototype_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{method}_prototypes_{proto_count}_distribution.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved prototype distribution plot: {filepath}")

def generate_ablation_target_set_other(pool_X, pool_y, minority_class, total_size=2000):
    """
    Generate ablation target set for non-MNIST datasets:
    - minority_class: 1% of target set
    - other classes: remaining samples split evenly

    Args:
        pool_X: Pool of target features
        pool_y: Pool of target labels
        minority_class: The class to be assigned 1% of the target set
        total_size: Total size of target set

    Returns:
        target_X, target_y: Generated target set
    """
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values

    classes = np.unique(pool_y)
    target_indices = []

    # Minority class: 1%
    minority_samples = int(total_size * 0.01)
    min_idx = np.where(pool_y == minority_class)[0]
    if len(min_idx) >= minority_samples:
        selected_min = np.random.choice(min_idx, size=minority_samples, replace=False)
        target_indices.extend(selected_min)
    else:
        target_indices.extend(min_idx)
        print(f"Warning: Only {len(min_idx)} samples available for class {minority_class}, needed {minority_samples}")

    # Majority classes: evenly split remaining samples
    majority_classes = [cls for cls in classes if cls != minority_class]
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
    for cls in classes:
        count = np.sum(final_target_y == cls)
        percentage = (count / len(target_indices)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")

    return pool_X[target_indices], pool_y[target_indices]

if __name__ == "__main__":
    datasets = [
        ("adult", 1),   # minority_class=1 (change as needed)
        ("crime", 1),   # minority_class=1 (change as needed)
        ("drug", 1),    # minority_class=1 (change as needed)
        ("credit", 1),  # minority_class=1 (change as needed)
    ]
    for dataset_name, minority_class in datasets:
        print(f"\n{'='*80}")
        print(f"Running ablation experiment for {dataset_name.upper()} (minority class: {minority_class})")
        print(f"{'='*80}")
        dataset_result = data.load_dataset(dataset_name)
        if dataset_result is None:
            print(f"Failed to load {dataset_name} dataset")
            continue
        (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
        scaler = StandardScaler()
        source_X = scaler.fit_transform(source_X.astype(np.float32))
        target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
        source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
        target_X, target_y = generate_ablation_target_set_other(target_pool_X, target_pool_y, minority_class=minority_class, total_size=2000)
        plot_source_target_histograms(source_y, target_y, dataset_name, experiment_type='ablation')
        # ...run your ablation experiments and evaluation as needed...
        scaler = StandardScaler()
        source_X = scaler.fit_transform(source_X.astype(np.float32))
        target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
        source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
        target_X, target_y = generate_ablation_target_set_other(target_pool_X, target_pool_y, minority_class=minority_class, total_size=2000)
        plot_source_target_histograms(source_y, target_y, dataset_name, experiment_type='ablation')
        # ...run your ablation experiments and evaluation as needed...
    print(f"Saved source/target distribution plot: {filepath}")

def plot_prototype_distribution_histogram(prototypes_y, dataset_name, method, proto_count):
    """
    Plot histogram showing class distribution in selected prototypes.
    
    Args:
        prototypes_y: Prototype labels
        dataset_name: Name of the dataset
        method: Prototype selection method
        proto_count: Number of prototypes
    """
    classes = np.unique(prototypes_y)
    prototype_counts = [np.sum(prototypes_y == cls) for cls in classes]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if cls == 0 else 'lightblue' for cls in classes]  # Highlight class 0
    bars = plt.bar(classes, prototype_counts, color=colors, alpha=0.7, edgecolor='black')
    
    plt.title(f'{dataset_name} - {method.upper()}\nPrototype Distribution ({proto_count} total prototypes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Prototypes Selected', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(prototype_counts):
        percentage = (count / len(prototypes_y)) * 100
        plt.text(classes[i], count + max(prototype_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{method}_prototypes_{proto_count}_distribution.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved prototype distribution plot: {filepath}")
    plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f'{dataset_name}_{method}_prototypes_{proto_count}_distribution.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved prototype distribution plot: {filepath}")

def generate_ablation_target_set_other(pool_X, pool_y, minority_class, total_size=2000):
    """
    Generate ablation target set for non-MNIST datasets:
    - minority_class: 1% of target set
    - other classes: remaining samples split evenly

    Args:
        pool_X: Pool of target features
        pool_y: Pool of target labels
        minority_class: The class to be assigned 1% of the target set
        total_size: Total size of target set

    Returns:
        target_X, target_y: Generated target set
    """
    if hasattr(pool_X, 'values'):
        pool_X = pool_X.values
    if hasattr(pool_y, 'values'):
        pool_y = pool_y.values

    classes = np.unique(pool_y)
    target_indices = []

    # Minority class: 1%
    minority_samples = int(total_size * 0.01)
    min_idx = np.where(pool_y == minority_class)[0]
    if len(min_idx) >= minority_samples:
        selected_min = np.random.choice(min_idx, size=minority_samples, replace=False)
        target_indices.extend(selected_min)
    else:
        target_indices.extend(min_idx)
        print(f"Warning: Only {len(min_idx)} samples available for class {minority_class}, needed {minority_samples}")

    # Majority classes: evenly split remaining samples
    majority_classes = [cls for cls in classes if cls != minority_class]
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
    for cls in classes:
        count = np.sum(final_target_y == cls)
        percentage = (count / len(target_indices)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")

    return pool_X[target_indices], pool_y[target_indices]

if __name__ == "__main__":
    datasets = [
        ("adult", 1),   # minority_class=1 (change as needed)
        ("crime", 1),   # minority_class=1 (change as needed)
        ("drug", 1),    # minority_class=1 (change as needed)
        ("credit", 1),  # minority_class=1 (change as needed)
    ]
    for dataset_name, minority_class in datasets:
        print(f"\n{'='*80}")
        print(f"Running ablation experiment for {dataset_name.upper()} (minority class: {minority_class})")
        print(f"{'='*80}")
        dataset_result = data.load_dataset(dataset_name)
        if dataset_result is None:
            print(f"Failed to load {dataset_name} dataset")
            continue
        (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
        scaler = StandardScaler()
        source_X = scaler.fit_transform(source_X.astype(np.float32))
        target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
        source_X, source_y = balance_source_set(source_X, source_y, target_total_size=5000)
        target_X, target_y = generate_ablation_target_set_other(target_pool_X, target_pool_y, minority_class=minority_class, total_size=2000)
        plot_source_target_histograms(source_y, target_y, dataset_name, experiment_type='ablation')
        # ...run your ablation experiments and evaluation as needed...
