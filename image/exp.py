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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMDcritic'))
from baselines.MMDcritic.mmd_critic import Dataset, select_prototypes
from baselines.spotgreedy import SPOT_GreedySubsetSelection
from proto_selection_evals import data
from fairOT import FairOptimalTransport
from proto_selection_evals import evaluation

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

def select_prototypes_fair_ot(X, y, sims, k_per_class=10, method='approx', regularization=0.05):
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
    selected_indices = selected_indices.cpu().numpy()
    
    # Convert to numpy array
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
    target_marginal = np.ones(len(X)) / len(X)
    
    device = "cuda"
    target_marginal_torch = torch.from_numpy(target_marginal).float().to(device)
    
    print(f"Using SPOTgreedy to select {total_prototypes} prototypes...")
    # Use SPOTgreedy to select prototypes
    selected_indices = SPOT_GreedySubsetSelection(dist, target_marginal_torch, total_prototypes)
    selected_indices = selected_indices.cpu().numpy()
    
    prototypes_X = X[selected_indices]
    prototypes_y = y[selected_indices]
    
    return prototypes_X, prototypes_y

def evaluate_1nn(P_X, P_y, target_X, target_y):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(P_X, P_y)
    pred = clf.predict(target_X)
    acc = accuracy_score(target_y, pred)
    return acc

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
    
    (source_X, source_y), (target_pool_X, target_pool_y) = dataset_result
    
    # Normalize the data
    scaler = StandardScaler()
    source_X = scaler.fit_transform(source_X.astype(np.float32))
    target_pool_X = scaler.transform(target_pool_X.astype(np.float32))
    
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
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"{dataset_name} Experiments with {method.upper()} prototype selection")
        print(f"{'='*70}")
        
        # Select prototypes from source set using specified method
        # print("Selecting prototypes from source set...")
        # prototypes_X, prototypes_y = prototype_selection(source_X, source_y, target_X, target_y, k_per_class=k_proto, method=method)
        
        method_results = {}
        
        for skew_percent in skew_percent_list:
            print(f"\n--- Testing with {skew_percent}% skew ---")
            
            skew_results = []
            
            for run in range(runs):
                print(f"Run {run+1}/{runs} for {skew_percent}% skew")
                
                # Generate target set with specified skew from target pool
                target_X, target_y = generate_target_set_from_pool(target_pool_X, target_pool_y, 
                                                                    skew_percent, total_size=2000)
                print("Selecting prototypes from source set...")
                prototypes_X, prototypes_y = prototype_selection(source_X, source_y, 
                                                                    target_X, target_y, 
                                                                    k_per_class=k_proto, method=method)
                
                # Evaluate 1-NN accuracy
                accuracy = evaluate_1nn(prototypes_X, prototypes_y, target_X, target_y)
                skew_results.append(accuracy)
                
                print(f"  Accuracy: {accuracy:.4f}")
            
            if skew_results:
                mean_acc = np.mean(skew_results)
                std_acc = np.std(skew_results)
                print(f"\n{skew_percent}% skew results: {mean_acc:.4f} ± {std_acc:.4f}")
                method_results[f'skew_{skew_percent}'] = {'mean': mean_acc, 'std': std_acc}
        
        results_summary[method] = method_results
    
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


def run_experiments(dataset_name, total_target=1000, k_proto=10, skew_percent_list=[50, 70, 90], runs=10, methods=['spotgreedy', 'mmd_critic', 'uniform']):
    # Datasets with pre-split source/target pools
    # if dataset_name in ['Letter', 'USPS', 'tinyimagenet', 'Flickr']:
    if dataset_name in ['tinyimagenet', 'Flickr']:
        return run_split_dataset_experiments(dataset_name, k_proto=k_proto, runs=runs, methods=methods)
    
    raise NotImplementedError

if __name__ == "__main__":
    # Test with small datasets first and fewer runs for faster testing
    #methods_to_test = ['spotgreedy', 'mmd_critic', 'fairot_approx', 'fairot_exact', 'uniform']
    methods_to_test = ['fairot_exact']  # Reduced for faster testing
    print("Starting comprehensive protorSPOtype selection evaluation...")
    print(f"Methods to test: {', '.join(methods_to_test)}")
    print("=" * 80)
    
    all_results = {}
    # DATASETS = ['Letter', 'USPS', 'tinyimagenet', 'Flickr']
    DATASETS = ['tinyimagenet', 'Flickr']
    
    # Datasets with specific experimental protocols
    # protocol_datasets = ['Letter', 'USPS', 'tinyimagenet']
    protocol_datasets = ['tinyimagenet']
    for dataset in protocol_datasets:
        print(f"\n{'*'*80}")
        print(f"DATASET: {dataset} (Paper-specific Protocol)")
        print(f"{'*'*80}")
        
        results = run_experiments(
            dataset_name=dataset,
            runs=1,  # Reduced for faster testing
            methods=methods_to_test,
            k_proto=5  # Reduced for faster computation
        )
        all_results[dataset] = results
    
    # Optional: Test Flickr if available
    print(f"\n{'*'*80}")
    print(f"DATASET: Flickr (Optional - if file available)")
    print(f"{'*'*80}")
    
    flickr_results = run_experiments(
        dataset_name='Flickr',
        runs=1,
        methods=methods_to_test,
        k_proto=5
    )
    all_results['Flickr'] = flickr_results
    
    print(f"\n{'*'*80}")
    print("FINAL COMPREHENSIVE SUMMARY")
    print(f"{'*'*80}")
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        if results:
            if dataset in DATASETS:
                # Datasets with skew-based metrics
                for skew in [10, 30, 50, 70]:
                    metric = f'skew_{skew}'
                    if any(metric in results.get(method, {}) for method in methods_to_test):
                        print(f"  {metric}:")
                        for method in methods_to_test:
                            if method in results and metric in results[method]:
                                result = results[method][metric]
                                print(f"    {method:12}: {result['mean']:.4f}")
            else:
                # Standard datasets with balanced + skew metrics
                for metric in ['balanced', 'skew_50', 'skew_70', 'skew_90']:
                    if any(metric in results.get(method, {}) for method in methods_to_test):
                        print(f"  {metric}:")
                        for method in methods_to_test:
                            if method in results and metric in results[method]:
                                result = results[method][metric]
                                print(f"    {method:12}: {result['mean']:.4f}")
        else:
            print("  No results available or dataset not accessible")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    # Summary of best performing methods
    print("\nSUMMARY OF BEST PERFORMING METHODS:")
    for dataset, results in all_results.items():
        if results:
            print(f"\n{dataset}:")
            if dataset in DATASETS:
                # Find best method for each skew level
                for skew in [10, 30, 50, 70]:
                    metric = f'skew_{skew}'
                    best_method = None
                    best_score = -1
                    for method in methods_to_test:
                        if method in results and metric in results[method]:
                            score = results[method][metric]['mean']
                            if score > best_score:
                                best_score = score
                                best_method = method
                    if best_method:
                        print(f"  {metric}: {best_method} ({best_score:.4f})")
            else:
                # Standard datasets
                for metric in ['balanced', 'skew_50', 'skew_70', 'skew_90']:
                    best_method = None
                    best_score = -1
                    for method in methods_to_test:
                        if method in results and metric in results[method]:
                            score = results[method][metric]['mean']
                            if score > best_score:
                                best_score = score
                                best_method = method
                    if best_method:
                        print(f"  {metric}: {best_method} ({best_score:.4f})")
