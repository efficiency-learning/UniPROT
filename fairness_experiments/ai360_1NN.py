from aif360.datasets.adult_dataset import AdultDataset
from aif360.datasets.bank_dataset import BankDataset
from aif360.datasets.compas_dataset import CompasDataset
from aif360.datasets.german_dataset import GermanDataset
from aif360.datasets.meps_dataset_panel19_fy2015 import MEPSDataset19
from aif360.metrics import BinaryLabelDatasetMetric, DatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer

from IPython.display import Markdown, display

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

import json
from collections import OrderedDict

from fairness_experiments.setup_german_data import setup_german_credit_dataset
from fairness_experiments.setup_all_data import setup_all_datasets
from baselines.SPOTgreedy import SPOT_GreedySubsetSelection

dedicated_folder = "logs"
os.makedirs(dedicated_folder, exist_ok=True)
# Setup all required datasets
print("Setting up required datasets...")
setup_all_datasets()

# Load the dataset using AIF360
dataset = GermanDataset(
    protected_attribute_names=['age'],  # age is used as the protected attribute
    privileged_classes=[lambda x: x >= 25],  # age >= 25 is considered privileged
    features_to_drop=['personal_status', 'sex']  # Drop redundant features
)

dataset_orig = GermanDataset(protected_attribute_names=['age'],           # this dataset also contains protected
                                                                          # attribute for "sex" which we do not
                                                                          # consider in this evaluation
                             privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
                             features_to_drop=['personal_status', 'sex']) # ignore sex-related attributes

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]


print("Original one hot encoded german dataset shape: ",dataset_orig.features.shape)
print("Train dataset shape: ", dataset_orig_train.features.shape)
print("Test dataset shape: ", dataset_orig_test.features.shape)


df, dict_df = dataset_orig.convert_to_dataframe()

print("Shape: ", df.shape)
print(df.columns)
df.head(5)

print("Key: ", dataset_orig.metadata['protected_attribute_maps'][1])
df['age'].value_counts().plot(kind='bar')
plt.xlabel("Age (0 = under 25, 1 = over 25)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(dedicated_folder, "german_credit_age_distribution.png"))

print("Key: ", dataset_orig.metadata['label_maps'])
df['credit'].value_counts().plot(kind='bar')
plt.xlabel("Credit (1 = Good Credit, 2 = Bad Credit)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(dedicated_folder, "german_credit_class_distribution.png"))

# Import additional required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from FairOT.FairOptimalTransport import FairOptimalTransport

def load_dataset(dataset_name):
    """Load and preprocess different datasets"""
    if dataset_name == "meps19":
        dataset = MEPSDataset19()
        protected_attribute = 'RACE'
    elif dataset_name == "adult":
        dataset = AdultDataset()
        # Subsample Adult dataset to 2000 samples
        total_samples = len(dataset.features)
        subsample_idx = np.random.choice(total_samples, size=2000, replace=False)
        dataset.features = dataset.features[subsample_idx]
        dataset.labels = dataset.labels[subsample_idx]
        dataset.protected_attributes = dataset.protected_attributes[subsample_idx]
        protected_attribute = 'sex'
    elif dataset_name == "german":
        dataset = GermanDataset()
        protected_attribute = 'age'
    elif dataset_name == "compas":
        dataset = CompasDataset()
        protected_attribute = 'race'
    elif dataset_name == "bank":
        dataset = BankDataset()
        protected_attribute = 'age'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, protected_attribute

def evaluate_dataset(dataset_name, base_prototypes=50):
    """Run complete evaluation pipeline for a dataset"""
    print(f"\nEvaluating {dataset_name.upper()} dataset...")
    
    # Load dataset
    dataset_orig, protected_attribute = load_dataset(dataset_name)
    
    # Convert to numpy arrays
    X = dataset_orig.features
    y = (dataset_orig.labels.ravel() == 1).astype(int)
    protected = dataset_orig.protected_attributes[:, 0]
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Protected attribute: {protected_attribute}")
    
    # Calculate dataset-specific parameters
    total_samples = len(X)
    
    # Set subsample fraction based on dataset size (updated)
    if dataset_name == "adult":
        subsample_frac = 0.15
    elif total_samples > 10000:
        subsample_frac = 0.01
    elif total_samples > 5000:
        subsample_frac = 0.05
    else:
        subsample_frac = 0.15
    
    n_samples = int(subsample_frac * total_samples)
    n_prototypes = max(50, int(n_samples))
    n_prototypes = min(n_prototypes, n_samples)
    
    
    # Run each baseline for 100 seeds and collect results
    n_runs = 100
    seeds = list(range(42, 142))  # 100 unique seeds for robust evaluation
    results = {'fairot_eps': [], 'uniform': [], 'spotgreedy': []}

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    def train_evaluate_1nn(X_train, y_train, X_test, y_test, protected_test):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        # For binary classification, get probabilities for AUC
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        else:
            # fallback: use predictions as probabilities
            probs = preds
        auc = roc_auc_score(y_test, probs)
        p1 = preds[protected_test == 1].mean() if np.any(protected_test == 1) else 0.0
        p0 = preds[protected_test == 0].mean() if np.any(protected_test == 0) else 0.0
        dpd = abs(p1 - p0)
        return auc, dpd

    for run_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        # FairOT baseline
        device = torch.device('cpu')
        X_torch = torch.from_numpy(X).float().to(device)
        y_torch = torch.from_numpy(y).long().to(device)
        protected_torch = torch.from_numpy(protected).long().to(device)
        fair_ot = FairOptimalTransport(regularization=0.1)
        sims = torch.mm(X_torch, X_torch.t())
        sims = sims / torch.norm(sims, p=2)
        selected_indices_eps, _ = fair_ot.prototype_selection(sims, n_prototypes, method='approx', epsilon=0.001)
        if isinstance(selected_indices_eps, torch.Tensor):
            selected_indices_eps = selected_indices_eps.cpu().numpy()
        elif isinstance(selected_indices_eps, list):
            selected_indices_eps = np.array(selected_indices_eps)
        X_fairot = X[selected_indices_eps]
        y_fairot = y[selected_indices_eps]
        protected_fairot = protected[selected_indices_eps]
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X_fairot, y_fairot, protected_fairot, test_size=0.3, random_state=seed, stratify=y_fairot
        )
        auc_fairot_eps, dpd_fairot_eps = train_evaluate_1nn(X_train, y_train, X_test, y_test, prot_test)
        results['fairot_eps'].append((auc_fairot_eps, dpd_fairot_eps))

        # Uniform baseline
        uniform_indices = np.random.choice(len(X), size=n_prototypes, replace=False)
        X_uniform = X[uniform_indices]
        y_uniform = y[uniform_indices]
        protected_uniform = protected[uniform_indices]
        X_train_u, X_test_u, y_train_u, y_test_u, prot_train_u, prot_test_u = train_test_split(
            X_uniform, y_uniform, protected_uniform, test_size=0.3, random_state=seed, stratify=y_uniform
        )
        auc_uniform, dpd_uniform = train_evaluate_1nn(X_train_u, y_train_u, X_test_u, y_test_u, prot_test_u)
        results['uniform'].append((auc_uniform, dpd_uniform))

        # SpotGreedy baseline
        if isinstance(X, torch.Tensor):
            X_spot = X.cpu().numpy()
        else:
            X_spot = X
        spotgreedy_indices = SPOT_GreedySubsetSelection(X_spot, n_prototypes)
        if isinstance(spotgreedy_indices, torch.Tensor):
            spotgreedy_indices = spotgreedy_indices.cpu().numpy()
        X_spotgreedy = X[spotgreedy_indices]
        y_spotgreedy = y[spotgreedy_indices]
        protected_spotgreedy = protected[spotgreedy_indices]
        X_train_s, X_test_s, y_train_s, y_test_s, prot_train_s, prot_test_s = train_test_split(
            X_spotgreedy, y_spotgreedy, protected_spotgreedy, test_size=0.3, random_state=seed, stratify=y_spotgreedy
        )
        auc_spotgreedy, dpd_spotgreedy = train_evaluate_1nn(X_train_s, y_train_s, X_test_s, y_test_s, prot_test_s)
        results['spotgreedy'].append((auc_spotgreedy, dpd_spotgreedy))

    # Compute mean and variance for each baseline
    summary = {}
    for key in results:
        aucs = [x[0] for x in results[key]]
        dpds = [x[1] for x in results[key]]
        summary[key] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'dpd_mean': np.mean(dpds),
            'dpd_std': np.std(dpds),
            'num_prototypes': n_prototypes
        }

    # Plot mean and variance (shaded region)
    plt.figure(figsize=(6, 6))
    colors = ['purple', 'blue', 'green']
    labels = ['FairOT (epsilon)', 'Uniform', 'SpotGreedy']
    for i, key in enumerate(['fairot_eps', 'uniform', 'spotgreedy']):
        aucs = [x[0] for x in results[key]]
        dpds = [x[1] for x in results[key]]
        # Plot all individual points for each baseline with same color
        plt.scatter(dpds, aucs, c=colors[i], s=40, alpha=0.7, label=labels[i] if i == 0 else None)
    plt.xlabel('Demographic Parity Difference (lower is better)')
    plt.ylabel('AUC (higher is better)')
    plt.title(f'{dataset_name.upper()} Fairness-Utility Tradeoff (Scatter over 100 seeds)')
    plt.grid(True, alpha=0.3)
    plt.legend([labels[0], labels[1], labels[2]])
    plt.savefig(os.path.join(dedicated_folder, f"scatter_fairness_utility_{dataset_name}_all_baselines_new.png"))
    plt.close()

    # Print summary table
    print("\nSummary Results (Mean ± Std over 100 seeds):")
    print("-" * 50)
    print(f"{'Method':<18} {'AUC':<15} {'DPD':<15}")
    print("-" * 50)
    for key, label in zip(['fairot_eps', 'uniform', 'spotgreedy'], labels):
        auc_mean = summary[key]['auc_mean']
        auc_std = summary[key]['auc_std']
        dpd_mean = summary[key]['dpd_mean']
        dpd_std = summary[key]['dpd_std']
        print(f"{label:<18} {auc_mean:.4f} ± {auc_std:.4f}   {dpd_mean:.4f} ± {dpd_std:.4f}")

    return summary

# Evaluate all datasets
#datasets = ['bank', 'meps19']
#datasets = ['adult','german','compas','bank', 'meps19']
datasets = ['german']
#datasets = ['meps19']
results = {}

for dataset_name in datasets:
    results[dataset_name] = evaluate_dataset(dataset_name)
    print("Successful")
    # Plot all baselines for individual dataset
    plt.figure(figsize=(6, 6))
    metrics = results[dataset_name]
    dpds = [metrics['fairot_eps']['dpd_mean'], metrics['uniform']['dpd_mean'], metrics['spotgreedy']['dpd_mean']]
    aucs = [metrics['fairot_eps']['auc_mean'], metrics['uniform']['auc_mean'], metrics['spotgreedy']['auc_mean']]
    colors = ['purple', 'blue', 'green']
    # Get prototype counts for legend
    prototype_counts = [
        metrics['fairot_eps']['num_prototypes'],
        metrics['uniform']['num_prototypes'],
        metrics['spotgreedy']['num_prototypes']
    ]
    labels = [
        f'FairOT (epsilon) [{prototype_counts[0]}]',
        f'Uniform [{prototype_counts[1]}]',
        f'SpotGreedy [{prototype_counts[2]}]'
    ]
    for i in range(3):
        plt.scatter(dpds[i], aucs[i], c=colors[i], s=100, label=labels[i])
        plt.annotate(labels[i], (dpds[i], aucs[i]), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Demographic Parity Difference (lower is better)')
    plt.ylabel('AUC (higher is better)')
    plt.title(f'{dataset_name.upper()} Fairness-Utility Tradeoff')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(dedicated_folder, f"fairness_utility_{dataset_name}_all_baselines.png"))
    plt.close()

# Print summary table
print("\nSummary Results:")
print("-" * 50)
print(f"{'Dataset':<10} {'Method':<18} {'AUC':<8} {'DPD':<8}")
print("-" * 50)
for dataset in results:
    metrics_fairot = results[dataset]['fairot_eps']
    metrics_uniform = results[dataset]['uniform']
    metrics_spotgreedy = results[dataset]['spotgreedy']
    print(f"{dataset:<10} {'FairOT (epsilon)':<18} {metrics_fairot['auc_mean']:.4f} {metrics_fairot['dpd_mean']:.4f}")
    print(f"{dataset:<10} {'Uniform':<18} {metrics_uniform['auc_mean']:.4f} {metrics_uniform['dpd_mean']:.4f}")
    print(f"{dataset:<10} {'SpotGreedy':<18} {metrics_spotgreedy['auc_mean']:.4f} {metrics_spotgreedy['dpd_mean']:.4f}")

# Save results to file
with open(os.path.join(dedicated_folder, "all_datasets_results_fairot_eps.txt"), "w") as f:
    f.write("Fairness-Utility Analysis Across Datasets (FairOT epsilon)\n")
    f.write("=" * 50 + "\n\n")
    for dataset in results:
        f.write(f"\n{dataset.upper()} Dataset:\n")
        f.write("-" * 20 + "\n")
        metrics_fairot = results[dataset]['fairot_eps']
        metrics_uniform = results[dataset]['uniform']
        metrics_spotgreedy = results[dataset]['spotgreedy']
        f.write(f"FairOT (epsilon):\n")
        f.write(f"  AUC: {metrics_fairot['auc_mean']:.4f}\n")
        f.write(f"  DPD: {metrics_fairot['dpd_mean']:.4f}\n")
        f.write(f"  num_prototypes: {metrics_fairot['num_prototypes']}\n")
        f.write(f"Uniform Selection:\n")
        f.write(f"  AUC: {metrics_uniform['auc_mean']:.4f}\n")
        f.write(f"  DPD: {metrics_uniform['dpd_mean']:.4f}\n")
        f.write(f" num_prototypes: {metrics_uniform['num_prototypes']}\n")
        f.write(f"SpotGreedy Selection:\n")
        f.write(f"  AUC: {metrics_spotgreedy['auc_mean']:.4f}\n")
        f.write(f"  DPD: {metrics_spotgreedy['dpd_mean']:.4f}\n")
        f.write(f" num_prototypes: {metrics_spotgreedy['num_prototypes']}\n")
