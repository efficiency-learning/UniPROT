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

import json
from collections import OrderedDict

from FairOT.datasets.setup_german_data import setup_german_credit_dataset
from FairOT.datasets.setup_all_data import setup_all_datasets
from baselines.SPOTgreedy import SPOT_GreedySubsetSelection

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
plt.savefig("german_credit_age_distribution.png")

print("Key: ", dataset_orig.metadata['label_maps'])
df['credit'].value_counts().plot(kind='bar')
plt.xlabel("Credit (1 = Good Credit, 2 = Bad Credit)")
plt.ylabel("Frequency")
plt.savefig("german_credit_class_distribution.png")

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
    if dataset_name == "adult":
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
    elif dataset_name == "meps19":
        dataset = MEPSDataset19()
        protected_attribute = 'RACE'
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
    
    # ---------------------------
    # FairOT Selection (stochastic epsilon greedy)
    # ---------------------------
    print("\nRunning FairOT prototype selection (stochastic epsilon=0.1)...")
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).long()
    protected_torch = torch.from_numpy(protected).long()
    fair_ot = FairOptimalTransport(regularization=0.1, device='cpu')
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

    # ---------------------------
    # MLP Training and Evaluation
    # ---------------------------
    def train_evaluate_mlp(X_train, y_train, X_test, y_test, protected_test, hidden_dim=64, epochs=30):
        """Train MLP and evaluate fairness-utility metrics"""
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = (probs > 0.5).astype(int)
            auc = roc_auc_score(y_test.numpy(), probs)
            p1 = preds[protected_test == 1].mean()
            p0 = preds[protected_test == 0].mean()
            dpd = abs(p1 - p0)
        return auc, dpd

    # FairOT evaluation (stochastic epsilon greedy)
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X_fairot, y_fairot, protected_fairot, test_size=0.3, random_state=42, stratify=y_fairot
    )
    auc_fairot_eps, dpd_fairot_eps = train_evaluate_mlp(X_train, y_train, X_test, y_test, prot_test)

    # Print results
    print("\nResults:")
    print("FairOT Selection (stochastic epsilon):")
    print(f"  AUC (utility): {auc_fairot_eps:.4f}")
    print(f"  Demographic Parity Difference (fairness): {dpd_fairot_eps:.4f}")

    # Plot fairness-utility comparison
    plt.figure(figsize=(6, 6))
    plt.scatter([dpd_fairot_eps], [auc_fairot_eps], c=['purple'], s=100)
    plt.annotate('FairOT (epsilon)', (dpd_fairot_eps, auc_fairot_eps), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Demographic Parity Difference (lower is better)')
    plt.ylabel('AUC (higher is better)')
    plt.title('Fairness-Utility Tradeoff Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"fairness_utility_comparison_{dataset_name}_fairot_eps.png")
    plt.close()

    # Return metrics for comparison
    return {
        'fairot_eps': {'auc': auc_fairot_eps, 'dpd': dpd_fairot_eps}
    }

# Evaluate all datasets
datasets = ['adult', 'german', 'compas', 'bank', 'meps19']
results = {}

for dataset_name in datasets:
    try:
        results[dataset_name] = evaluate_dataset(dataset_name)
        # Plot all baselines for individual dataset
        plt.figure(figsize=(6, 6))
        metrics = results[dataset_name]
        dpds = [metrics['fairot_eps']['dpd'], metrics['uniform']['dpd'], metrics['spotgreedy']['dpd']]
        aucs = [metrics['fairot_eps']['auc'], metrics['uniform']['auc'], metrics['spotgreedy']['auc']]
        colors = ['purple', 'blue', 'green']
        labels = ['FairOT (epsilon)', 'Uniform', 'SpotGreedy']
        for i in range(3):
            plt.scatter(dpds[i], aucs[i], c=colors[i], s=100, label=labels[i])
            plt.annotate(labels[i], (dpds[i], aucs[i]), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Demographic Parity Difference (lower is better)')
        plt.ylabel('AUC (higher is better)')
        plt.title(f'{dataset_name.upper()} Fairness-Utility Tradeoff')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"fairness_utility_{dataset_name}_all_baselines.png")
        plt.close()
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Print summary table
print("\nSummary Results:")
print("-" * 50)
print(f"{'Dataset':<10} {'Method':<18} {'AUC':<8} {'DPD':<8}")
print("-" * 50)
for dataset in results:
    metrics_fairot = results[dataset]['fairot_eps']
    metrics_uniform = results[dataset]['uniform']
    metrics_spotgreedy = results[dataset]['spotgreedy']
    print(f"{dataset:<10} {'FairOT (epsilon)':<18} {metrics_fairot['auc']:.4f} {metrics_fairot['dpd']:.4f}")
    print(f"{dataset:<10} {'Uniform':<18} {metrics_uniform['auc']:.4f} {metrics_uniform['dpd']:.4f}")
    print(f"{dataset:<10} {'SpotGreedy':<18} {metrics_spotgreedy['auc']:.4f} {metrics_spotgreedy['dpd']:.4f}")

# Save results to file
with open("all_datasets_results_fairot_eps.txt", "w") as f:
    f.write("Fairness-Utility Analysis Across Datasets (FairOT epsilon)\n")
    f.write("=" * 50 + "\n\n")
    for dataset in results:
        f.write(f"\n{dataset.upper()} Dataset:\n")
        f.write("-" * 20 + "\n")
        metrics_fairot = results[dataset]['fairot_eps']
        metrics_uniform = results[dataset]['uniform']
        metrics_spotgreedy = results[dataset]['spotgreedy']
        f.write(f"FairOT (epsilon):\n")
        f.write(f"  AUC: {metrics_fairot['auc']:.4f}\n")
        f.write(f"  DPD: {metrics_fairot['dpd']:.4f}\n")
        f.write(f"Uniform Selection:\n")
        f.write(f"  AUC: {metrics_uniform['auc']:.4f}\n")
        f.write(f"  DPD: {metrics_uniform['dpd']:.4f}\n")
        f.write(f"SpotGreedy Selection:\n")
        f.write(f"  AUC: {metrics_spotgreedy['auc']:.4f}\n")
        f.write(f"  DPD: {metrics_spotgreedy['dpd']:.4f}\n")