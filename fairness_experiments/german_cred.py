from aif360.datasets import GermanDataset
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
from baselines.SPOTgreedy import SPOT_GreedySubsetSelection

# Setup German Credit dataset
setup_german_credit_dataset()

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

# Convert AIF360 dataset to numpy arrays for processing
X = dataset_orig.features
y = (dataset_orig.labels.ravel() == 1).astype(int)  # Convert to binary (1 = good, 0 = bad)
protected = dataset_orig.protected_attributes[:, 0]  # Get age attribute

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Total samples",len(X))
# ---------------------------
# Uniform subsampling
# ---------------------------
subsample_frac = 0.05  # German Credit is smaller, use larger fraction
n_samples = int(subsample_frac * len(X))
uniform_indices = np.random.choice(len(X), size=n_samples, replace=False)
X_sub = X[uniform_indices]
y_sub = y[uniform_indices]
protected_sub = protected[uniform_indices]

# ---------------------------
# FairOT Selection
# ---------------------------
# Convert to torch tensors (on CPU)
X_torch = torch.from_numpy(X).float()
y_torch = torch.from_numpy(y).long()
protected_torch = torch.from_numpy(protected).long()

# Initialize FairOT
fair_ot = FairOptimalTransport(regularization=0.01)

# Compute similarity matrix
sims = torch.mm(X_torch, X_torch.t())
sims = sims / torch.norm(sims, p=2)

# Select prototypes
n_prototypes = 50
selected_indices, _ = fair_ot.prototype_selection(sims, n_prototypes, method='approx')

# Convert indices to numpy
if isinstance(selected_indices, torch.Tensor):
    selected_indices = selected_indices.cpu().numpy()
elif isinstance(selected_indices, list):
    selected_indices = np.array(selected_indices)

# Get selected samples
X_fairot = X[selected_indices]
y_fairot = y[selected_indices]
protected_fairot = protected[selected_indices]

# ---------------------------
# SPOTgreedy Selection
# ---------------------------
# Create target distribution (uniform across classes)
target_marginal = torch.ones(len(X)) / len(X)

# Use SPOTgreedy to select prototypes
spot_indices = SPOT_GreedySubsetSelection(sims, target_marginal, n_prototypes)
if isinstance(spot_indices, torch.Tensor):
    spot_indices = spot_indices.numpy()
elif isinstance(spot_indices, list):
    spot_indices = np.array(spot_indices)

# Get selected samples for SPOTgreedy
X_spot = X[spot_indices]
y_spot = y[spot_indices]
protected_spot = protected[spot_indices]

# ---------------------------
# MLP Training and Evaluation
# ---------------------------
def train_evaluate_mlp(X_train, y_train, X_test, y_test, protected_test, hidden_dim=64, epochs=30):
    """Train MLP and evaluate fairness-utility metrics"""
    # Always use CPU
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # Define model (on CPU)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2)
    )
    
    # Train
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)
        
        # Compute metrics
        auc = roc_auc_score(y_test.numpy(), probs)
        
        # Compute demographic disparity
        p1 = preds[protected_test == 1].mean()
        p0 = preds[protected_test == 0].mean()
        dpd = abs(p1 - p0)
        
    return auc, dpd

# Evaluate both approaches
print("\nEvaluating Uniform Sampling vs FairOT Selection...")

# Uniform sampling evaluation
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X_sub, y_sub, protected_sub, test_size=0.3, random_state=42, stratify=y_sub
)
auc_unif, dpd_unif = train_evaluate_mlp(X_train, y_train, X_test, y_test, prot_test)

# FairOT evaluation
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X_fairot, y_fairot, protected_fairot, test_size=0.3, random_state=42, stratify=y_fairot
)
auc_fairot, dpd_fairot = train_evaluate_mlp(X_train, y_train, X_test, y_test, prot_test)

# SPOTgreedy evaluation
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X_spot, y_spot, protected_spot, test_size=0.3, random_state=42, stratify=y_spot
)
auc_spot, dpd_spot = train_evaluate_mlp(X_train, y_train, X_test, y_test, prot_test)

# Print results
print("\nResults:")
print("Uniform Sampling:")
print(f"  AUC (utility): {auc_unif:.4f}")
print(f"  Demographic Parity Difference (fairness): {dpd_unif:.4f}")
print("\nFairOT Selection:")
print(f"  AUC (utility): {auc_fairot:.4f}")
print(f"  Demographic Parity Difference (fairness): {dpd_fairot:.4f}")
print("\nSPOTgreedy Selection:")
print(f"  AUC (utility): {auc_spot:.4f}")
print(f"  Demographic Parity Difference (fairness): {dpd_spot:.4f}")
print("\nImprovement vs Uniform:")
print(f"  AUC improvement: {auc_spot - auc_unif:.4f}")
print(f"  DPD improvement: {dpd_unif - dpd_spot:.4f}")

# Plot fairness-utility comparison
plt.figure(figsize=(10, 6))
plt.scatter([dpd_unif, dpd_fairot, dpd_spot], 
           [auc_unif, auc_fairot, auc_spot],
           c=['red', 'blue', 'green'], s=100)
plt.annotate('Uniform', (dpd_unif, auc_unif), xytext=(5, 5), 
             textcoords='offset points')
plt.annotate('FairOT', (dpd_fairot, auc_fairot), xytext=(5, 5),
             textcoords='offset points')
plt.annotate('SPOTgreedy', (dpd_spot, auc_spot), xytext=(5, 5),
             textcoords='offset points')
plt.xlabel('Demographic Parity Difference (lower is better)')
plt.ylabel('AUC (higher is better)')
plt.title('Fairness-Utility Tradeoff Comparison')
plt.grid(True, alpha=0.3)
plt.savefig("fairness_utility_comparison_all_supp_50.png")
plt.close()