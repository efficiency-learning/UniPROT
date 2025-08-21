import pandas as pd
import numpy as np
import urllib.request
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/ganesh/ICLR/FairOT/FairOT')
from FairOptimalTransport import FairOptimalTransport
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1. Load UCI Communities & Crime dataset
# ---------------------------
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"

# Load attribute names
attr_names = []
with urllib.request.urlopen(names_url) as f:
    for line in f:
        line = line.decode("utf-8").strip()
        if line.startswith("@attribute"):
            name = line.split()[1]
            attr_names.append(name)

# Load data
df = pd.read_csv(uci_url, header=None, names=attr_names, na_values="?")

# Drop non-predictive attributes
df = df.drop(columns=["state", "county", "community", "communityname", "fold"])

# Handle missing values
print(f"Initial shape: {df.shape}")
print("\nMissing values before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing values with median for each column
for column in df.columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"Final shape: {df.shape}")

# ---------------------------
# 2. Sensitive attribute (race percentage Black)
# ---------------------------
sensitive_attr = (df["racepctblack"] > df["racepctblack"].median()).astype(int)

# ---------------------------
# 3. Target variable
# ---------------------------
# Binarize ViolentCrimesPerPop (above median crime = 1)
y = (df["ViolentCrimesPerPop"] > df["ViolentCrimesPerPop"].median()).astype(int)

# ---------------------------
# 4. Uniform subsampling
# ---------------------------
print("Total samples", len(df))

subsample_frac = 0.15  # 30% subsample
df_sub = df.sample(frac=subsample_frac, random_state=42)
print("Total samples", len(df_sub))
sensitive_sub = sensitive_attr.loc[df_sub.index]
y_sub = y.loc[df_sub.index]

# ---------------------------
# 5. Compute Demographic Parity Difference
# ---------------------------
def demographic_parity(y, sensitive):
    """Compute demographic parity difference"""
    p1 = y[sensitive == 1].mean()
    p0 = y[sensitive == 0].mean()
    return abs(p1 - p0), p0, p1

dpd, p0, p1 = demographic_parity(y_sub, sensitive_sub)

print(f"Demographic Parity Difference: {dpd:.4f}")
print(f"P(Y=1 | A=0): {p0:.4f}, P(Y=1 | A=1): {p1:.4f}")

# ---------------------------
# 6. Prepare data for FairOT
# ---------------------------
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=["ViolentCrimesPerPop"]))

# Convert to torch tensors (CPU only)
X_torch = torch.from_numpy(X).float()
y_torch = torch.from_numpy(y.values).long()
sensitive_full = sensitive_attr.loc[df.index]
sensitive_torch = torch.from_numpy(sensitive_sub.values).long()

# Initialize FairOT (CPU only)
fair_ot = FairOptimalTransport(regularization=0.01)

# ---------------------------
# 7. FairOT Selection
# ---------------------------
# Compute similarity matrix
sims = torch.mm(X_torch, X_torch.t())
sims = sims / torch.norm(sims, p=2)

# Select prototypes
n_prototypes = 100
selected_indices, objectives = fair_ot.prototype_selection(sims, n_prototypes, method='approx')

# Handle different return types from prototype selection
if isinstance(selected_indices, torch.Tensor):
    selected_indices = selected_indices.cpu().numpy()
elif isinstance(selected_indices, list):
    selected_indices = np.array(selected_indices)
else:
    raise TypeError(f"Unexpected type for selected_indices: {type(selected_indices)}")

# Ensure indices are valid
selected_indices = selected_indices[selected_indices < len(df)]

# ---------------------------
# SPOTgreedy Selection
# ---------------------------
from baselines.SPOTgreedy import SPOT_GreedySubsetSelection

# Create target distribution (uniform across classes)
target_marginal = torch.ones(len(df)) / len(df)

# Use SPOTgreedy to select prototypes
spot_indices = SPOT_GreedySubsetSelection(sims, target_marginal, n_prototypes)
if isinstance(spot_indices, torch.Tensor):
    spot_indices = spot_indices.numpy()
elif isinstance(spot_indices, list):
    spot_indices = np.array(spot_indices)

# Ensure indices are valid
spot_indices = spot_indices[spot_indices < len(df)]

# ---------------------------
# 8. Analyze Selected Prototypes
# ---------------------------
# Compute demographic statistics for selected prototypes
selected_sensitive = sensitive_full.iloc[selected_indices]
selected_y = y.iloc[selected_indices]

# Calculate demographic parity for selected prototypes
selected_dpd, selected_p0, selected_p1 = demographic_parity(selected_y, selected_sensitive)

print("\nPrototype Selection Analysis:")
print("-" * 50)
print(f"Number of prototypes: {len(selected_indices)}")
print(f"Sensitive attribute distribution: {selected_sensitive.mean():.3f}")
print(f"Target distribution: {selected_y.mean():.3f}")
print("\nDemographic Parity Analysis:")
print(f"Original DPD: {dpd:.4f}")
print(f"Selected Prototypes DPD: {selected_dpd:.4f}")
print(f"DPD Improvement: {dpd - selected_dpd:.4f}")
print(f"\nSelected P(Y=1 | A=0): {selected_p0:.4f}")
print(f"Selected P(Y=1 | A=1): {selected_p1:.4f}")

# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Original distribution
sns.countplot(data=pd.DataFrame({'Sensitive': sensitive_full, 'Target': y}), 
             x='Sensitive', hue='Target', ax=ax1)
ax1.set_title('Original Distribution')

# Selected prototypes distribution
sns.countplot(data=pd.DataFrame({'Sensitive': selected_sensitive, 'Target': selected_y}), 
             x='Sensitive', hue='Target', ax=ax2)
ax2.set_title('Selected Prototypes Distribution')

plt.tight_layout()

# Create output directory
import os
from datetime import datetime

output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save plot
plot_path = os.path.join(output_dir, f'crime_fairot_analysis_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

# Save statistics to accompanying text file
stats_path = os.path.join(output_dir, f'crime_fairot_analysis_{timestamp}.txt')
with open(stats_path, 'w') as f:
    f.write(f"Crime Dataset FairOT Analysis\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Original Analysis:\n")
    f.write(f"  Demographic Parity Difference: {dpd:.4f}\n")
    f.write(f"  P(Y=1 | A=0): {p0:.4f}, P(Y=1 | A=1): {p1:.4f}\n\n")
    f.write(f"Selected Prototypes Analysis:\n")
    f.write(f"  Number of prototypes: {len(selected_indices)}\n")
    f.write(f"  Demographic Parity Difference: {selected_dpd:.4f}\n")
    f.write(f"  DPD Improvement: {dpd - selected_dpd:.4f}\n")
    f.write(f"  P(Y=1 | A=0): {selected_p0:.4f}, P(Y=1 | A=1): {selected_p1:.4f}\n")
    f.write(f"  Sensitive attribute distribution: {selected_sensitive.mean():.3f}\n")
    f.write(f"  Target distribution: {selected_y.mean():.3f}\n")

plt.show()
plt.close()

# ---------------------------
# 9. Fairness-Utility Tradeoff: Train MLP on selected prototypes
# ---------------------------
def train_mlp(X_train, y_train, X_test, y_test, sensitive_test, epochs=30, lr=1e-3, hidden_dim=64):
    """Train a 2-layer MLP and evaluate AUC and demographic disparity"""
    # Convert to tensors (CPU only)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1], hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        logits_test = model(X_test)
        probs = torch.softmax(logits_test, dim=1)[:, 1].numpy()
        y_pred = torch.argmax(logits_test, dim=1).numpy()
        auc = roc_auc_score(y_test.numpy(), probs)

    # Compute demographic disparity on test set
    dpd_test, p0_test, p1_test = demographic_parity(pd.Series(y_pred), pd.Series(sensitive_test))
    return auc, dpd_test, p0_test, p1_test

# Prepare train/test split for selected prototypes
X_selected = df.iloc[selected_indices].drop(columns=["ViolentCrimesPerPop"]).values
y_selected = selected_y.values
sensitive_selected = selected_sensitive.values

# Use a random 70/30 split for evaluation
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X_selected, y_selected, sensitive_selected, test_size=0.3, random_state=42, stratify=y_selected
)

auc, dpd_mlp, p0_mlp, p1_mlp = train_mlp(X_train, y_train, X_test, y_test, sensitive_test)

print("\nMLP Classifier Fairness-Utility Tradeoff (Selected Prototypes):")
print("-" * 50)
print(f"AUC (utility): {auc:.4f}")
print(f"Demographic Parity Difference (fairness): {dpd_mlp:.4f}")
print(f"P(Y=1 | A=0): {p0_mlp:.4f}, P(Y=1 | A=1): {p1_mlp:.4f}")

# Save to stats file
with open(stats_path, 'a') as f:
    f.write("\nMLP Classifier Fairness-Utility Tradeoff (Selected Prototypes):\n")
    f.write(f"  AUC (utility): {auc:.4f}\n")
    f.write(f"  Demographic Parity Difference (fairness): {dpd_mlp:.4f}\n")
    f.write(f"  P(Y=1 | A=0): {p0_mlp:.4f}, P(Y=1 | A=1): {p1_mlp:.4f}\n")

# ---------------------------
# 10. Compare with Uniform Sampling
# ---------------------------
# Select uniform random subset of same size as FairOT
uniform_indices = np.random.choice(len(df), size=n_prototypes, replace=False)

# Prepare uniform sampled data
X_uniform = df.iloc[uniform_indices].drop(columns=["ViolentCrimesPerPop"]).values
y_uniform = y.iloc[uniform_indices].values
sensitive_uniform = sensitive_attr.iloc[uniform_indices].values

# Split uniform data
X_train_unif, X_test_unif, y_train_unif, y_test_unif, sensitive_train_unif, sensitive_test_unif = train_test_split(
    X_uniform, y_uniform, sensitive_uniform, test_size=0.3, random_state=42, stratify=y_uniform
)

# Train MLP on uniform samples
auc_unif, dpd_unif, p0_unif, p1_unif = train_mlp(X_train_unif, y_train_unif, X_test_unif, y_test_unif, sensitive_test_unif)

# Compare results
print("\nMethod Comparison:")
print("-" * 50)
print("FairOT Selection:")
print(f"  AUC (utility): {auc:.4f}")
print(f"  Demographic Parity Difference (fairness): {dpd_mlp:.4f}")
print("\nUniform Selection:")
print(f"  AUC (utility): {auc_unif:.4f}")
print(f"  Demographic Parity Difference (fairness): {dpd_unif:.4f}")
print("\nImprovement with FairOT:")
print(f"  AUC improvement: {auc - auc_unif:.4f}")
print(f"  DPD improvement: {dpd_unif - dpd_mlp:.4f}")

# Add comparison to stats file
with open(stats_path, 'a') as f:
    f.write("\nMethod Comparison:\n")
    f.write("FairOT Selection:\n")
    f.write(f"  AUC (utility): {auc:.4f}\n")
    f.write(f"  Demographic Parity Difference (fairness): {dpd_mlp:.4f}\n")
    f.write("\nUniform Selection:\n")
    f.write(f"  AUC (utility): {auc_unif:.4f}\n")
    f.write(f"  Demographic Parity Difference (fairness): {dpd_unif:.4f}\n")
    f.write("\nImprovement with FairOT:\n")
    f.write(f"  AUC improvement: {auc - auc_unif:.4f}\n")
    f.write(f"  DPD improvement: {dpd_unif - dpd_mlp:.4f}\n")

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['FairOT', 'Uniform']
aucs = [auc, auc_unif]
dpds = [dpd_mlp, dpd_unif]

plt.scatter(dpds, aucs, c=['blue', 'red'], s=100)
for i, method in enumerate(methods):
    plt.annotate(method, (dpds[i], aucs[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Demographic Parity Difference (lower is better)')
plt.ylabel('AUC (higher is better)')
plt.title('Fairness-Utility Tradeoff Comparison')
plt.grid(True, alpha=0.3)

# Save comparison plot
plot_path = os.path.join(output_dir, f'fairness_utility_comparison_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------
# 11. Create train/test split of full dataset first
# ---------------------------
X_all = df.drop(columns=["ViolentCrimesPerPop"]).values
y_all = y.values
sensitive_all = sensitive_attr.values

X_train_full, X_test_full, y_train_full, y_test_full, sensitive_train_full, sensitive_test_full = train_test_split(
    X_all, y_all, sensitive_all, test_size=0.3, random_state=42, stratify=y_all
)

def evaluate_method(X_selected, y_selected, name="Method"):
    """Train MLP on selected samples and evaluate on full test set"""
    # Train MLP
    model = train_mlp(X_selected, y_selected, X_test_full, y_test_full, sensitive_test_full)
    
    # Get predictions on test set
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_full).float())
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)
        
        # Compute metrics on full test set
        auc = roc_auc_score(y_test_full, probs)
        dpd, p0, p1 = demographic_parity(pd.Series(preds), pd.Series(sensitive_test_full))
    
    print(f"\n{name} Results:")
    print(f"  AUC (utility): {auc:.4f}")
    print(f"  DPD (fairness): {dpd:.4f}")
    
    return auc, dpd

# ---------------------------
# Extract data for each method before evaluation
# ---------------------------
# For Uniform sampling (already have X_uniform, y_uniform)

# For FairOT
X_fairot = df.iloc[selected_indices].drop(columns=["ViolentCrimesPerPop"]).values
y_fairot = y.iloc[selected_indices].values

# For SPOTgreedy
X_spot = df.iloc[spot_indices].drop(columns=["ViolentCrimesPerPop"]).values
y_spot = y.iloc[spot_indices].values

# Now evaluate each method on full test set
print("\nEvaluating all methods on full test set...")
auc_unif, dpd_unif = evaluate_method(X_uniform, y_uniform, "Uniform")
auc_fairot, dpd_fairot = evaluate_method(X_fairot, y_fairot, "FairOT")
auc_spot, dpd_spot = evaluate_method(X_spot, y_spot, "SPOTGreedy")

# Compare results
print("\nMethod Comparison:")
print("-" * 50)
for method, auc, dpd in [
    ("Uniform", auc_unif, dpd_unif),
    ("FairOT", auc_fairot, dpd_fairot),
    ("SPOTGreedy", auc_spot, dpd_spot)
]:
    print(f"{method}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  DPD: {dpd:.4f}")

# Plot comparison
plt.figure(figsize=(8, 6))
methods = ['Uniform', 'FairOT', 'SPOTGreedy']
aucs = [auc_unif, auc_fairot, auc_spot]
dpds = [dpd_unif, dpd_fairot, dpd_spot]
colors = ['red', 'blue', 'green']

plt.scatter(dpds, aucs, c=colors, s=100)
for i, method in enumerate(methods):
    plt.annotate(method, (dpds[i], aucs[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Demographic Parity Difference (lower is better)')
plt.ylabel('AUC (higher is better)')
plt.title('Fairness-Utility Tradeoff Comparison (Full Test Set)')
plt.grid(True, alpha=0.3)

# Save final comparison plot
final_plot_path = os.path.join(output_dir, f'fairness_utility_comparison_full_test_{timestamp}.png')
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
plt.close()
plt.close()
