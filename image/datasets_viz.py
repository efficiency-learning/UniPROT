import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys

# Add UCI datasets path to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FairOT.datasets.fairness_experiments.uci_datasets import load_communities_crime

def load_and_process_datasets():
    """Load all datasets and process them for visualization"""
    datasets = {}
    
    # Adult Dataset
    print("Loading Adult dataset...")
    try:
        adult = fetch_openml("adult", version=2, as_frame=True)
        X_adult = adult.data.select_dtypes(include=[np.number])
        y_adult = (adult.target == '>50K').astype(int)
        datasets['Adult'] = (X_adult, y_adult, 'Income >50K')
    except Exception as e:
        print(f"Error loading Adult dataset: {e}")

    # Crime Dataset
    print("Loading Crime dataset...")
    try:
        # Load using UCI loader
        result = load_communities_crime()
        if result is not None:
            (X_crime, y_crime), _ = result
            # Convert to pandas DataFrame for consistency
            feature_names = [f'feature_{i}' for i in range(X_crime.shape[1])]
            X_crime = pd.DataFrame(X_crime, columns=feature_names)
            datasets['Crime'] = (X_crime, y_crime, 'High Crime Rate')
        else:
            print("Failed to load Crime dataset, skipping...")
    except Exception as e:
        print(f"Error loading Crime dataset: {e}")

    # German Credit Dataset
    print("Loading German Credit dataset...")
    try:
        credit = fetch_openml("credit-g", version=1, as_frame=True)
        X_credit = credit.data.select_dtypes(include=[np.number])
        y_credit = (credit.target == 'good').astype(int)
        datasets['Credit'] = (X_credit, y_credit, 'Good Credit')
    except Exception as e:
        print(f"Error loading Credit dataset: {e}")

    # Drug Dataset
    print("Loading Drug dataset...")
    try:
        drug = fetch_openml("illicit-drugs", version=1, as_frame=True)
        X_drug = drug.data.select_dtypes(include=[np.number])
        y_drug = (drug.target != "Never Used").astype(int)
        datasets['Drug'] = (X_drug, y_drug, 'Drug Usage')
    except Exception as e:
        print(f"Error loading Drug dataset: {e}")

    return datasets

def plot_class_distribution(datasets):
    """Plot class distribution for all datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (name, (X, y, label)) in enumerate(datasets.items()):
        ax = axes[idx]
        class_counts = np.bincount(y)
        total = len(y)
        percentages = class_counts / total * 100
        
        sns.barplot(x=['Negative', 'Positive'], y=percentages, ax=ax)
        ax.set_title(f'{name} Dataset Class Distribution')
        ax.set_ylabel('Percentage')
        
        # Add percentage labels on bars
        for i, p in enumerate(percentages):
            ax.text(i, p, f'{p:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_feature_distributions(datasets):
    """Plot feature distributions by class"""
    for name, (X, y, label) in datasets.items():
        n_features = min(5, X.shape[1])  # Plot first 5 features
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i in range(n_features):
            feature_name = X.columns[i]
            sns.boxplot(x=y, y=X.iloc[:, i], ax=axes[i])
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel(feature_name)
            axes[i].set_title(f'{name} - {feature_name} Distribution by Class')
        
        plt.tight_layout()

def calculate_fairness_metrics(datasets):
    """Calculate basic fairness metrics"""
    metrics = {}
    for name, (X, y, label) in datasets.items():
        # Class imbalance ratio
        class_counts = np.bincount(y)
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        # Feature disparities
        feature_disparities = []
        for col in X.columns:
            means_by_class = [X[y == c][col].mean() for c in [0, 1]]
            disparity = abs(means_by_class[0] - means_by_class[1])
            feature_disparities.append(disparity)
        
        metrics[name] = {
            'imbalance_ratio': imbalance_ratio,
            'avg_feature_disparity': np.mean(feature_disparities)
        }
    
    return metrics

def main():
    # Create output directory for plots
    output_dir = 'dataset_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    datasets = load_and_process_datasets()
    
    # Plot class distributions
    fig = plot_class_distribution(datasets)
    fig.savefig(os.path.join(output_dir, 'class_distributions.png'))
    plt.close(fig)
    
    # Plot feature distributions
    plot_feature_distributions(datasets)
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # Calculate and print fairness metrics
    metrics = calculate_fairness_metrics(datasets)
    print("\nFairness Metrics:")
    for name, metric in metrics.items():
        print(f"\n{name} Dataset:")
        print(f"Class Imbalance Ratio: {metric['imbalance_ratio']:.3f}")
        print(f"Average Feature Disparity: {metric['avg_feature_disparity']:.3f}")

if __name__ == "__main__":
    main()
