import numpy as np
import matplotlib.pyplot as plt
from FairOT.FairOptimalTransport import FairOptimalTransport

def main():
    np.random.seed(42)
    n = 100
    k = 50
    reg = 0.01

    # Generate random 2D points and similarity matrix
    X = np.random.randn(n, 2)
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)
    print(f"Using synthetic Gaussian data (n={n}, sigma={sigma})")
    print(f"Similarity matrix range: [{np.min(S):.3f}, {np.max(S):.3f}]")

    # FairOT selection (CPU only)
    fair_ot = FairOptimalTransport(regularization=reg, device='cpu')
    import torch
    S_torch = torch.from_numpy(S).float()
    print("\nRunning FairOT prototype selection on synthetic data...")
    selected_indices, objectives = fair_ot.prototype_selection(S_torch, k, method='approx')
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.cpu().numpy()
    elif isinstance(selected_indices, list):
        selected_indices = np.array(selected_indices)
    print(f"Selected prototype indices (first 10): {selected_indices[:10]}")
    print(f"Objective values (first 10): {objectives[:10]}")
    print(f"Final objective value: {objectives[-1]:.4f}")

    # Plot objective values
    plt.figure(figsize=(8, 4))
    plt.plot(objectives, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Objective Value')
    plt.title('FairOT Greedy Prototype Selection (Synthetic Data)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fairOT_synthetic_objective_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
    plt.title('FairOT Greedy Prototype Selection (Synthetic Data)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fairOT_synthetic_objective_curve.png")
    plt.close()