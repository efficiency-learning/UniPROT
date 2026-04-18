# --- UniPROT Regularization Scan Example ---
import matplotlib.pyplot as plt
import numpy as np
import torch
from UniPROT.UniPROT import UniPROT

def run_uniprot_regscan(X, y, protected, n_prototypes=50, regularization_values=None):
    if regularization_values is None:
        regularization_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    for reg in regularization_values:
        uniprot = UniPROT(regularization=reg, device='cpu')
        X_torch = torch.from_numpy(X).float()
        sims = torch.mm(X_torch, X_torch.t())
        sims = sims / torch.norm(sims, p=2)
        selected_indices, _ = uniprot.prototype_selection(sims, n_prototypes, method='approx', epsilon=0.001)
        if isinstance(selected_indices, torch.Tensor):
            selected_indices = selected_indices.cpu().numpy()
        elif isinstance(selected_indices, list):
            selected_indices = np.array(selected_indices)
        X_sel = X[selected_indices]
        y_sel = y[selected_indices]
        protected_sel = protected[selected_indices]
        # Example: utility = mean(y_sel), fairness = abs(mean(protected_sel==1) - mean(protected_sel==0))
        utility = np.mean(y_sel)
        fairness = abs(np.mean(protected_sel==1) - np.mean(protected_sel==0))
        results.append({'reg': reg, 'utility': utility, 'fairness': fairness})
    return results

def plot_uniprot_regscan(results, out_path):
    plt.figure(figsize=(7, 7))
    for res in results:
        plt.scatter(res['fairness'], res['utility'], c='purple', s=100)
        plt.annotate(f"reg={res['reg']}", (res['fairness'], res['utility']), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Fairness (abs diff in protected group means)')
    plt.ylabel('Utility (mean label)')
    plt.title('UniPROT Regularization Scan')
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

# Example usage:
# X, y, protected = ... # load your data here
# results = run_uniprot_regscan(X, y, protected)
# plot_uniprot_regscan(results, 'UniPROT_regscan_example.png')
import numpy as np
import matplotlib.pyplot as plt
from UniPROT.UniPROT import UniPROT

def main():
    np.random.seed(42)
    n = 2000
    k = 50
    reg = 0.1

    # Generate random 2D points and similarity matrix
    X = np.random.randn(n, 2)
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)
    print(f"Using synthetic Gaussian data (n={n}, sigma={sigma})")
    print(f"Similarity matrix range: [{np.min(S):.3f}, {np.max(S):.3f}]")

    import torch
    S_torch = torch.from_numpy(S).float()
    uniprot = UniPROT(regularization=reg, device='cpu')

    # --- Approximate greedy selection ---
    print("\nRunning UniPROT prototype selection (approximate greedy)...")
    selected_indices_approx, objectives_approx = uniprot.prototype_selection(S_torch, k, method='approx')
    if isinstance(selected_indices_approx, torch.Tensor):
        selected_indices_approx = selected_indices_approx.cpu().numpy()
    elif isinstance(selected_indices_approx, list):
        selected_indices_approx = np.array(selected_indices_approx)
    print(f"Approximate: Selected prototype indices (first 10): {selected_indices_approx[:10]}")
    print(f"Approximate: Objective values (first 10): {objectives_approx[:10]}")
    print(f"Approximate: Final objective value: {objectives_approx[-1]:.4f}")

    # --- Stochastic greedy selection (fraction) ---
    print("\nRunning UniPROT prototype selection (stochastic greedy, frac=0.2)...")
    selected_indices_stoch, objectives_stoch = uniprot.prototype_selection(S_torch, k, method='approx', stochastic_frac=0.2)
    if isinstance(selected_indices_stoch, torch.Tensor):
        selected_indices_stoch = selected_indices_stoch.cpu().numpy()
    elif isinstance(selected_indices_stoch, list):
        selected_indices_stoch = np.array(selected_indices_stoch)
    print(f"Stochastic (frac): Selected prototype indices (first 10): {selected_indices_stoch[:10]}")
    print(f"Stochastic (frac): Objective values (first 10): {objectives_stoch[:10]}")
    print(f"Stochastic (frac): Final objective value: {objectives_stoch[-1]:.4f}")

    # --- Stochastic greedy selection (epsilon) ---
    print("\nRunning UniPROT prototype selection (stochastic greedy, epsilon=0.1)...")
    selected_indices_eps, objectives_eps = uniprot.prototype_selection(S_torch, k, method='approx', epsilon=0.1)
    if isinstance(selected_indices_eps, torch.Tensor):
        selected_indices_eps = selected_indices_eps.cpu().numpy()
    elif isinstance(selected_indices_eps, list):
        selected_indices_eps = np.array(selected_indices_eps)
    print(f"Stochastic (epsilon): Selected prototype indices (first 10): {selected_indices_eps[:10]}")
    print(f"Stochastic (epsilon): Objective values (first 10): {objectives_eps[:10]}")
    print(f"Stochastic (epsilon): Final objective value: {objectives_eps[-1]:.4f}")

    # --- Exact greedy selection ---
    print("\nRunning UniPROT prototype selection (exact greedy)...")
    selected_indices_exact, objectives_exact = uniprot.prototype_selection(S_torch, k, method='exact')
    if isinstance(selected_indices_exact, torch.Tensor):
        selected_indices_exact = selected_indices_exact.cpu().numpy()
    elif isinstance(selected_indices_exact, list):
        selected_indices_exact = np.array(selected_indices_exact)
    print(f"Exact: Selected prototype indices (first 10): {selected_indices_exact[:10]}")
    print(f"Exact: Objective values (first 10): {objectives_exact[:10]}")
    print(f"Exact: Final objective value: {objectives_exact[-1]:.4f}")

    # --- Plot comparison ---
    plt.figure(figsize=(10, 5))
    plt.plot(objectives_approx, marker='o', label='Approximate Greedy')
    plt.plot(objectives_stoch, marker='x', label='Stochastic Greedy (frac=0.2)')
    plt.plot(objectives_eps, marker='^', label='Stochastic Greedy (epsilon=0.1)')
    plt.plot(objectives_exact, marker='s', label='Exact Greedy')
    plt.xlabel('Step')
    plt.ylabel('Objective Value')
    plt.title('UniPROT Prototype Selection: Objective Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("UniPROT_synthetic_objective_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()