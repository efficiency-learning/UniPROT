import numpy as np
from typing import Callable, List, Set
from colm.train.sinkhorn import pot_partial_extended, pot_partial_library
import matplotlib.pyplot as plt
import time


def greedy_fairot(S: np.ndarray, k: int, reg: float=1e-2) -> List[int]:
    """
    Greedy algorithm with approximate gain for fair prototype selection.
    Args:
        f: Function to compute gain (approximate gain function)
        S: Similarity matrix (n x n)
        k: Cardinality constraint
        reg: Entropic regularization parameter
    Returns:
        List of selected prototype indices (P)
    """
    n = S.shape[0]
    P = []
    candidates = set(range(n))
    for i in range(k):
        # Solve partial OT for current P
        if len(P) == 0:
            gamma_P = None
        else:
            S_P = S[np.ix_(P, range(n))]
            mu_P = np.ones(len(P)) / len(P)
            gamma_P, _ = pot_partial_extended(S_P, k, mu_P, reg)
        # Greedy selection via approximate gain
        best_gain = -np.inf
        best_v = None
        for v in candidates - set(P):
            gain = approx_gain(P, gamma_P, v, S, k, reg)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        P.append(best_v)
    return P

def optimal_alpha_vectorized(sorted_S_a: np.ndarray, sorted_indices: np.ndarray, b: np.ndarray, reg: float, tol=1e-8) -> np.ndarray:
    """
    Compute optimal alpha using a vectorized approach for all possible partitions.
    Args:
        S_a: Similarity vector for candidate a (shape n,)
        b: Upper bound vector (shape n,)
        reg: Entropic regularization parameter
        tol: Tolerance for numerical checks
    Returns:
        alpha: Optimal solution (shape n,)
    """
    n = sorted_S_a.shape[0]
    # Sort S_a in descending order and get sorted indices
    sorted_b = b[sorted_indices]
    # Find the partition index p
    cumulative_sum = np.cumsum(sorted_b)
    p = np.searchsorted(cumulative_sum, 1, side='right')  # Find the index where sum(b[:p]) <= 1

    # Compute scaling factor for interior points
    sum_boundary = np.sum(sorted_b[p:])
    sum_exp_interior = np.sum(np.exp(sorted_S_a[:p] / reg))
    scaling_factor = (1.0 - sum_boundary) / sum_exp_interior if sum_boundary < 1.0 else 0

    # Compute alpha values
    alpha = np.zeros(n)
    alpha[sorted_indices[:p]] = scaling_factor * np.exp(sorted_S_a[:p] / reg)
    alpha[sorted_indices[p:]] = sorted_b[p:]

    return alpha


def optimal_alpha(S_a: np.ndarray, b: np.ndarray, reg: float, tol=1e-8, max_iter=100) -> np.ndarray:
    """
    Compute optimal alpha using the closed-form KKT solution.
    
    Based on the coordinate-wise analysis:
    - Interior points (0 < α_i < b_i): α_i = scaling_factor * exp(S_a[i]/λ)
    - Boundary points (α_i = b_i): α_i = b_i
    
    Where scaling_factor = (1 - sum(b_i for boundary points)) / sum(exp(S_a[i]/λ) for interior points)
    
    Algorithm: Sort S_a points and iteratively find optimal partition between interior/boundary
    """
    n = S_a.shape[0]
    
    # Precompute exp(S_a[i]/λ) for efficiency
    exp_S_scaled = np.exp(S_a / reg)
    
    # Sort indices by S_a values (descending order - highest similarity first)
    sorted_indices = np.argsort(-S_a)
    
    best_alpha = None
    best_objective = -np.inf
    
    # Try all possible partitions: first p points are interior, rest are boundary
    for p in range(n + 1):  # p = 0, 1, ..., n
        if p == 0:
            # All points are boundary
            if np.sum(b) > 0:
                alpha = b / np.sum(b)
            else:
                alpha = np.zeros(n)
        elif p == n:
            # All points are interior (unconstrained softmax)
            alpha = exp_S_scaled / np.sum(exp_S_scaled)
            # Check if this violates any boundary constraint
            if np.any(alpha > b + tol):
                continue  # Invalid partition
        else:
            # Mixed case: first p points (by sorted order) are interior
            interior_mask = np.zeros(n, dtype=bool)
            interior_mask[sorted_indices[:p]] = True
            boundary_mask = ~interior_mask
            
            # Check if this partition makes sense:
            # Interior points should have potential to be < b_i
            # (otherwise they should be boundary)
            sum_boundary = np.sum(b[boundary_mask])
            sum_exp_interior = np.sum(exp_S_scaled[interior_mask])
            
            if sum_boundary >= 1.0:
                # Boundary points already sum to ≥ 1, set interior to 0
                alpha = np.zeros(n)
                alpha[boundary_mask] = b[boundary_mask] / sum_boundary
            else:
                # Normal case: compute scaling factor
                scaling_factor = (1.0 - sum_boundary) / sum_exp_interior
                alpha = np.zeros(n)
                alpha[boundary_mask] = b[boundary_mask]
                alpha[interior_mask] = scaling_factor * exp_S_scaled[interior_mask]
                
                # Verify that interior points are actually < b_i
                if np.any(alpha[interior_mask] > b[interior_mask] + tol):
                    continue  # Invalid partition
        
        # Check if this alpha satisfies all constraints
        if (abs(np.sum(alpha) - 1.0) < tol and 
            np.all(alpha >= -tol) and 
            np.all(alpha <= b + tol)):
            
            # Compute objective value for this partition
            objective = np.sum(S_a * alpha) + reg * np.sum(-alpha * np.log(alpha + 1e-12))
            
            if objective > best_objective:
                best_objective = objective
                best_alpha = alpha.copy()
    
    if best_alpha is None:
        # Fallback: normalize b if no valid partition found
        best_alpha = b / np.sum(b) if np.sum(b) > 0 else np.ones(n) / n
    
    # Final verification
    assert np.abs(np.sum(best_alpha) - 1.0) < tol, f"Sum constraint violated: {np.sum(best_alpha)}"
    assert np.all(best_alpha >= -tol), f"Non-negativity violated: min = {np.min(best_alpha)}"
    assert np.all(best_alpha <= b + tol), f"Upper bound violated: max excess = {np.max(best_alpha - b)}"
    
    return best_alpha


def approx_gain(P: List[int], gamma_P, v: int, S: np.ndarray, S_a: np.ndarray, sorted_indices: np.ndarray ,b:np.ndarray, k: int, reg: float) -> float:
    """
    Approximate gain function for greedy selection using feasible extension.
    Args:
        P: Current set of prototypes
        gamma_P: Current OT plan (can be None if P is empty)
        v: Candidate index to add
        S: Similarity matrix
        k: Cardinality constraint
        reg: Entropic regularization parameter
    Returns:
        Approximate gain of adding v to P
    """
    n = S.shape[0]
    if gamma_P is None or len(P) == 0:
        # If P is empty, just solve for {v}
        S_P_new = S[np.ix_([v], range(n))]
        mu_P_new = np.ones(1)
        gamma_P_new, obj_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
        obj_old = 0.0
        return obj_new - obj_old
    else:
        m = len(P)
        S_P = S[np.ix_(P, range(n))]
 # ensure non-negative upper bounds
        # Use closed-form for optimal alpha
        #alpha = np.zeros(n)
        alpha = optimal_alpha_vectorized(S_a.flatten(), sorted_indices, b, reg)
        gamma_tilde = np.vstack([gamma_P, alpha.reshape(1, n)])
        obj = np.sum(S_P * gamma_P) + np.sum(S_a * alpha)
        mask = gamma_tilde > 0
        entropy = -np.sum(gamma_tilde[mask] * np.log(gamma_tilde[mask]))
        obj = obj + reg * entropy
        mask_old = gamma_P > 0
        entropy_old = -np.sum(gamma_P[mask_old] * np.log(gamma_P[mask_old]))
        obj_old = np.sum(S_P * gamma_P) + reg * entropy_old
        return obj - obj_old

def test_optimal_alpha_constraints():
    np.random.seed(42)
    n = 10
    k = 5
    reg = 0.1
    S = np.random.rand(n, n)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    # Select a random set P of size m
    m = 3
    P = np.random.choice(n, m, replace=False).tolist()
    S_P = S[np.ix_(P, range(n))]
    mu_P = np.ones(m) / m
    gamma_P_star, _ = pot_partial_extended(S_P, k, mu_P, reg)
    # Pick a candidate v not in P
    v = np.random.choice(list(set(range(n)) - set(P)))
    S_a = S[v, :]
    mu_T = k * np.ones(n) / n
    col_sums = np.sum(gamma_P_star, axis=0)
    b = mu_T - col_sums
    b = np.clip(b, 0, None)
    alpha = optimal_alpha(S_a, sorted_indices,      b, reg)
    print("alpha:", alpha)
    print("sum(alpha):", np.sum(alpha))
    print("min(alpha):", np.min(alpha))
    print("max(alpha):", np.max(alpha))
    print("b:", b)
    # Check constraints with visual signals
    all_pass = True
    for i in range(n):
        interior = (alpha[i] < b[i] - 1e-8)
        boundary = (abs(alpha[i] - b[i]) < 1e-6)
        nonneg = (alpha[i] >= -1e-8)
        kkt_interior = False
        kkt_boundary = False
        if interior and nonneg:
            # KKT: alpha_i = exp((S_a[i] - beta)/reg - 1) for some beta
            kkt_interior = True
        if boundary and nonneg:
            # KKT: beta <= S_a[i] - reg * (1 + log(b[i]))
            kkt_boundary = True
        if interior and nonneg and kkt_interior:
            print(f"\u2705 alpha[{i}] interior OK: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
        elif boundary and nonneg and kkt_boundary:
            print(f"\u2705 alpha[{i}] boundary OK: {alpha[i]:.4f} == b[{i}]={b[i]:.4f}")
        elif not nonneg:
            print(f"\u274C alpha[{i}] NEGATIVE: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
            all_pass = False
        else:
            print(f"\u274C alpha[{i}] violates KKT: {alpha[i]:.4f} vs b[{i}]={b[i]:.4f}")
            all_pass = False
    if abs(np.sum(alpha) - 1) < 1e-6:
        print("\u2705 sum(alpha) == 1")
    else:
        print(f"\u274C sum(alpha) = {np.sum(alpha):.6f} (should be 1)")
        all_pass = False
    if all_pass:
        print("\u2705 All coordinate-wise KKT constraints satisfied.")
    else:
        print("\u274C Some constraints failed.")

def greedy_fair_prototype_selection_with_obj(f: Callable, S: np.ndarray, k: int, reg: float) -> (List[int], List[float]):
    """
    Greedy algorithm with objective tracking for fair prototype selection.
    Returns selected indices and list of f(P) values at each step.
    """
    n = S.shape[0]
    P = []
    candidates = set(range(n))
    obj_values = []
    gamma_P = None
    for i in range(k):
        # Solve partial OT for current P
        if len(P) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P, range(n))]
            mu_P = np.ones(len(P)) / len(P)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values.append(obj_P)
        # Greedy selection via approximate gain
        best_gain = -np.inf
        best_v = None
        for v in candidates - set(P):
            gain = f(P, gamma_P, v, S, k, reg)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        P.append(best_v)
    return P, obj_values


def load_mnist_subset(n_samples=100, subset_classes=None):
    """
    Load a subset of MNIST data.
    Args:
        n_samples: Number of samples to load
        subset_classes: List of classes to include (e.g., [0, 1, 2]) or None for all
    Returns:
        X: Data matrix (n_samples, 784)
        y: Labels (n_samples,)
    """
    try:
        from sklearn.datasets import fetch_openml
        # Load MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Filter by classes if specified
        if subset_classes is not None:
            mask = np.isin(y, subset_classes)
            X, y = X[mask], y[mask]
        
        # Sample random subset
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X, y = X[indices], y[indices]
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        return X, y
    except ImportError:
        print("scikit-learn not available, using random data instead")
        return np.random.rand(n_samples, 784), np.random.randint(0, 10, n_samples)

def compute_similarity_matrix(X, method='gaussian', sigma=1.0):
    """
    Compute similarity matrix from data.
    Args:
        X: Data matrix (n_samples, n_features)
        method: 'gaussian', 'cosine', or 'linear'
        sigma: Bandwidth for Gaussian kernel
    Returns:
        S: Similarity matrix (n_samples, n_samples)
    """
    n = X.shape[0]
    
    if method == 'gaussian':
        # Gaussian kernel based on Euclidean distance
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        S = np.exp(-dists**2 / (2 * sigma**2))
    elif method == 'cosine':
        # Cosine similarity
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        S = X_norm @ X_norm.T
        S = np.clip(S, 0, 1)  # Ensure non-negative
    elif method == 'linear':
        # Linear kernel (dot product)
        S = X @ X.T
        S = S / np.max(S)  # Normalize to [0, 1]
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    # Ensure diagonal is 1
    np.fill_diagonal(S, 1.0)
    return S

def main():
    np.random.seed(42)  # For reproducibility
    n = 100
    k = 10
    reg = 0.05
    
    # Choose data source
    data_source = 'mnist'  # 'gaussian' or 'mnist'
    
    if data_source == 'gaussian':
        print("Using Gaussian random data...")
        # Generate random 2D points
        X = np.random.randn(n, 2)
        # Gaussian similarity matrix
        sigma = 1.0
        S = compute_similarity_matrix(X, method='gaussian', sigma=sigma)
    elif data_source == 'mnist':
        print("Using MNIST subset...")
        # Load MNIST subset (first 3 classes for diversity)
        X, y = load_mnist_subset(n_samples=n, subset_classes=[0, 1, 2])
        print(f"Loaded {len(X)} MNIST samples with classes: {np.unique(y)}")
        
        # Compute similarity matrix
        sigma = 50.0  # Larger sigma for high-dimensional MNIST data
        S = compute_similarity_matrix(X, method='gaussian', sigma=sigma)
        print(f"Similarity matrix range: [{np.min(S):.3f}, {np.max(S):.3f}]")
    
    # Rest of the code remains the same...
    # --- Approx-gain greedy (pot_partial_extended) ---
    print("Running approx-gain greedy (extended)...")
    P_approx = []
    obj_values_approx = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx, range(n))]
            mu_P = np.ones(len(P_approx)) / len(P_approx)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_approx.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx):
            approx = approx_gain(P_approx, gamma_P, v, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx.append(best_v)
    # Final objective
    S_P = S[np.ix_(P_approx, range(n))]
    mu_P = np.ones(len(P_approx)) / len(P_approx)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_approx.append(obj_P)

    # --- Actual-gain greedy (pot_partial_extended) ---
    print("Running actual-gain greedy (extended)...")
    P_actual = []
    obj_values_actual = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual, range(n))]
            mu_P = np.ones(len(P_actual)) / len(P_actual)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_actual.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual):
            P_new = P_actual + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual.append(best_v)
    # Final objective
    S_P = S[np.ix_(P_actual, range(n))]
    mu_P = np.ones(len(P_actual)) / len(P_actual)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_actual.append(obj_P)

    # --- Approx-gain greedy (pot_partial_library) ---
    print("Running approx-gain greedy (library)...")
    P_approx_lib = []
    obj_values_approx_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx_lib, range(n))]
            mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_approx_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx_lib):
            approx = approx_gain(P_approx_lib, gamma_P, v, S, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx_lib.append(best_v)
    # Final objective
    S_P = S[np.ix_(P_approx_lib, range(n))]
    mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_approx_lib.append(obj_P)

    # --- Actual-gain greedy (pot_partial_library) ---
    print("Running actual-gain greedy (library)...")
    P_actual_lib = []
    obj_values_actual_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual_lib, range(n))]
            mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_actual_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual_lib):
            P_new = P_actual_lib + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_library(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual_lib.append(best_v)
    # Final objective
    S_P = S[np.ix_(P_actual_lib, range(n))]
    mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_actual_lib.append(obj_P)

    # --- Plot all four curves on the same plot ---
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, obj_values_approx, marker='o', label='Approx-gain (extended)')
    plt.plot(steps, obj_values_actual, marker='s', color='orange', label='Actual-gain (extended)')
    plt.plot(steps, obj_values_approx_lib, marker='^', color='green', label='Approx-gain (library)')
    plt.plot(steps, obj_values_actual_lib, marker='d', color='red', label='Actual-gain (library)')
    plt.title(f'Objective value: all greedy strategies\n({data_source.upper()} data)')
    plt.xlabel('Greedy step (k)')
    plt.ylabel('Objective value f(P)')
    plt.grid(True)
    plt.legend()
    plt.xticks(steps)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Plot incremental gains
    plt.subplot(1, 2, 2)
    inc_approx = np.diff(obj_values_approx)
    inc_actual = np.diff(obj_values_actual)
    inc_approx_lib = np.diff(obj_values_approx_lib)
    inc_actual_lib = np.diff(obj_values_actual_lib)
    
    steps_inc = range(1, k+1)
    plt.plot(steps_inc, inc_approx, marker='o', label='Approx-gain (extended)')
    plt.plot(steps_inc, inc_actual, marker='s', color='orange', label='Actual-gain (extended)')
    plt.plot(steps_inc, inc_approx_lib, marker='^', color='green', label='Approx-gain (library)')
    plt.plot(steps_inc, inc_actual_lib, marker='d', color='red', label='Actual-gain (library)')
    plt.title('Incremental gains')
    plt.xlabel('Greedy step (k)')
    plt.ylabel('Incremental gain')
    plt.grid(True)
    plt.legend()
    plt.xticks(steps_inc)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()
    
    # Print selected prototypes for MNIST
    if data_source == 'mnist':
        print(f"\nSelected prototypes (approx-gain extended): {P_approx}")
        print(f"Their classes: {y[P_approx]}")
        print(f"Class distribution: {np.bincount(y[P_approx])}")

# Example usage
def main_synthetic():
    n = 100
    k = 14
    reg = 0.05
    # Generate random 2D points
    X = np.random.randn(n, 2)
    # Gaussian similarity matrix
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)

    # --- Approx-gain greedy (pot_partial_extended) ---
    P_approx = []
    obj_values_approx = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx, range(n))]
            mu_P = np.ones(len(P_approx)) / len(P_approx)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_approx.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx):
            approx = approx_gain(P_approx, gamma_P, v, S, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx.append(best_v)
    S_P = S[np.ix_(P_approx, range(n))]
    mu_P = np.ones(len(P_approx)) / len(P_approx)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_approx.append(obj_P)

    # --- Actual-gain greedy (pot_partial_extended) ---
    P_actual = []
    obj_values_actual = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual, range(n))]
            mu_P = np.ones(len(P_actual)) / len(P_actual)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_actual.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual):
            P_new = P_actual + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual.append(best_v)
    S_P = S[np.ix_(P_actual, range(n))]
    mu_P = np.ones(len(P_actual)) / len(P_actual)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_actual.append(obj_P)

    # --- Approx-gain greedy (pot_partial_library) ---
    P_approx_lib = []
    obj_values_approx_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx_lib, range(n))]
            mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_approx_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx_lib):
            approx = approx_gain(P_approx_lib, gamma_P, v, S, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx_lib.append(best_v)
    S_P = S[np.ix_(P_approx_lib, range(n))]
    mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_approx_lib.append(obj_P)

    # --- Actual-gain greedy (pot_partial_library) ---
    P_actual_lib = []
    obj_values_actual_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual_lib, range(n))]
            mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_actual_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual_lib):
            P_new = P_actual_lib + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_library(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual_lib.append(best_v)
    S_P = S[np.ix_(P_actual_lib, range(n))]
    mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_actual_lib.append(obj_P)

    # --- Plot all four curves on the same plot ---
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, obj_values_approx, marker='o', label='Approx-gain (extended)')
    plt.plot(steps, obj_values_actual, marker='s', color='orange', label='Actual-gain (extended)')
    plt.plot(steps, obj_values_approx_lib, marker='^', color='green', label='Approx-gain (library)')
    plt.plot(steps, obj_values_actual_lib, marker='d', color='red', label='Actual-gain (library)')
    plt.title('Objective value: all greedy strategies')
    plt.xlabel('Greedy step (k)')
    plt.ylabel('Objective value f(P)')
    plt.grid(True)
    plt.legend()
    plt.xticks(steps)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def main_synthetic_new():
    np.random.seed(42)  # For reproducibility
    n = 2000
    k = 50
    reg_values = [0.01, 0.05, 0.1, 0.5]  # Multiple regularization values
    
    # Generate random 2D points
    X = np.random.randn(n, 2)
    # Gaussian similarity matrix
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)
    print(f"Using synthetic Gaussian data (n={n}, sigma={sigma})")
    print(f"Similarity matrix range: [{np.min(S):.3f}, {np.max(S):.3f}]")
    
    # Store results for all regularization values
    all_results = {}
    
    for reg in reg_values:
        print(f"\n{'='*50}")
        print(f"Running experiments with reg = {reg}")
        print(f"{'='*50}")
        '''
        # --- Approx-gain greedy (pot_partial_extended) ---
        print("Running approx-gain greedy (extended)...")
        P_approx = []
        obj_values_approx = []
        gamma_P = None
        obj_P = 0.0
        for step in range(k):
            if len(P_approx) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_approx, range(n))]
                mu_P = np.ones(len(P_approx)) / len(P_approx)
                gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
            obj_values_approx.append(obj_P)
            best_gain = -np.inf
            best_v = None
            for v in set(range(n)) - set(P_approx):
                approx = approx_gain(P_approx, gamma_P, v, S, k, reg)
                if approx > best_gain:
                    best_gain = approx
                    best_v = v
            P_approx.append(best_v)
        # Final objective
        S_P = S[np.ix_(P_approx, range(n))]
        mu_P = np.ones(len(P_approx)) / len(P_approx)
        _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_approx.append(obj_P)

        # --- Actual-gain greedy (pot_partial_extended) ---
        print("Running actual-gain greedy (extended)...")
        P_actual = []
        obj_values_actual = []
        gamma_P = None
        obj_P = 0.0
        for step in range(k):
            if len(P_actual) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_actual, range(n))]
                mu_P = np.ones(len(P_actual)) / len(P_actual)
                gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
            obj_values_actual.append(obj_P)
            best_gain = -np.inf
            best_v = None
            for v in set(range(n)) - set(P_actual):
                P_new = P_actual + [v]
                S_P_new = S[np.ix_(P_new, range(n))]
                mu_P_new = np.ones(len(P_new)) / len(P_new)
                _, obj_P_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
                actual = obj_P_new - obj_P
                if actual > best_gain:
                    best_gain = actual
                    best_v = v
            P_actual.append(best_v)
        # Final objective
        S_P = S[np.ix_(P_actual, range(n))]
        mu_P = np.ones(len(P_actual)) / len(P_actual)
        _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_actual.append(obj_P)
        '''
        # --- Approx-gain greedy (pot_partial_library) ---
        print("Running approx-gain greedy (library)...")
        P_approx_lib = []
        obj_values_approx_lib = []
        gamma_P = None
        obj_P = 0.0
        mu_T = k * np.ones(n) / n

        sorted_indices_all = np.argsort(-S, axis=1)  # Sort indices for all rows of S
        sorted_S_all = np.take_along_axis(S, sorted_indices_all, axis=1)  # Sort S along rows
        print("All of S is sorted")
        for step in range(k):
            start_time = time.time()

            if len(P_approx_lib) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_approx_lib, range(n))]
                mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
                print("POT library called \n")
                gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
                print(f"Objective val for current proto at step {step}", obj_P)
            obj_values_approx_lib.append(obj_P)
            best_gain = -np.inf
            best_v = None
            if gamma_P is None:
                col_sums = np.zeros(n)  # Initialize col_sums to zeros if gamma_P is None
            else:
                col_sums = np.sum(gamma_P, axis=0)
            

            #col_sums = np.sum(gamma_P, axis=0)
            b = mu_T - col_sums
            b = np.clip(b, 0, None)
            

            # Optimize the loop over candidates
            candidates = np.array(list(set(range(n)) - set(P_approx_lib)))  # Convert to NumPy array for faster indexing
            sorted_indices_candidates = sorted_indices_all[candidates]  # Precompute sorted indices for candidates
            sorted_S_candidates = sorted_S_all[candidates]  # Precompute sorted similarity vectors for candidates
            #TODO: focus code
            # Vectorized computation of approximate gains for all candidates
            gains = np.array([
                approx_gain(P_approx_lib, gamma_P, v, S, sorted_S_candidates[i], sorted_indices_candidates[i], b, k, reg)
                for i, v in enumerate(candidates)
            ])

            # Select the best candidate
            best_gain_idx =  np.argmax(gains)
            best_gain = gains[best_gain_idx]
            best_v = candidates[best_gain_idx]

            P_approx_lib.append(best_v)
            end_time = time.time()
            print(f"Step {step+1}/{k}: Selected {best_v} in {end_time - start_time:.4f} seconds")

            # Final objective
        
        S_P = S[np.ix_(P_approx_lib, range(n))]
        mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
        _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_approx_lib.append(obj_P)

        '''
 
        # --- Actual-gain greedy (pot_partial_library) ---
        print("Running actual-gain greedy (library)...")
        P_actual_lib = []
        obj_values_actual_lib = []
        gamma_P = None
        obj_P = 0.0
        for step in range(k):
            if len(P_actual_lib) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_actual_lib, range(n))]
                mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
                gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
            obj_values_actual_lib.append(obj_P)
            best_gain = -np.inf
            best_v = None
            for v in set(range(n)) - set(P_actual_lib):
                P_new = P_actual_lib + [v]
                S_P_new = S[np.ix_(P_new, range(n))]
                mu_P_new = np.ones(len(P_new)) / len(P_new)
                _, obj_P_new = pot_partial_library(S_P_new, k, mu_P_new, reg)
                actual = obj_P_new - obj_P
                if actual > best_gain:
                    best_gain = actual
                    best_v = v
            P_actual_lib.append(best_v)
        # Final objective
        S_P = S[np.ix_(P_actual_lib, range(n))]
        mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
        _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_actual_lib.append(obj_P)
        
        '''
        # Store results for this regularization value
        all_results[reg] = {
            #'obj_values_approx': obj_values_approx,
            #'obj_values_actual': obj_values_actual,
            'obj_values_approx_lib': obj_values_approx_lib,
           # 'obj_values_actual_lib': obj_values_actual_lib,
            #'P_approx': P_approx,
            #'P_actual': P_actual,
            'P_approx_lib': P_approx_lib,
            #'P_actual_lib': P_actual_lib
        }

    # --- Plot all results ---
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    
    # Create subplots for each regularization value
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, reg in enumerate(reg_values):
        if i >= 4:  # Only plot first 4 reg values
            break
            
        ax = axes[i]
        results = all_results[reg]
        
        #ax.plot(steps, results['obj_values_approx'], marker='o', label='Approx-gain (extended)', linewidth=2)
       # ax.plot(steps, results['obj_values_actual'], marker='s', color='orange', label='Actual-gain (extended)', linewidth=2)
        ax.plot(steps, results['obj_values_approx_lib'], marker='^', color='green', label='Approx-gain (library)', linewidth=2)
        #ax.plot(steps, results['obj_values_actual_lib'], marker='d', color='red', label='Actual-gain (library)', linewidth=2)
        
        ax.set_title(f'Objective value: reg = {reg}\n(Synthetic Gaussian data)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Greedy step (k)')
        ax.set_ylabel('Objective value f(P)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.suptitle(f'Comparison of Greedy Strategies for Different Regularization Values (Synthetic Data)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    # Plot incremental gains
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, reg in enumerate(reg_values):
        if i >= 4:  # Only plot first 4 reg values
            break
            
        ax = axes[i]
        results = all_results[reg]
        
        #inc_approx = np.diff(results['obj_values_approx'])
        #inc_actual = np.diff(results['obj_values_actual'])
        inc_approx_lib = np.diff(results['obj_values_approx_lib'])
        inc_actual_lib = np.diff(results['obj_values_actual_lib'])
        
        steps_inc = range(1, k+1)
        #ax.plot(steps_inc, inc_approx, marker='o', label='Approx-gain (extended)', linewidth=2)
        #ax.plot(steps_inc, inc_actual, marker='s', color='orange', label='Actual-gain (extended)', linewidth=2)
        ax.plot(steps_inc, inc_approx_lib, marker='^', color='green', label='Approx-gain (library)', linewidth=2)
        #ax.plot(steps_inc, inc_actual_lib, marker='d', color='red', label='Actual-gain (library)', linewidth=2)
        
        ax.set_title(f'Incremental gains: reg = {reg}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Greedy step (k)')
        ax.set_ylabel('Incremental gain')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.suptitle(f'Incremental Gains for Different Regularization Values (Synthetic Data)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    # Summary comparison: final objective values
    plt.figure(figsize=(12, 8))
    
    #final_obj_approx = [all_results[reg]['obj_values_approx'][-1] for reg in reg_values]
    #final_obj_actual = [all_results[reg]['obj_values_actual'][-1] for reg in reg_values]
    final_obj_approx_lib = [all_results[reg]['obj_values_approx_lib'][-1] for reg in reg_values]
    final_obj_actual_lib = [all_results[reg]['obj_values_actual_lib'][-1] for reg in reg_values]
    
    x = np.arange(len(reg_values))
    width = 0.2
    
    #plt.bar(x - 1.5*width, final_obj_approx, width, label='Approx-gain (extended)', alpha=0.8)
    #plt.bar(x - 0.5*width, final_obj_actual, width, label='Actual-gain (extended)', alpha=0.8)
    plt.bar(x + 0.5*width, final_obj_approx_lib, width, label='Approx-gain (library)', alpha=0.8)
    plt.bar(x + 1.5*width, final_obj_actual_lib, width, label='Actual-gain (library)', alpha=0.8)
    
    plt.xlabel('Regularization parameter')
    plt.ylabel('Final objective value f(P)')
    plt.title(f'Final Objective Values for Different Regularization Parameters\n(Synthetic Gaussian data, k={k})')
    plt.xticks(x, [f'{reg}' for reg in reg_values])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nSynthetic data experiments completed for regularization values: {reg_values}")
    print(f"Data dimensions: n={n}, k={k}, sigma={sigma}")
    
    # Print performance summary
    print(f"\nPerformance Summary (Final Objective Values):")
    for reg in reg_values:
        results = all_results[reg]
        print(f"reg={reg}:")
        #print(f"  Approx-gain (extended): {results['obj_values_approx'][-1]:.4f}")
        #print(f"  Actual-gain (extended): {results['obj_values_actual'][-1]:.4f}")
        print(f"  Approx-gain (library):  {results['obj_values_approx_lib'][-1]:.4f}")
        print(f"  Actual-gain (library):  {results['obj_values_actual_lib'][-1]:.4f}")
if __name__ == "__main__":
    #test_optimal_alpha_constraints()
    main_synthetic_new()