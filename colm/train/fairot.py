import numpy as np
from typing import Callable, List, Set
from colm.train.sinkhorn import pot_partial_extended
import matplotlib.pyplot as plt

def greedy_fairot(S: np.ndarray, k: int, reg: float=1e-2) -> List[int]:

    n = S.shape[0]
    P = []
    candidates = set(range(n))
    gains = []
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
        gains.append(best_gain)
        P.append(best_v)
    print("BEst gains", gains)
    return P

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



import numpy as np
from typing import List, Optional

# def approx_gain(P: List[int], gamma_P: Optional[np.ndarray], v: int, S: np.ndarray, k: int, reg: float) -> float:
#     n = S.shape[0]

#     if gamma_P is None or not P:
#         S_v = S[[v], :]
#         gamma_new, obj_new = pot_partial_extended(S_v, k, np.ones(1), reg)
#         return obj_new

#     S_P = S[P]
#     S_v = S[[v]]
#     mu_T = np.full(n, k / n)
#     col_sums = gamma_P.sum(axis=0)
#     b = np.clip(mu_T - col_sums, 0, None)
#     alpha = optimal_alpha(S_v.ravel(), b, reg)
    
#     gamma_new = np.vstack([gamma_P, alpha])
#     obj_new = S_P.dot(gamma_P.T).sum() + S_v.dot(alpha).sum()
#     obj_new += reg * stable_entropy(gamma_new)

#     obj_old = S_P.dot(gamma_P.T).sum() + reg * stable_entropy(gamma_P)
#     return obj_new - obj_old

# def stable_entropy(gamma: np.ndarray) -> float:
#     mask = gamma > 0
#     return -np.sum(gamma[mask] * np.log(np.maximum(gamma[mask], 1e-12)))

def approx_gain(P: List[int], gamma_P, v: int, S: np.ndarray, k: int, reg: float) -> float:
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
        S_a = S[v, :].reshape(1, n)
        col_sums = np.sum(gamma_P, axis=0)
        mu_T = k * np.ones(n) / n
        b = mu_T - col_sums
        b = np.clip(b, 0, None)  
        alpha = optimal_alpha(S_a.flatten(), b, reg)
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
    m = 3
    P = np.random.choice(n, m, replace=False).tolist()
    S_P = S[np.ix_(P, range(n))]
    mu_P = np.ones(m) / m
    gamma_P_star, _ = pot_partial_extended(S_P, k, mu_P, reg)
    v = np.random.choice(list(set(range(n)) - set(P)))
    S_a = S[v, :]
    mu_T = k * np.ones(n) / n
    col_sums = np.sum(gamma_P_star, axis=0)
    b = mu_T - col_sums
    b = np.clip(b, 0, None)
    alpha = optimal_alpha(S_a, b, reg)
    print("alpha:", alpha)
    print("sum(alpha):", np.sum(alpha))
    print("min(alpha):", np.min(alpha))
    print("max(alpha):", np.max(alpha))
    print("b:", b)
    all_pass = True
    for i in range(n):
        interior = (alpha[i] < b[i] - 1e-8)
        boundary = (abs(alpha[i] - b[i]) < 1e-6)
        nonneg = (alpha[i] >= -1e-8)
        kkt_interior = False
        kkt_boundary = False
        if interior and nonneg:
            kkt_interior = True
        if boundary and nonneg:
            kkt_boundary = True
        if interior and nonneg and kkt_interior:
            print(f"alpha[{i}] interior OK: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
        elif boundary and nonneg and kkt_boundary:
            print(f" alpha[{i}] boundary OK: {alpha[i]:.4f} == b[{i}]={b[i]:.4f}")
        elif not nonneg:
            print(f"alpha[{i}] NEGATIVE: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
            all_pass = False
        else:
            print(f" alpha[{i}] violates KKT: {alpha[i]:.4f} vs b[{i}]={b[i]:.4f}")
            all_pass = False
    if abs(np.sum(alpha) - 1) < 1e-6:
        print(" sum(alpha) == 1")
    else:
        print(f" sum(alpha) = {np.sum(alpha):.6f} (should be 1)")
        all_pass = False
    if all_pass:
        print("All coordinate-wise KKT constraints satisfied.")
    else:
        print(" Some constraints failed.")

def greedy_fair_prototype_selection_with_obj(f: Callable, S: np.ndarray, k: int, reg: float) -> (List[int], List[float]):


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

def main():
    n = 100
    k = 15
    reg = 0.05
    X = np.random.randn(n, 2)
    # Gaussian similarity matrix
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)

    #  Approx-gain greedy 
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

    # Actual-gain greedy 
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

    #  Plot both curves on the same plot 
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, obj_values_approx, marker='o', label='Approx-gain greedy')
    plt.plot(steps, obj_values_actual, marker='s', color='orange', label='Actual-gain greedy')
    plt.title('Objective value: approx-gain vs actual-gain greedy')
    plt.xlabel('Greedy step (k)')
    plt.ylabel('Objective value f(P)')
    plt.grid(True)
    plt.legend()
    plt.xticks(steps)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_optimal_alpha_constraints()
    #main()