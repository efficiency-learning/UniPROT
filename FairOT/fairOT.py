import numpy as np
from typing import Callable, List, Set
from FairOT.sinkhorn import pot_partial_extended, pot_partial_library
import matplotlib.pyplot as plt
import time

def greedy_fair_prototype_selection(f: Callable, S: np.ndarray, k: int, reg: float) -> List[int]:
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
            gain = f(P, gamma_P, v, S, k, reg)
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


def main():
    np.random.seed(42)  # For reproducibility
    n = 2000
    k = 50
    reg_values = [0.01]  # Multiple regularization values
    
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
        # Store results for this regularization value
        all_results[reg] = {
            'obj_values_approx_lib': obj_values_approx_lib,
            'P_approx_lib': P_approx_lib,
        }


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
    main()
