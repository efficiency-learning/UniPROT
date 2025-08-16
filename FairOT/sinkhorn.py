import numpy as np
import ot  # POT library
import time

def pot_partial_extended(S_sub: np.ndarray, k: int, mu_P: np.ndarray,
                        reg: float, epsilon: float = 1.0,
                        beta: float = None) -> np.ndarray:
    m, n = S_sub.shape
    
    # Auto-compute beta if not provided (large constant to ensure non-negative similarities)
    if beta is None:
        C_original = -S_sub  # Original cost matrix
        beta = np.max(C_original) + 1.0  # Ensure all similarities are non-negative
    
    # Create extended similarity matrix according to paper:
    # S_tilde_{ij} = beta - C_{ij} where C = -S_sub
    # For original entries: S_tilde = beta - (-S_sub) = beta + S_sub
    S_extended_original = beta + S_sub

    # For dummy row: we want cost = epsilon, so similarity = beta - epsilon
    epsilon_max = np.max(C_original) + 0.01  # Ensure epsilon is the maximum cost
    epsilon_min = np.min(C_original) - 0.01  # Ensure epsilon is the minimum cost
    S_dummy_row = (beta - abs(epsilon_min)) * np.ones((1, n))

    # Create extended similarity matrix (add dummy row)
    S_extended = np.vstack([S_extended_original, S_dummy_row])

    # Convert back to cost matrix: C_extended = -S_extended
    C_extended = -S_extended

    # Extended source marginal: first m entries = 1/k, last entry = (k-m)/k
    mu_selected = np.ones(m) / k
    mu_dummy = np.ones(1) * ((k - m) / k)
    mu_P_ext = np.concatenate([mu_selected, mu_dummy])
    # Target marginal: uniform over n
    mu_T_ext = np.ones(n) / n

    # Solve OT with extended matrices
    gamma_extended = ot.sinkhorn(mu_P_ext, mu_T_ext, C_extended, reg,numItermax=1000)

    #print(f"Extended similarity matrix shape: {S_extended.shape}")
    #print(f"Dummy row similarity: {S_dummy_row.flatten()[0]:.3f} (should be {beta - epsilon_min})")
    #print(f"Original similarities: [{np.min(S_extended_original):.3f}, {np.max(S_extended_original):.3f}]")
    #print(f"Beta: {beta:.3f}, Epsilon: {epsilon_min}")
    #print(f"Dummy mass: {mu_dummy[0]:.6f}")

    #print("\n2. POT Extended Partial OT:")
    #print(f"Shape: {gamma_extended.shape}")
    #print(f"Total mass transported: {np.sum(gamma_extended):.6f}")
    #print(f"Row sums (should= mu_P): {np.sum(gamma_extended, axis=1)}")
    #print(f"Column sums: {np.sum(gamma_extended, axis=0)}")

    # Return only the non-dummy part (first m rows)
    gamma_non_dummy = m*gamma_extended[:m, :]
    #gamma_non_dummy = gamma_extended[:m, :]
    # Compute the objective value with non-extended S and gamma_non_dummy (first m rows)
    obj_value = np.sum(S_sub * gamma_non_dummy)
    #print(f"Objective value (non-extended S, gamma_extended): {obj_value:.6f}")

    # Compute the entropic regularization term
    mask = gamma_non_dummy > 0
    entropy = -np.sum(gamma_non_dummy[mask] * np.log(gamma_non_dummy[mask]))
    obj_value = obj_value + reg * entropy
    return gamma_non_dummy, obj_value


def pot_partial_library(S_sub: np.ndarray, k: int, mu_P: np.ndarray, reg: float) -> np.ndarray:
    start_time = time.time()
    m, n = S_sub.shape
    # Cost matrix is negative similarity
    C = -S_sub
    # Target marginal for partial transport
    mu_T = k * np.ones(n) / n
    mu_P = np.ones(m)
    # Print L1 norms for diagnostics
    l1_mu_P = np.sum(np.abs(mu_P))
    l1_mu_T = np.sum(np.abs(mu_T))
    #print(f"L1 norm of mu_P: {l1_mu_P:.6f}")
    #print(f"L1 norm of mu_T: {l1_mu_T:.6f}")
    #print(f"Min of L1 norms: {min(l1_mu_P, l1_mu_T):.6f}")
    # Mass to transport (partial transport parameter)
    mass_to_transport = min(m,k)
    if(mass_to_transport > min(l1_mu_P, l1_mu_T)):
        print("Warning: mass to transport exceeds min(|a|_1, |b|_1). This may lead to unexpected results.")
        mass_to_transport = min(l1_mu_P, l1_mu_T)
    #print(f" mass to transport is {mass_to_transport} while min(|a|_1, |b|_1) is {min(l1_mu_P, l1_mu_T)}")

    #print("Min of marginals: ", np.min((mu_P), (mu_T)))
    # Use POT's partial optimal transport
    gamma_star = ot.partial.entropic_partial_wasserstein(
        mu_P, mu_T, C, reg, numItermax=1000, m= mass_to_transport
    )

    #print("\n2. POT Library Partial OT:")
    #print(f"Shape: {gamma_star.shape}")
    #print(f"Total mass transported: {np.sum(gamma_star):.6f}")
    #print(f"Row sums (should= mu_P): {np.sum(gamma_star, axis=1)}")
    #print(f"Column sums: {np.sum(gamma_star, axis=0)}")
    
    obj_value = np.sum(S_sub * gamma_star)- reg*np.sum(gamma_star * np.log(gamma_star + 1e-10))  # Avoid log(0)
    end_time = time.time()
    print(f"Sinkhorn iterations in {end_time - start_time:.4f} seconds")

    return gamma_star, obj_value


def compare_partial_ot_methods(S_sub: np.ndarray, k: int, mu_P: np.ndarray, reg: float):
    """
    Compare different partial OT implementations.
    """
    #print("=" * 60)
    #print("PARTIAL OPTIMAL TRANSPORT COMPARISON")
    #print("=" * 60)
    
    # Method 1: Extended matrix approach
    #print("\n1. Extended Matrix Approach:")
    gamma_extended, obj_extended = pot_partial_extended(S_sub, k, mu_P, reg)
    #print(f"Shape: {gamma_extended.shape}")
    #print(f"Total mass transported: {np.sum(gamma_extended):.6f}")
    #print(f"Row sums (should ≤ mu_P): {np.sum(gamma_extended, axis=1)}")
    #print(f"Column sums: {np.sum(gamma_extended, axis=0)}")
    
    # Method 2: POT library partial OT
    #print("\n2. POT Library Partial OT:")
    gamma_library, obj_library = pot_partial_library(S_sub, k, mu_P, reg)
    #print(f"Shape: {gamma_library.shape}")
    #print(f"Total mass transported: {np.sum(gamma_library):.6f}")
    #print(f"Row sums (should= mu_P): {np.sum(gamma_library, axis=1)}")
    #print(f"Column sums: {np.sum(gamma_library, axis=0)}")
    

    # Compare solutions
    #print("\n4. Solution Comparison:")
    if gamma_extended.shape == gamma_library.shape:
        diff_ext_lib = np.linalg.norm(gamma_extended - gamma_library, 'fro')
        diff_ext_lib = diff_ext_lib/(np.linalg.norm(gamma_library, 'fro') * 1.0)
        print(f"(Relative) Frobenius norm difference (Extended Vanilla OT vs Partial OT): {diff_ext_lib:.6f}")
        print(f"(Relative) Diff between extended and partial ot", (obj_extended- obj_library)/(obj_library*1.0))
        print(f"POT library ot.partial.wasserstein: {obj_library:.6f}")
    return gamma_extended, gamma_library


# Test parameters
n = 20
P = [3, 7,8,19,10]
S = np.random.rand(n, n)
S = (S + S.T) / 2
np.fill_diagonal(S, 1.0)
S_P = S[np.ix_(P, range(n))]
mu_P_test = np.ones(len(P)) / len(P)
mu_T_test = np.ones(n) / n
reg_test = 0.1
k_test = 12


# Compare partial OT methods
gamma_ext, gamma_lib = compare_partial_ot_methods(
    S_P, k_test, mu_P_test, reg_test
)


'''
import matplotlib.pyplot as plt

def plot_frobenius_vs_P_n(P_list, n_list, k=10, reg=0.1):
    frob_norms = np.zeros((len(P_list), len(n_list)))
    obj_diffs = np.zeros((len(P_list), len(n_list)))
    for i, P_size in enumerate(P_list):
        for j, n in enumerate(n_list):
            P = np.random.choice(n, P_size, replace=False)
            S = np.random.rand(n, n)
            S = (S + S.T) / 2
            np.fill_diagonal(S, 1.0)
            S_P = S[np.ix_(P, range(n))]
            mu_P = np.ones(len(P)) / len(P)
            try:
                gamma_ext, obj_ext = pot_partial_extended(S_P, k, mu_P, reg)
                gamma_lib, obj_lib = pot_partial_library(S_P, k, mu_P, reg)
                if gamma_ext.shape == gamma_lib.shape:
                    frob_norms[i, j] = (np.linalg.norm(gamma_ext - gamma_lib, 'fro'))/(np.linalg.norm(gamma_lib, 'fro') *1.0)
                    obj_diffs[i, j] = obj_ext - obj_lib
                else:
                    frob_norms[i, j] = np.nan
                    obj_diffs[i, j] = np.nan
            except Exception as e:
                frob_norms[i, j] = np.nan
                obj_diffs[i, j] = np.nan
    # Plot Frobenius norm
    for i, P_size in enumerate(P_list):
        plt.plot(n_list, frob_norms[i, :], label=f'|P|={P_size}')
    plt.xlabel('n')
    plt.ylabel('Frobenius norm difference')
    plt.title('Frobenius norm (Extended vs Library) vs n for different |P|')
    plt.legend()
    plt.show()
    # Plot objective value difference
    for i, P_size in enumerate(P_list):
        plt.plot(n_list, obj_diffs[i, :], label=f'|P|={P_size}')
    plt.xlabel('n')
    plt.ylabel('Objective value difference (Extended - Partial OT)')
    plt.title('Objective value difference vs n for different |P|')
    plt.legend()
    plt.show()

# Example usage:
P_list = [3, 5, 10, 20]
n_list = [50, 100, 200, 500]
plot_frobenius_vs_P_n(P_list, n_list, k=10, reg=0.1)
'''