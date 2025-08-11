import numpy as np
from typing import Callable, List, Set, Tuple, Optional
import matplotlib.pyplot as plt

import torch
import ot  # POT library
from tqdm import tqdm
from FairOT.sinkhorn import pot_partial_extended, pot_partial_library

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUPY_AVAILABLE = True
except ImportError:
    #print("Warning: CuPy not available, falling back to numpy/torch")
    CUPY_AVAILABLE = False
    cp = None
    cp_sparse = None

class FairOptimalTransport:
    def __init__(self, regularization: float = 0.05, similarity_sigma: float = 1.0, device: str = 'cuda', use_sparse: bool = True):
        """
        Initialize the prototype selector.

        Args:
            regularization: Entropic OT regularization strength
            similarity_sigma: Gaussian similarity parameter (unused here, placeholder)
            device: Device for computation ('cuda' or 'cpu')
            use_sparse: Whether to use sparse matrices for computation
        """
        self.reg = regularization
        self.sigma = similarity_sigma
        self.device = device
        self.use_sparse = use_sparse and CUPY_AVAILABLE

        self.similarity_matrix = None
        self.similarity_matrix_sparse = None
        self.num_examples = None
        self.ot_solver = pot_partial_library_sparse if self.use_sparse else pot_partial_library
        #print(f"Using {'sparse' if self.use_sparse else 'dense'} OT solver on {self.device}")

    def prototype_selection(
        self,
        similarity_matrix,
        num_prototypes: int,
        method: str = 'approx'
    ) -> Tuple[List[int], List[float]]:
        # Accepts torch.Tensor, numpy, or cupy arrays
        if self.use_sparse and CUPY_AVAILABLE:
            # Convert to cupy array if not already
            if not isinstance(similarity_matrix, cp.ndarray):
                similarity_matrix = cp.array(similarity_matrix)
            self.similarity_matrix = similarity_matrix
            self.num_examples = similarity_matrix.shape[0]
        elif isinstance(similarity_matrix, torch.Tensor):
            self.similarity_matrix = similarity_matrix.to(self.device)
            self.num_examples = similarity_matrix.size(0)
        elif isinstance(similarity_matrix, np.ndarray):
            self.similarity_matrix = similarity_matrix
            self.num_examples = similarity_matrix.shape[0]
        else:
            raise TypeError("Unsupported type for similarity_matrix. Must be torch.Tensor, numpy.ndarray, or cupy.ndarray.")

        if method == 'exact':
            return self._greedy_selection_exact(num_prototypes)
        elif method == 'approx':
            return self._greedy_selection_approx(num_prototypes)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'approx' or 'exact'.")

    def _greedy_selection_exact(self, k: int) -> Tuple[List[int], List[float]]:
        """Greedy selection with exact gain computation."""
        #print("Running actual-gain greedy (extended)...")
        P_actual = []
        obj_values_actual = []
        gamma_P = None
        obj_P = 0.0
        
        # Get similarity matrix and dimensions
        if self.use_sparse and CUPY_AVAILABLE:
            S = cp.asnumpy(self.similarity_matrix)  # Convert cupy to numpy for processing
        elif isinstance(self.similarity_matrix, torch.Tensor):
            S = self.similarity_matrix.cpu().numpy()
        else:
            S = self.similarity_matrix
        
        n = S.shape[1]  # number of target samples
        
        for step in tqdm(range(k), desc="Exact greedy selection"):
            if len(P_actual) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_actual, range(n))]
                mu_P = np.ones(len(P_actual)) / len(P_actual)
                gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, self.reg)
                if isinstance(obj_P, torch.Tensor):
                    obj_P = obj_P.item()
            
            obj_values_actual.append(obj_P)
            best_gain = -np.inf
            best_v = None
            
            for v in set(range(S.shape[0])) - set(P_actual):  # Use S.shape[0] for number of source samples
                P_new = P_actual + [v]
                S_P_new = S[np.ix_(P_new, range(n))]
                mu_P_new = np.ones(len(P_new)) / len(P_new)
                _, obj_P_new = pot_partial_extended(S_P_new, k, mu_P_new, self.reg)
                if isinstance(obj_P_new, torch.Tensor):
                    obj_P_new = obj_P_new.item()
                
                actual = obj_P_new - obj_P
                if actual > best_gain:
                    best_gain = actual
                    best_v = v
            
            P_actual.append(best_v)
            #print(f"Step {step+1}/{k}: Selected {best_v} with gain {best_gain:.6f}")
        
        # Final objective
        S_P = S[np.ix_(P_actual, range(n))]
        mu_P = np.ones(len(P_actual)) / len(P_actual)
        _, obj_P = pot_partial_extended(S_P, k, mu_P, self.reg)
        if isinstance(obj_P, torch.Tensor):
            obj_P = obj_P.item()
        obj_values_actual.append(obj_P)
        
        return P_actual, obj_values_actual
    def _greedy_selection_approx(self, k: int) -> Tuple[List[int], List[float]]:
        """Greedy selection with approximate gain computation."""
        import time
        
        #print("Running approx-gain greedy (library)...")
        P_approx_lib = []
        obj_values_approx_lib = []
        gamma_P = None
        obj_P = 0.0
        
        # Get similarity matrix and dimensions
        if self.use_sparse and CUPY_AVAILABLE:
            S = cp.asnumpy(self.similarity_matrix)  # Convert cupy to numpy for processing
        elif isinstance(self.similarity_matrix, torch.Tensor):
            S = self.similarity_matrix.cpu().numpy()
        else:
            S = self.similarity_matrix
        
        n = S.shape[1]  # number of target samples
        mu_T = k * np.ones(n) / n

        sorted_indices_all = np.argsort(-S, axis=1)  # Sort indices for all rows of S
        sorted_S_all = np.take_along_axis(S, sorted_indices_all, axis=1)  # Sort S along rows
        #print("All of S is sorted")
        
        for step in tqdm(range(k), desc="Approx greedy selection"):
            start_time = time.time()

            if len(P_approx_lib) == 0:
                gamma_P = None
                obj_P = 0.0
            else:
                S_P = S[np.ix_(P_approx_lib, range(n))]
                mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
                #print("POT library called \n")
                gamma_P, obj_P = self.ot_solver(S_P, k, mu_P, self.reg)
                if isinstance(obj_P, torch.Tensor):
                    obj_P = obj_P.item()
                ##print(f"Objective val for current proto at step {step}", obj_P)
            
            obj_values_approx_lib.append(obj_P)
            best_gain = -np.inf
            best_v = None
            
            if gamma_P is None:
                col_sums = np.zeros(n)  # Initialize col_sums to zeros if gamma_P is None
            else:
                if isinstance(gamma_P, torch.Tensor):
                    gamma_P = gamma_P.cpu().numpy()
                col_sums = np.sum(gamma_P, axis=0)

            b = mu_T - col_sums
            b = np.clip(b, 0, None)

            # Optimize the loop over candidates
            candidates = np.array(list(set(range(n)) - set(P_approx_lib)))  # Convert to NumPy array for faster indexing
            sorted_indices_candidates = sorted_indices_all[candidates]  # Precompute sorted indices for candidates
            sorted_S_candidates = sorted_S_all[candidates]  # Precompute sorted similarity vectors for candidates
            
            # Vectorized computation of approximate gains for all candidates
            gains = np.array([
                self._approx_gain(P_approx_lib, gamma_P, v, S, sorted_S_candidates[i], sorted_indices_candidates[i], b, k, self.reg)
                for i, v in enumerate(candidates)
            ])

            # Select the best candidate
            best_gain_idx = np.argmax(gains)
            best_gain = gains[best_gain_idx]
            best_v = candidates[best_gain_idx]

            P_approx_lib.append(best_v)
            end_time = time.time()
            #print(f"Step {step+1}/{k}: Selected {best_v} in {end_time - start_time:.4f} seconds")

        # Final objective
        S_P = S[np.ix_(P_approx_lib, range(n))]
        mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
        _, obj_P = self.ot_solver(S_P, k, mu_P, self.reg)
        if isinstance(obj_P, torch.Tensor):
            obj_P = obj_P.item()
        obj_values_approx_lib.append(obj_P)
        
        return P_approx_lib, obj_values_approx_lib
        

    def _approx_gain(self,P: List[int], gamma_P, v: int, S: np.ndarray, S_a: np.ndarray, sorted_indices: np.ndarray ,b:np.ndarray, k: int, reg: float) -> float:
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
        n = S.shape[1]
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
            alpha = self._optimal_alpha_vectorized(S_a.flatten(), sorted_indices, b, reg)
            gamma_tilde = np.vstack([gamma_P, alpha.reshape(1, n)])
            obj = np.sum(S_P * gamma_P) + np.sum(S_a * alpha)
            mask = gamma_tilde > 0
            entropy = -np.sum(gamma_tilde[mask] * np.log(gamma_tilde[mask]))
            obj = obj + reg * entropy
            mask_old = gamma_P > 0
            entropy_old = -np.sum(gamma_P[mask_old] * np.log(gamma_P[mask_old]))
            obj_old = np.sum(S_P * gamma_P) + reg * entropy_old
            return obj - obj_old



    def _optimal_alpha_vectorized(self, sorted_S_a: np.ndarray, sorted_indices: np.ndarray, b: np.ndarray, reg: float, tol=1e-8) -> np.ndarray:
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




