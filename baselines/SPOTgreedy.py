"""
baselines/SPOTgreedy.py
=======================
Greedy prototype selection via the SPOT (Submodular Prototype OT) algorithm.

The algorithm iteratively selects the source point that maximises the
submodular coverage objective:

    F(S) = sum_j min_{i in S} C[i, j] * targetMarginal[j]

where C is the source-to-target cost matrix and targetMarginal is the
(typically uniform) target distribution.

Reference
---------
SPOT: Scalable Prototype selection via Optimal Transport (AAAI 2024).
"""

import numpy as np
import scipy
from scipy import sparse
import time


def SPOT_GreedySubsetSelection(
    C: np.ndarray,
    target_marginal: np.ndarray | None,
    m: int,
) -> np.ndarray:
    """Select *m* prototypes from the source set via greedy SPOT.

    The algorithm maintains a running minimum cost from each target point to
    the already-selected subset and, at each step, chooses the source point
    whose addition yields the largest reduction in the expected transport cost.

    Args:
        C: Cost matrix of shape ``(numY, numX)`` where ``numY`` is the number
            of source points and ``numX`` the number of target points.  Can be
            a ``numpy.ndarray`` or a ``torch.Tensor`` (converted internally).
        target_marginal: Target distribution of shape ``(numX,)`` summing to 1.
            If ``None``, a uniform distribution over target points is used.
        m: Number of prototypes to select. Must satisfy ``m < numY``.

    Returns:
        selected: 1-D integer array of length *m* containing the row indices
            (into the source set) of the selected prototypes.
    """
    # Accept torch tensors by converting to numpy.
    if hasattr(C, "cpu"):
        C = C.cpu().numpy()
    if target_marginal is not None and hasattr(target_marginal, "cpu"):
        target_marginal = target_marginal.cpu().numpy()

    numY, numX = C.shape

    # Use provided target marginal or fall back to uniform.
    if target_marginal is None:
        target_marginal = np.ones(numX) / numX
    target_marginal = target_marginal.reshape(1, numX)

    # Initialise bookkeeping structures.
    S = np.zeros((1, m), dtype=int)
    sizeS = 0
    curr_min_costs = np.ones((1, numX)) * 1_000_000  # running min cost per target
    curr_min_source_idx = np.zeros((1, numX), dtype=int)  # which prototype covers each target
    chosen_elements: list[int] = []
    all_source_idx = np.arange(numY)

    start = time.time()

    while sizeS < m:
        # Candidates are source points not yet selected.
        remaining = all_source_idx[~np.in1d(all_source_idx, np.array(chosen_elements))]

        # Marginal gain for each remaining candidate:
        # gain_i = sum_j max(0, curr_min_cost_j - C[i, j]) * target_marginal_j
        marginal_gains = np.maximum(curr_min_costs - C, 0)
        marginal_gains = np.matmul(marginal_gains, target_marginal.T)
        increment_values = marginal_gains[remaining]

        # Select the candidate with the highest marginal gain.
        best_local_idx = np.argmax(increment_values)
        chosen = remaining[best_local_idx]
        chosen_elements.append(chosen)
        S[0][sizeS] = chosen

        # Update the running minimum cost and which prototype covers each target.
        improvement_mask = (curr_min_costs - C[chosen, :]) > 0
        curr_min_costs[0, improvement_mask[0]] = C[chosen][improvement_mask[0]]
        curr_min_source_idx[0, improvement_mask[0]] = sizeS

        # On the final iteration compute the optimal transport plan.
        if sizeS == m - 1:
            gamma_opt = sparse.csr_matrix(
                (target_marginal[0], (curr_min_source_idx[0], range(numX))),
                shape=(m, numX),
            )
            curr_opt_weights = np.sum(gamma_opt, axis=1).flatten()

        sizeS += 1

    elapsed = time.time() - start
    print(f"SPOT selected {m} prototypes in {elapsed:.2f}s: {S[0]}")
    return S[0]
