import torch
import time


def SPOT_GreedySubsetSelection(C, target_marginal, m):
    """
    Greedy subset selection for OT prototype selection using PyTorch.

    Args:
        C (torch.Tensor): Cost matrix of shape (num_sources, num_targets), device-aware.
        target_marginal (torch.Tensor): Row vector of target distribution (1D or 2D shape), non-negative, sums to 1.
        m (int): Number of prototypes to select.

    Returns:
        torch.Tensor: Indices of selected prototypes, shape (m,)
    """
    device = C.device
    target_marginal = target_marginal / target_marginal.sum()
    target_marginal = target_marginal.view(1, -1)  # Ensure shape (1, num_targets)

    num_sources, num_targets = C.shape

    selected_indices = torch.zeros(m, dtype=torch.long, device=device)
    min_costs = torch.full((1, num_targets), 1e6, device=device)
    min_source_indices = torch.zeros((1, num_targets), dtype=torch.long, device=device)

    all_sources = torch.arange(num_sources, device=device)
    chosen_set = set()

    start_time = time.time()
    for step in range(m):
        mask = torch.tensor([i not in chosen_set for i in all_sources.tolist()], device=device)
        remaining_sources = all_sources[mask]

        # Compute gain for each remaining source
        gain_matrix = torch.clamp(min_costs - C, min=0.0)
        gain = gain_matrix @ target_marginal.t()  # Shape: (num_sources, 1)
        gain_values = gain[remaining_sources]

        best_idx = torch.argmax(gain_values)
        chosen = remaining_sources[best_idx]
        selected_indices[step] = chosen
        chosen_set.add(chosen.item())

        # Update min_costs and corresponding indices
        better_mask = (min_costs - C[chosen, :]) > 0
        min_costs[0, better_mask[0]] = C[chosen, better_mask[0]]
        min_source_indices[0, better_mask[0]] = step

    # log_final_transport_plan(min_source_indices, target_marginal, m, num_targets)

    elapsed = time.time() - start_time
    # print("Selected indices:", selected_indices)
    # print("Total time:", elapsed, "seconds")
    return selected_indices


def log_final_transport_plan(min_source_indices, target_marginal, m, num_targets):
    """
    Construct and log the optimal transport plan gammaOpt and its marginals.

    Args:
        min_source_indices (torch.Tensor): Shape (1, num_targets), index of selected prototype per target.
        target_marginal (torch.Tensor): Shape (1, num_targets), marginal over targets.
        m (int): Number of prototypes.
        num_targets (int): Number of target points.
    """
    print("targetMarginal:\n", target_marginal)

    row_indices = min_source_indices[0]  # shape: (num_targets,)
    col_indices = torch.arange(num_targets, device=target_marginal.device)
    values = target_marginal[0]

    gamma_opt = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=(m, num_targets)
    )
    print("gammaOpt (sparse):\n", gamma_opt)

    curr_opt_w = torch.sparse.sum(gamma_opt, dim=1).to_dense().flatten()
    print("currOptw:\n", curr_opt_w)