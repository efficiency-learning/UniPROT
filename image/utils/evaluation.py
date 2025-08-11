import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Callable, Dict


import torch
import torch.nn.functional as F


def compute_cost_matrix(X_source: torch.Tensor, X_target: torch.Tensor, metric: str = "euclidean", return_sims=False) -> torch.Tensor:
    """
    Computes cost matrix between two sets of vectors.

    Args:
        X_source: (n, d)
        X_target: (m, d)
        metric: 'euclidean', 'dot', or 'cosine'

    Returns:
        torch.Tensor of shape (n, m)
    """
    if metric == "euclidean":
        if(return_sims): return torch.cdist(X_source, X_target, p=2), -torch.cdist(X_source, X_target, p=2)
        return torch.cdist(X_source, X_target, p=2)

    elif metric == "dot":
        # Negative dot product as cost (maximize dot = minimize negative dot)
        if(return_sims): return - X_source @ X_target.T, X_source @ X_target.T
        return - X_source @ X_target.T

    elif metric == "cosine":
        # Normalize
        X1 = F.normalize(X_source, dim=1)
        X2 = F.normalize(X_target, dim=1)
        cosine_sim = X1 @ X2.T
        if(return_sims): return 1-cosine_sim, cosine_sim
        return 1 - cosine_sim  # cost = 1 - similarity

    else:
        raise ValueError(f"Unsupported metric: {metric}")



def get_uniform_marginal(n: int, device: torch.device = "cpu") -> torch.Tensor:
    mu = torch.ones(n, device=device)
    return mu / mu.sum()


def split_data_percent(
    X_all: torch.Tensor,
    y_all: torch.Tensor,
    source_percent: float,
    target_percent: float,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Splits the full dataset into source and target sets by percent.

    Returns:
        dict with keys: source_x, source_y, target_x, target_y
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = X_all.shape[0]
    num_source = int(N * source_percent)
    assert 0 < num_source < N, "source_percent must be between 0 and 1"

    remaining = N - num_source
    num_target = int(remaining * target_percent)
    assert num_target > 0, "target_percent too small after removing source samples"

    # Split indices
    indices = np.random.permutation(N)
    source_idx = indices[:num_source]
    remaining_idx = indices[num_source:]
    target_idx = np.random.choice(remaining_idx, size=num_target, replace=False)

    return {
        "source_x": X_all[source_idx],
        "source_y": y_all[source_idx],
        "target_x": X_all[target_idx],
        "target_y": y_all[target_idx],
    }


def run_prototype_selection_eval(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    selector_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    method: str,
    num_prototypes: list[int],
    distance_metric: str = "euclidean",
) -> float:
    """
    Runs prototype selection and evaluates 1-NN classification on target set.

    Args:
        source_x, source_y: source dataset
        target_x, target_y: target dataset
        selector_fn: function (C, target_marginal, m) -> indices into source_x
        num_prototypes: number of prototypes to select

    Returns:
        accuracy (float)
    """
    C = compute_cost_matrix(source_x, target_x, metric=distance_metric)
    target_marginal = get_uniform_marginal(C.shape[1], device=C.device)
    acc = []
    for num in num_prototypes:
        selected_indices = selector_fn(C, target_marginal, num)
        selected_X = source_x[selected_indices]
        selected_y = source_y[selected_indices]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(selected_X.cpu().numpy(), selected_y.cpu().numpy())
        y_pred = knn.predict(target_x.cpu().numpy())
        _acc = accuracy_score(target_y.cpu().numpy(), y_pred)
        acc.append(_acc)
        print(f"Accuracy {method}@{num}: {_acc}")
    return acc



