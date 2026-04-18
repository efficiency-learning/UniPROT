"""
baselines/main.py
=================
Entry point for baseline prototype selection experiments on TinyImageNet.

Pipeline
--------
1. Extract CNN/ViT features from TinyImageNet using a HuggingFace model.
2. Split the features into a source pool and a target evaluation set.
3. Run prototype selection via SPOT-Greedy and random baselines.
4. Evaluate each set of prototypes with 1-NN classification on the target set
   for several prototype budget sizes.

Usage
-----
    python main.py

Set ``TINYIMAGENET_ROOT`` to the path of your local TinyImageNet directory
before running.
"""

import torch
from loader import get_tinyimagenet_loader
from features import load_feature_extractor, extract_features
from SPOTgreedy import SPOT_GreedySubsetSelection
from evaluation import split_data_percent, run_prototype_selection_eval

# --------------------------------------------------------------------------- #
# Selector wrappers                                                            #
# --------------------------------------------------------------------------- #

# Path to a local copy of TinyImageNet-200 (update before running).
TINYIMAGENET_ROOT = "/home/ganesh/AAAI26/spot/SPOT/python/tiny/tiny-imagenet-200"


def spot_selector(C: torch.Tensor, target_marginal: torch.Tensor, m: int) -> torch.Tensor:
    """Wrap :func:`SPOT_GreedySubsetSelection` to match the selector API.

    Args:
        C: Cost matrix of shape ``(n_source, n_target)``.
        target_marginal: Uniform target distribution of shape ``(n_target,)``.
        m: Number of prototypes to select.

    Returns:
        1-D tensor of ``m`` selected source indices.
    """
    return SPOT_GreedySubsetSelection(C, target_marginal, m)


def random_selector(C: torch.Tensor, target_marginal: torch.Tensor, m: int) -> torch.Tensor:
    """Random baseline: sample *m* source indices uniformly at random.

    Args:
        C: Cost matrix (shape used only to determine the source set size).
        target_marginal: Unused – kept for API compatibility.
        m: Number of prototypes to select.

    Returns:
        1-D tensor of ``m`` randomly sampled source indices.
    """
    return torch.randint(0, C.shape[0], (m,))


# --------------------------------------------------------------------------- #
# Main experiment                                                              #
# --------------------------------------------------------------------------- #

def main():
    """Run the full feature extraction and prototype selection pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Feature extraction ---
    processor, model = load_feature_extractor("microsoft/resnet-50")

    loader, _ = get_tinyimagenet_loader(
        root=TINYIMAGENET_ROOT,
        image_processor=processor,
        batch_size=128,
        split="train",
        num_workers=4,
    )

    features, labels = extract_features(loader, model, device=device)
    # Flatten spatial dimensions if present (e.g. ResNet pooler output is already (N, d)).
    features = features.view(features.size(0), -1)

    print("Features shape:", features.shape)
    print("Labels shape:  ", labels.shape)

    # --- Source / target split ---
    splits = split_data_percent(
        X_all=features,
        y_all=labels,
        source_percent=0.5,
        target_percent=0.5,
        seed=0,
    )

    # --- Prototype selection evaluation ---
    metric = "euclidean"
    prototype_budgets = [100, 200, 500, 1000, 5000]

    print(f"\nDistance metric: {metric}")

    run_prototype_selection_eval(
        source_x=splits["source_x"],
        source_y=splits["source_y"],
        target_x=splits["target_x"],
        target_y=splits["target_y"],
        selector_fn=spot_selector,
        method="spot",
        distance_metric=metric,
        num_prototypes=prototype_budgets,
    )

    random_acc = run_prototype_selection_eval(
        source_x=splits["source_x"],
        source_y=splits["source_y"],
        target_x=splits["target_x"],
        target_y=splits["target_y"],
        selector_fn=random_selector,
        method="random",
        distance_metric=metric,
        num_prototypes=prototype_budgets,
    )
    print("Random selector accuracies:", random_acc)


if __name__ == "__main__":
    main()
