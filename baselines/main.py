import torch
from loader import get_tinyimagenet_loader
from features import load_feature_extractor, extract_features
from SPOTgreedy import SPOT_GreedySubsetSelection

from evaluation import split_data_percent, run_prototype_selection_eval

# Define selector wrapper
def spot_selector(C, target_marginal, m):
    return SPOT_GreedySubsetSelection(C, target_marginal, m)

def random_selector(C, target_marginal, m):
    return torch.randint(0, C.shape[0], (m,))

def main():
    processor, model = load_feature_extractor("microsoft/resnet-50")  # or e.g. "google/vit-base-patch16-224"

    # Load TinyImageNet train loader
    loader, dataset = get_tinyimagenet_loader(
        root="/home/ganesh/AAAI26/spot/SPOT/python/tiny/tiny-imagenet-200",
        image_processor=processor,
        batch_size=128,
        split="train",
        num_workers=4
    )

    features, labels = extract_features(loader, model, device="cuda" if torch.cuda.is_available() else "cpu")
    features = features.view(features.size(0), -1)


    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    # Step 1: Split data (reuse this across methods!)
    splits = split_data_percent(
        X_all=features,
        y_all=labels,
        source_percent=0.5,
        target_percent=0.5,
        seed=0,
    )
    metric = "euclidean"
    print(f"Metric {metric}")

    accuracy = run_prototype_selection_eval(
        source_x=splits["source_x"],
        source_y=splits["source_y"],
        target_x=splits["target_x"],
        target_y=splits["target_y"],
        selector_fn=spot_selector,
        method = "spot",
        distance_metric=metric,
        num_prototypes=[100,200,500, 1000, 5000],
    )
    accuracy = run_prototype_selection_eval(
        source_x=splits["source_x"],
        source_y=splits["source_y"],
        target_x=splits["target_x"],
        target_y=splits["target_y"],
        selector_fn=random_selector,
        method = "random",
        distance_metric=metric,
        num_prototypes=[100,200,500, 1000, 5000],
    )
    print(f"accuracy random",accuracy)
    
if __name__ == "__main__":
    main()
