"""
baselines/features.py
=====================
Feature extraction utilities using HuggingFace Transformers.

Supports any model accessible via ``AutoModel`` and ``AutoImageProcessor``
(e.g. ResNet-50, ViT).  Pooled features are extracted from the model and
collected into contiguous tensors for downstream prototype selection.
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm


def load_feature_extractor(model_name: str = "microsoft/resnet-50"):
    """Load an image processor and vision model from the HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier (e.g. "google/vit-base-patch16-224").

    Returns:
        processor: ``AutoImageProcessor`` instance for pre-processing images.
        model: ``AutoModel`` instance set to evaluation mode.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


@torch.no_grad()
def extract_features(dataloader, model, device: str = "cuda"):
    """Extract pooled feature vectors from every batch in *dataloader*.

    The function uses ``pooler_output`` when available (e.g. ResNet), and
    falls back to the ``[CLS]`` token (first position of ``last_hidden_state``)
    for transformer models such as ViT.

    Args:
        dataloader: PyTorch ``DataLoader`` yielding ``(images, labels)`` batches.
            Images must already be pre-processed pixel tensors (i.e. passed
            through the corresponding ``AutoImageProcessor``).
        model: A HuggingFace vision model.
        device: Target device for inference (``"cuda"`` or ``"cpu"``).

    Returns:
        features: Float tensor of shape ``(N, d)`` containing all pooled
            embeddings concatenated in dataloader order.
        labels: Long tensor of shape ``(N,)`` containing the corresponding
            class labels.
    """
    all_features = []
    all_labels = []

    model = model.to(device)

    for batch in tqdm(dataloader, desc="Extracting features"):
        images, labels = batch
        images = images.to(device)

        outputs = model(pixel_values=images)
        # Use the pooler output when available; otherwise fall back to [CLS] token.
        pooled = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            else outputs.last_hidden_state[:, 0]
        )
        all_features.append(pooled.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)
