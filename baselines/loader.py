"""
baselines/loader.py
===================
Dataset loading utilities for the TinyImageNet benchmark.

Builds ``torchvision.datasets.ImageFolder`` datasets and ``DataLoader``
instances for TinyImageNet, automatically adapting pre-processing
transforms to the size expected by a given HuggingFace image processor.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from PIL import Image


def get_tinyimagenet_transform(image_processor, split: str = "train"):
    """Build a torchvision ``Compose`` transform matching *image_processor*'s expectations.

    Handles ``image_processor.size`` in ``dict``, ``tuple/list``, and ``int``
    formats produced by different HuggingFace model families.

    Args:
        image_processor: A HuggingFace ``AutoImageProcessor`` whose ``size``,
            ``image_mean``, and ``image_std`` attributes are used to construct
            the normalisation step.
        split: Dataset split – used for future extension (e.g. augmentation on
            ``"train"``). Currently unused.

    Returns:
        A ``transforms.Compose`` pipeline: Resize → ToTensor → Normalize.
    """
    size = image_processor.size
    if isinstance(size, dict):
        height = size.get("height", 224)
        width = size.get("width", 224)
    elif isinstance(size, (tuple, list)):
        height, width = size
    elif isinstance(size, int):
        height = width = size
    else:
        raise ValueError(f"Unrecognized image_processor.size format: {size}")

    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])


def get_tinyimagenet_dataset(root: str, image_processor, split: str = "train"):
    """Construct a ``torchvision.datasets.ImageFolder`` for TinyImageNet.

    Expected directory layout::

        <root>/train/<class_id>/<image>.JPEG
        <root>/val/images/<image>.JPEG

    Args:
        root: Path to the TinyImageNet root directory.
        image_processor: HuggingFace image processor used to determine
            resize dimensions and normalisation statistics.
        split: ``"train"`` or ``"val"``.

    Returns:
        A ``torchvision.datasets.ImageFolder`` dataset with the appropriate
        pre-processing transform applied.
    """
    subdir = "train" if split == "train" else "val"
    path = os.path.join(root, subdir)
    transform = get_tinyimagenet_transform(image_processor, split)
    return datasets.ImageFolder(path, transform=transform)


def get_tinyimagenet_loader(
    root: str,
    image_processor,
    batch_size: int = 64,
    split: str = "train",
    num_workers: int = 4,
    shuffle: bool = True,
):
    """Construct a ``DataLoader`` for TinyImageNet together with the underlying dataset.

    Args:
        root: Path to the TinyImageNet root directory.
        image_processor: HuggingFace image processor used to configure transforms.
        batch_size: Number of samples per mini-batch.
        split: ``"train"`` or ``"val"``.
        num_workers: Number of worker processes for parallel data loading.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        loader: A configured ``torch.utils.data.DataLoader``.
        dataset: The underlying ``ImageFolder`` dataset instance.
    """
    dataset = get_tinyimagenet_dataset(root, image_processor, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), dataset
