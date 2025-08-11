import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from PIL import Image


def get_tinyimagenet_transform(image_processor, split="train"):
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


def get_tinyimagenet_dataset(root, image_processor, split="train"):
    """
    Returns a torchvision ImageFolder dataset for TinyImageNet.
    Assumes directory layout: root/train/class_x/xxx.JPEG, root/val/images/...
    """
    subdir = "train" if split == "train" else "val"
    path = os.path.join(root, subdir)
    transform = get_tinyimagenet_transform(image_processor, split)
    return datasets.ImageFolder(path, transform=transform)


def get_tinyimagenet_loader(root, image_processor, batch_size=64, split="train", num_workers=4, shuffle=True):
    dataset = get_tinyimagenet_dataset(root, image_processor, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), dataset
