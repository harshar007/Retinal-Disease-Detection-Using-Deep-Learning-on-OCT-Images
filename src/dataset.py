"""
Module: src.dataset
Data loading, preprocessing, augmentation, and PyTorch Dataset/DataLoader pipeline
for Retinal Optical Coherence Tomography (OCT) image classification.
"""

import os
import io
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedShuffleSplit

# 4 Target Classes
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "CNV": "Choroidal Neovascularization: Pathologic growth of abnormal blood vessels beneath the retina, causing severe vision loss and subretinal fluid accumulation.",
    "DME": "Diabetic Macular Edema: Accumulation of fluid in the macula due to leaking blood vessels from diabetic retinopathy, leading to retinal thickening.",
    "DRUSEN": "Drusen: Yellow extracellular lipid deposits beneath the retinal pigment epithelium (RPE), characteristic early sign of Age-related Macular Degeneration (AMD).",
    "NORMAL": "Normal Retina: Preserved retinal layer architecture with continuous foveal contour and no intraretinal or subretinal fluid/deposits."
}

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    """
    Returns train and evaluation torchvision transforms.
    Train includes data augmentation (flips, slight rotations, affine scaling, color jitter).
    Evaluation uses deterministic resizing and normalization.
    """
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


class OCTDataset(Dataset):
    """
    PyTorch Dataset for Retinal OCT Images.
    Supports in-memory PIL images or raw image bytes.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        if isinstance(item, bytes):
            img = Image.open(io.BytesIO(item)).convert("RGB")
        elif isinstance(item, Image.Image):
            img = item.convert("RGB")
        elif isinstance(item, str):
            img = Image.open(item).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(item)}")

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.labels[idx])
        return img, label


def load_kermany_oct_data(cache_dir=None, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Downloads/loads the Kermany2017 OCT dataset from Hugging Face cache.
    Returns:
        dict: {
            'train_images': [...], 'train_labels': [...],
            'val_images': [...], 'val_labels': [...],
            'test_images': [...], 'test_labels': [...]
        }
    """
    parquet_path = hf_hub_download(
        repo_id="zacharielegault/Kermany2017-OCT",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=cache_dir
    )

    table = pq.read_table(parquet_path)
    images_bytes = [table["image"][i]["bytes"].as_py() for i in range(len(table))]
    labels = [table["label"][i].as_py() for i in range(len(table))]

    # Stratified split: first split test, then train/val
    labels_np = np.array(labels)
    total_samples = len(labels)

    # Split off test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(range(total_samples), labels_np))

    # Split train and val from train_val
    val_rel_ratio = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=random_state)
    train_sub_idx, val_sub_idx = next(sss_val.split(train_val_idx, labels_np[train_val_idx]))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    data_splits = {
        "train_images": [images_bytes[i] for i in train_idx],
        "train_labels": [labels[i] for i in train_idx],
        "val_images": [images_bytes[i] for i in val_idx],
        "val_labels": [labels[i] for i in val_idx],
        "test_images": [images_bytes[i] for i in test_idx],
        "test_labels": [labels[i] for i in test_idx],
    }

    print(f"Dataset loaded: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return data_splits


def get_oct_dataloaders(data_splits, batch_size=32, img_size=224, num_workers=0):
    """
    Creates train, val, and test PyTorch DataLoaders with appropriate transforms.
    """
    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_ds = OCTDataset(data_splits["train_images"], data_splits["train_labels"], transform=train_tf)
    val_ds = OCTDataset(data_splits["val_images"], data_splits["val_labels"], transform=eval_tf)
    test_ds = OCTDataset(data_splits["test_images"], data_splits["test_labels"], transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
