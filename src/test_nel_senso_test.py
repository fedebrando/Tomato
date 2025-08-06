import os
import torch
from torchvision import transforms
from dataset import TomatoDataset
from collections import Counter

def test_dataset_loading():
    root_dir = '../images'
    seed = 42
    test_pct = 0.15

    transform = transforms.ToTensor()

    # Load datasets with same seed and transform
    train_dataset = TomatoDataset(root_dir=root_dir, mode='train', transform=transform, seed=seed, test_pct=test_pct)
    test_dataset = TomatoDataset(root_dir=root_dir, mode='test', transform=transform, seed=seed, test_pct=test_pct)

    # === Basic checks ===
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    assert len(train_dataset) + len(test_dataset) > 0, "Dataset is empty!"
    assert len(set(train_dataset.samples).intersection(set(test_dataset.samples))) == 0, "Train and test sets overlap!"

    # === Check sample loading ===
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image not converted to tensor"
    assert isinstance(label, torch.Tensor), "Label not a tensor"
    assert img.shape[0] == 3, "Image should have 3 channels"
    assert label.ndim == 2, "Label should be 2D (H, W)"
    assert img.shape[1:] == label.shape, f"Image and label size mismatch: {img.shape[1:]} vs {label.shape}"

    # === Check class values are in expected range ===
    unique_classes = torch.unique(label).tolist()
    print(f"Unique class values in one label: {unique_classes}")
    assert all(0 <= int(c) <= 5 for c in unique_classes), "Unexpected class value in label"

    print("✅ Dataset loading test passed.")

if __name__ == "__main__":
    test_dataset_loading()
