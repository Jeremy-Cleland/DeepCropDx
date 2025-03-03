import os
import glob
import random
import subprocess
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from collections import Counter


class CropDiseaseDataset(Dataset):
    """Dataset class for loading crop disease images"""

    def __init__(self, images_paths, labels, transform=None):
        self.images_paths = images_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataset_from_directory(
    data_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    save_splits=True,
    output_dir=None,
):
    """
    Create datasets from a directory of images organized by class folders
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    ), "Ratios must sum to 1"

    # Initialize dictionaries to store class information
    class_counts = {}
    class_to_idx = {}

    # Get all image files recursively in the data directory
    all_images = []
    all_labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)

                # Determine the class from the parent directory name
                parent_dir = os.path.basename(os.path.dirname(full_path))

                # Store the image path
                all_images.append(full_path)

                # For simplicity, use the parent directory as the class name
                if parent_dir not in class_counts:
                    class_counts[parent_dir] = 1
                    class_to_idx[parent_dir] = len(class_to_idx)
                else:
                    class_counts[parent_dir] += 1

                # Add the label using the class-to-index mapping
                all_labels.append(class_to_idx[parent_dir])

    # Create reverse mapping
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"Class {class_name}: {count} images")

    # Handle special case where all data goes to test set
    if (
        test_ratio > 0.999
    ):  # Using 0.999 instead of 1.0 to account for floating-point errors
        print("Using all data for testing (test_ratio=1.0)")
        train_images, train_labels = [], []
        val_images, val_labels = [], []
        test_images, test_labels = all_images, all_labels
    else:
        # Split the dataset normally
        random.seed(seed)

        try:
            train_images, test_images, train_labels, test_labels = train_test_split(
                all_images,
                all_labels,
                test_size=(val_ratio + test_ratio),
                stratify=all_labels,
                random_state=seed,
            )

            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_images, test_images, val_labels, test_labels = train_test_split(
                test_images,
                test_labels,
                test_size=(1 - val_ratio_adjusted),
                stratify=test_labels,
                random_state=seed,
            )
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to random (non-stratified) splitting")

            train_images, test_images, train_labels, test_labels = train_test_split(
                all_images,
                all_labels,
                test_size=(val_ratio + test_ratio),
                stratify=None,
                random_state=seed,
            )

            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_images, test_images, val_labels, test_labels = train_test_split(
                test_images,
                test_labels,
                test_size=(1 - val_ratio_adjusted),
                stratify=None,
                random_state=seed,
            )

    print(f"\nTotal images: {len(all_images)}")
    print(
        f"Training: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}"
    )

    dataset_info = {
        "train": {"images": train_images, "labels": train_labels},
        "val": {"images": val_images, "labels": val_labels},
        "test": {"images": test_images, "labels": test_labels},
    }

    class_info = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_counts": class_counts,
    }

    # Save the splits to disk if requested
    if save_splits and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save mappings
        with open(os.path.join(output_dir, "class_mapping.txt"), "w") as f:
            for class_name, idx in class_to_idx.items():
                f.write(f"{class_name},{idx}\n")

        # Save splits to CSV
        for split_name, split_data in dataset_info.items():
            split_df = pd.DataFrame(
                {"image_path": split_data["images"], "label": split_data["labels"]}
            )
            split_df.to_csv(
                os.path.join(output_dir, f"{split_name}_set.csv"), index=False
            )

        print(f"\nDataset splits saved to {output_dir}")

    return dataset_info, class_info


def get_data_transforms(img_size=224):
    """
    Get enhanced data transformations for training and validation/testing
    """
    # Mean and std from ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return {"train": train_transforms, "val": val_transforms, "test": val_transforms}


def create_dataloaders(
    dataset_info, transforms, batch_size=32, num_workers=None, device=None
):
    """
    Create PyTorch DataLoaders for training, validation and test sets
    with optimizations for Apple Silicon

    Args:
        dataset_info (dict): Dictionary containing dataset information
        transforms (dict): Dictionary of transforms for each split
        batch_size (int): Batch size
        num_workers (int, optional): Number of worker processes. If None, automatically determined.
        device (torch.device, optional): Device being used

    Returns:
        dict: Dictionary of DataLoaders
    """
    dataloaders = {}

    # Auto-determine optimal number of workers if not specified
    if num_workers is None:
        num_workers = min(
            8, os.cpu_count() or 4
        )  # Adjust based on your specific Mac model

    # Set pin_memory and other optimizations based on device
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2

    # Specific optimizations for different devices
    if device is not None:
        if device.type == "mps":
            # For Apple Silicon, pinned memory can be used but its benefit depends on the PyTorch version
            # Check PyTorch version - newer versions have better pin_memory support for MPS
            try:
                pytorch_version = torch.__version__
                if pytorch_version >= "2.2.0":
                    # PyTorch 2.2+ has better MPS+pin_memory support
                    pin_memory = True
                    print("Using pinned memory with MPS (PyTorch 2.2+)")

                    # Check if this is M4 or newer
                    import platform

                    if platform.system() == "Darwin" and platform.machine() == "arm64":
                        try:

                            result = subprocess.run(
                                ["sysctl", "hw.model"], capture_output=True, text=True
                            )
                            if result.returncode == 0:
                                hardware_info = result.stdout.strip()
                                if any(x in hardware_info for x in ["M4", "M3", "M2"]):
                                    # Latest M-series optimizations
                                    prefetch_factor = (
                                        3  # More aggressive prefetching for M4 Max
                                    )
                        except:
                            pass
                else:
                    # Older PyTorch - pin_memory might not be beneficial with MPS
                    pin_memory = False
                    print("Disabled pinned memory for MPS (PyTorch < 2.2)")
            except:
                # If cannot determine version, default to False for safety
                pin_memory = False

        elif device.type == "cuda":
            # CUDA optimizations
            pin_memory = True
            prefetch_factor = 4  # More aggressive prefetching for CUDA

    for split in ["train", "val", "test"]:
        dataset = CropDiseaseDataset(
            images_paths=dataset_info[split]["images"],
            labels=dataset_info[split]["labels"],
            transform=transforms[split],
        )

        shuffle = True if split == "train" else False

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,  # Keep worker processes alive between iterations
            prefetch_factor=prefetch_factor,  # Number of batches loaded in advance by each worker
        )

    return dataloaders


def process_data(
    input_dir, output_dir, img_size=224, batch_size=32, num_workers=None, device=None
):
    """
    Process the crop disease dataset and create DataLoaders
    """
    # Create dataset splits
    dataset_info, class_info = create_dataset_from_directory(
        input_dir, save_splits=True, output_dir=output_dir
    )

    # Get data transforms
    transforms = get_data_transforms(img_size=img_size)

    # Create DataLoaders with device info
    dataloaders = create_dataloaders(
        dataset_info,
        transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    return dataloaders, class_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process crop disease dataset")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to raw data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save processed data"
    )
    parser.add_argument("--img_size", type=int, default=224, help="Target image size")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoaders"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )

    args = parser.parse_args()

    dataloaders, class_info = process_data(
        args.input_dir,
        args.output_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
    )

    print("\nDataset processing complete.")
    print(f"Number of classes: {len(class_info['class_to_idx'])}")
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
