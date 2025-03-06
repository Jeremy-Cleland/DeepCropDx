import numpy as np
from collections import Counter
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image


def dataset_quality_check(dataloaders, class_info, output_dir=None):
    """
    Run quality checks on dataset and report issues.

    Args:
        dataloaders: Dictionary containing train, val, test dataloaders
        class_info: Dictionary containing class information
        output_dir: Directory to save visualization, if provided

    Returns:
        report_dict: Dictionary with data quality metrics
    """
    report_dict = {}

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize class names
    class_names = []
    if "idx_to_class" in class_info:
        max_idx = max(class_info["idx_to_class"].keys())
        class_names = [
            class_info["idx_to_class"].get(i, f"Unknown-{i}")
            for i in range(max_idx + 1)
        ]

    # Check each split
    for phase in ["train", "val", "test"]:
        if phase not in dataloaders:
            print(f"Warning: {phase} split not found in dataloaders")
            continue

        # Extract labels
        labels = []
        for _, batch_labels in dataloaders[phase]:
            labels.extend(batch_labels.numpy())

        # Calculate class distribution
        class_counts = Counter(labels)
        print(f"\n{phase.capitalize()} set class distribution:")

        phase_report = {
            "total_samples": len(labels),
            "class_distribution": {},
            "imbalance_ratio": 0.0,
            "missing_classes": [],
        }

        # Create visualization data
        classes = []
        counts = []

        # Check for each class
        max_count = 0
        min_count = float("inf")

        for idx in range(len(class_names)):
            count = class_counts.get(idx, 0)

            if count > 0:
                max_count = max(max_count, count)
                min_count = min(min_count, count)

            class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            print(f"  {class_name}: {count} images")

            phase_report["class_distribution"][class_name] = count
            classes.append(class_name)
            counts.append(count)

            if count == 0:
                phase_report["missing_classes"].append(class_name)

        # Calculate imbalance ratio
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            phase_report["imbalance_ratio"] = imbalance_ratio
            print(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")

            # Alert on extreme imbalance
            if imbalance_ratio > 5:
                print(f"WARNING: Extreme class imbalance detected in {phase} set!")
        else:
            print(f"WARNING: Some classes have no samples in the {phase} set!")

        # Save distribution plot if output directory is provided
        if output_dir:
            plt.figure(figsize=(12, 6))

            # Sort by counts for better visualization
            sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
            sorted_classes, sorted_counts = (
                zip(*sorted_data) if sorted_data else ([], [])
            )

            # Create bar plot
            bars = plt.bar(sorted_classes, sorted_counts, color="skyblue")
            plt.xticks(rotation=90)
            plt.title(f"{phase.capitalize()} Set Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Number of Images")
            plt.tight_layout()

            # Save plot
            plt.savefig(
                os.path.join(output_dir, f"{phase}_class_distribution.png"), dpi=300
            )
            plt.close()

            # Create heatmap visualization for imbalance
            if len(class_names) > 1:
                plt.figure(figsize=(12, 10))
                distribution_matrix = np.zeros((len(class_names), 1))
                for i, class_name in enumerate(class_names):
                    if class_name in phase_report["class_distribution"]:
                        distribution_matrix[i, 0] = phase_report["class_distribution"][
                            class_name
                        ]

                sns.heatmap(
                    distribution_matrix,
                    annot=True,
                    fmt="d",
                    cmap="YlGnBu",
                    yticklabels=class_names,
                    xticklabels=["Count"],
                )
                plt.title(f"{phase.capitalize()} Set Class Distribution Heatmap")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"{phase}_distribution_heatmap.png"),
                    dpi=300,
                )
                plt.close()

        # Add to report
        report_dict[phase] = phase_report

    # Check for image quality issues in a sample of images
    if "train" in dataloaders:
        dataloader = dataloaders["train"]

        # Check a sample of images
        sample_size = min(100, len(dataloader.dataset))
        indices = np.random.choice(len(dataloader.dataset), sample_size, replace=False)

        # Statistics for image quality
        zero_pixel_count = 0
        low_contrast_count = 0
        tiny_image_count = 0
        extreme_aspect_ratio = 0

        image_quality_report = {"samples_checked": sample_size, "issues": {}}

        # Process each sample
        for idx in indices:
            try:
                # Get image path
                if hasattr(dataloader.dataset, "images_paths"):
                    img_path = dataloader.dataset.images_paths[idx]
                    # Open original image to check quality
                    img = Image.open(img_path).convert("RGB")

                    # Check image size
                    width, height = img.size
                    if width < 32 or height < 32:
                        tiny_image_count += 1

                    # Check aspect ratio
                    aspect_ratio = max(width, height) / max(
                        1, min(width, height)
                    )  # Avoid division by zero
                    if aspect_ratio > 3:
                        extreme_aspect_ratio += 1

                    # Convert to numpy to check pixel values
                    img_array = np.array(img)

                    # Check for zero/black pixels
                    if np.mean(img_array) < 10:  # very dark image
                        zero_pixel_count += 1

                    # Check contrast
                    if np.std(img_array) < 20:  # Low contrast
                        low_contrast_count += 1
            except Exception as e:
                print(f"Error checking image quality: {str(e)}")

        # Report findings
        if tiny_image_count > 0:
            print(f"WARNING: {tiny_image_count} images are very small (< 32px)")
            image_quality_report["issues"]["tiny_images"] = tiny_image_count

        if extreme_aspect_ratio > 0:
            print(
                f"WARNING: {extreme_aspect_ratio} images have extreme aspect ratios (> 3:1)"
            )
            image_quality_report["issues"][
                "extreme_aspect_ratio"
            ] = extreme_aspect_ratio

        if zero_pixel_count > 0:
            print(
                f"WARNING: {zero_pixel_count} images are very dark (low pixel values)"
            )
            image_quality_report["issues"]["very_dark"] = zero_pixel_count

        if low_contrast_count > 0:
            print(f"WARNING: {low_contrast_count} images have low contrast")
            image_quality_report["issues"]["low_contrast"] = low_contrast_count

        # Add to report
        report_dict["image_quality"] = image_quality_report

    return report_dict


def check_for_data_leakage(train_paths, val_paths, test_paths):
    """
    Check for data leakage between splits (same image in multiple splits).

    Args:
        train_paths: List of image paths in training set
        val_paths: List of image paths in validation set
        test_paths: List of image paths in test set

    Returns:
        leakage_info: Dictionary with information about any leakage detected
    """
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    train_val_intersection = train_set.intersection(val_set)
    train_test_intersection = train_set.intersection(test_set)
    val_test_intersection = val_set.intersection(test_set)

    leakage_info = {
        "train_val_leakage": len(train_val_intersection),
        "train_test_leakage": len(train_test_intersection),
        "val_test_leakage": len(val_test_intersection),
        "leakage_detected": False,
    }

    if train_val_intersection:
        print(
            f"WARNING: Data leakage detected! {len(train_val_intersection)} images are in both train and validation sets"
        )
        leakage_info["leakage_detected"] = True
        leakage_info["train_val_examples"] = list(train_val_intersection)[
            :10
        ]  # First 10 examples

    if train_test_intersection:
        print(
            f"WARNING: Data leakage detected! {len(train_test_intersection)} images are in both train and test sets"
        )
        leakage_info["leakage_detected"] = True
        leakage_info["train_test_examples"] = list(train_test_intersection)[
            :10
        ]  # First 10 examples

    if val_test_intersection:
        print(
            f"WARNING: Data leakage detected! {len(val_test_intersection)} images are in both validation and test sets"
        )
        leakage_info["leakage_detected"] = True
        leakage_info["val_test_examples"] = list(val_test_intersection)[
            :10
        ]  # First 10 examples

    if not leakage_info["leakage_detected"]:
        print("No data leakage detected between splits.")

    return leakage_info


def visualize_batch_with_transforms(
    dataset, transforms_list, num_samples=5, class_names=None
):
    """
    Visualize the effect of different transforms on the same images.

    Args:
        dataset: Dataset to sample images from
        transforms_list: Dictionary of transforms {name: transform_function}
        num_samples: Number of samples to visualize
        class_names: Optional list of class names

    Returns:
        fig: Matplotlib figure with visualization
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(
        num_samples,
        len(transforms_list) + 1,
        figsize=(3 * (len(transforms_list) + 1), 3 * num_samples),
    )

    # If only one sample, make axes 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    for i, transform_name in enumerate(["Original"] + list(transforms_list.keys())):
        axes[0, i].set_title(transform_name)

    # Process each sample
    for i, idx in enumerate(indices):
        img_path = dataset.images_paths[idx]
        label = dataset.labels[idx]
        class_name = (
            class_names[label]
            if class_names and label < len(class_names)
            else f"Class {label}"
        )

        # Original image
        orig_img = Image.open(img_path).convert("RGB")
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_ylabel(class_name)
        axes[i, 0].axis("off")

        # Apply each transform
        for j, (transform_name, transform_fn) in enumerate(transforms_list.items(), 1):
            try:
                transformed = transform_fn(orig_img)

                if isinstance(transformed, torch.Tensor):
                    # Convert tensor to numpy for visualization
                    img_np = transformed.permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                    axes[i, j].imshow(img_np)
                else:
                    axes[i, j].imshow(transformed)

            except Exception as e:
                axes[i, j].text(
                    0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", wrap=True
                )

            axes[i, j].axis("off")

    plt.tight_layout()
    return fig
