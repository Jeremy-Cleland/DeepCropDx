# Evaluation and benchmarking routines
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import glob

# Import project modules
from src.data.dataset import create_dataset_from_directory, CropDiseaseDataset
from src.models.efficientnet import (
    create_efficientnet_model,
    create_efficientnet_b3_model,
)
from src.models.mobilenet import (
    create_mobilenet_model,
    create_mobilenet_v3_model,
    create_mobilenet_model_with_attention,
)
from src.models.resnet import (
    create_resnet18_model,
    create_resnet50_model,
    create_resnet_with_attention,
)
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_classification_report,
    visualize_model_predictions,
    plot_examples_of_misclassifications,
    get_target_layer_for_model,
    GradCAM,
)


def find_best_model_version(model_dir, model_name):
    """
    Find the best model version in a directory

    Args:
        model_dir (str): Directory containing model files
        model_name (str): Base name of the model

    Returns:
        str: Path to the best versioned model, or None if not found
    """
    # First try to find versioned best models
    versioned_models = glob.glob(os.path.join(model_dir, f"{model_name}_v*_best.pth"))
    if versioned_models:
        # Sort by version number (highest first)
        try:
            versioned_models.sort(
                key=lambda x: int(os.path.basename(x).split("_v")[1].split("_")[0]),
                reverse=True,
            )
            return versioned_models[0]
        except (IndexError, ValueError):
            pass

    # Then try final models
    versioned_models = glob.glob(os.path.join(model_dir, f"{model_name}_v*_final.pth"))
    if versioned_models:
        try:
            versioned_models.sort(
                key=lambda x: int(os.path.basename(x).split("_v")[1].split("_")[0]),
                reverse=True,
            )
            return versioned_models[0]
        except (IndexError, ValueError):
            pass

    # Fall back to standard model files
    standard_models = ["best_model.pth", "final_model.pth"]
    for model_file in standard_models:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            return model_path

    return None


def find_model_path(model_path):
    """
    Find the model path, handle directory or file input

    Args:
        model_path (str): Path to model file or directory

    Returns:
        str: Path to model file
    """
    if os.path.isfile(model_path):
        return model_path

    if os.path.isdir(model_path):
        # If it's a directory, try to find a model file
        # First, look for model files in the directory
        model_name = os.path.basename(model_path)
        model_dir = model_path

        # Check if we're in a models/ subdirectory
        if os.path.basename(model_dir) == "models":
            model_name = os.path.basename(os.path.dirname(model_dir))
        else:
            # Check if there's a models/ subdirectory
            models_dir = os.path.join(model_dir, "models")
            if os.path.isdir(models_dir):
                model_dir = models_dir

        # Find the best version
        best_model = find_best_model_version(model_dir, model_name)
        if best_model:
            return best_model

    # If we got here, we couldn't find a suitable model file
    raise FileNotFoundError(f"Could not find a suitable model file at {model_path}")


def load_model(model_path, device="cpu"):
    """
    Load a trained model from checkpoint
    Args:
        model_path (str): Path to model checkpoint or directory containing models
        device (str): Device to load model on ('cpu', 'cuda', 'mps')
    Returns:
        tuple: (model, class_to_idx)
    """
    print(f"Looking for model at {model_path}...")

    # Handle if a directory was passed
    model_path = find_model_path(model_path)
    print(f"Loading model from {model_path}...")

    # Use safe_torch_load utility
    from src.utils.model_utils import safe_torch_load
    checkpoint = safe_torch_load(model_path, map_location=device, weights_only=True, fallback_to_unsafe=True)

    # Extract metadata
    if isinstance(checkpoint, dict):
        # Check for metadata in enhanced format
        if "model_type" in checkpoint:
            model_type = checkpoint["model_type"]
            print(f"Found model type in checkpoint: {model_type}")
        else:
            model_type = None

        if "class_to_idx" in checkpoint:
            class_to_idx = checkpoint["class_to_idx"]
            print(f"Found class_to_idx in checkpoint with {len(class_to_idx)} classes")
        else:
            class_to_idx = None

        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint

        # Print model metadata if available
        if "created_at" in checkpoint:
            print(f"Model created: {checkpoint['created_at']}")
        if "version" in checkpoint:
            print(f"Model version: {checkpoint['version']}")
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            print(
                f"Model metrics: Accuracy={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}"
            )
    else:
        # Legacy format with no metadata
        model_state = checkpoint
        model_type = None
        class_to_idx = None

    # If model type wasn't in the checkpoint, try to determine from filename
    if model_type is None:
        model_filename = os.path.basename(model_path)
        model_parent = os.path.basename(os.path.dirname(model_path))

        # Try different potential sources for model type
        for name in [model_filename, model_parent]:
            if "efficientnet_b3" in name:
                model_type = "efficientnet_b3"
                break
            elif "efficientnet" in name:
                model_type = "efficientnet"
                break
            elif "mobilenet_v3_large" in name:
                model_type = "mobilenet_v3_large"
                break
            elif "mobilenet_v3_small" in name:
                model_type = "mobilenet_v3_small"
                break
            elif "mobilenet_attention" in name:
                model_type = "mobilenet_attention"
                break
            elif "mobilenet" in name:
                model_type = "mobilenet"
                break
            elif "resnet50" in name or "resnet_50" in name:
                model_type = "resnet50"
                break
            elif "resnet_attention" in name:
                model_type = "resnet_attention"
                break
            elif "resnet18" in name or "resnet_18" in name:
                model_type = "resnet18"
                break

        if model_type is None:
            raise ValueError(f"Could not determine model type from {model_path}")

    # Get class mapping if it wasn't in the checkpoint
    if class_to_idx is None:
        # Try to find class mapping file in the model directory
        model_dir = os.path.dirname(model_path)
        class_mapping_path = os.path.join(model_dir, "class_mapping.txt")

        if os.path.exists(class_mapping_path):
            class_to_idx = {}
            with open(class_mapping_path, "r") as f:
                for line in f:
                    if line.strip():
                        class_name, idx = line.strip().split(",")
                        class_to_idx[class_name] = int(idx)
            print(f"Loaded class mapping from file with {len(class_to_idx)} classes")
        else:
            # Try to find class_mapping.txt in parent directory
            parent_dir = os.path.dirname(model_dir)
            class_mapping_path = os.path.join(
                parent_dir, "processed_data", "class_mapping.txt"
            )
            if os.path.exists(class_mapping_path):
                class_to_idx = {}
                with open(class_mapping_path, "r") as f:
                    for line in f:
                        if line.strip():
                            class_name, idx = line.strip().split(",")
                            class_to_idx[class_name] = int(idx)
                print(
                    f"Loaded class mapping from parent directory with {len(class_to_idx)} classes"
                )
            else:
                # Last attempt: check the registry
                registry_path = os.path.join(
                    os.path.dirname(parent_dir), "model_registry.json"
                )
                if os.path.exists(registry_path):
                    try:
                        with open(registry_path, "r") as f:
                            registry = json.load(f)
                        for entry in registry:
                            if entry.get("path") == os.path.relpath(
                                model_path, os.path.dirname(registry_path)
                            ):
                                if "class_to_idx" in entry:
                                    class_to_idx = entry["class_to_idx"]
                                    print(
                                        f"Loaded class mapping from registry with {len(class_to_idx)} classes"
                                    )
                                    break
                    except Exception as e:
                        print(f"Error loading from registry: {str(e)}")

                if class_to_idx is None:
                    print(
                        "Warning: No class mapping found! Model might not work correctly."
                    )
                    class_to_idx = {}

    # Create model based on determined type
    num_classes = len(class_to_idx) if class_to_idx else 0
    if num_classes == 0:
        print("Warning: No classes found! Using default value of 1000.")
        num_classes = 1000

    print(f"Creating {model_type} model with {num_classes} classes")

    if model_type == "efficientnet":
        model = create_efficientnet_model(num_classes)
    elif model_type == "efficientnet_b3":
        model = create_efficientnet_b3_model(num_classes)
    elif model_type == "mobilenet":
        model = create_mobilenet_model(num_classes)
    elif model_type == "mobilenet_v3_small":
        model = create_mobilenet_v3_model(num_classes, model_size="small")
    elif model_type == "mobilenet_v3_large":
        model = create_mobilenet_v3_model(num_classes, model_size="large")
    elif model_type == "mobilenet_attention":
        model = create_mobilenet_model_with_attention(num_classes)
    elif model_type == "resnet18":
        model = create_resnet18_model(num_classes)
    elif model_type == "resnet50":
        model = create_resnet50_model(num_classes)
    elif model_type == "resnet_attention":
        model = create_resnet_with_attention(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load model weights
    try:
        model.load_state_dict(model_state)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        print("Attempting to adjust state dict keys...")

        # Sometimes there's a prefix issue with DataParallel models
        adjusted_state_dict = {}
        for k, v in model_state.items():
            if k.startswith("module."):
                adjusted_state_dict[k[7:]] = v
            else:
                adjusted_state_dict[k] = v

        model.load_state_dict(adjusted_state_dict)
        print("Model weights loaded successfully after adjustment")

    model = model.to(device)
    model.eval()
    return model, class_to_idx


def get_transforms(img_size=224):
    """
    Get data transforms for evaluation
    Args:
        img_size (int): Image size
    Returns:
        torchvision.transforms: Transforms for evaluation
    """
    # Mean and std from ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Evaluate model on dataset
    Args:
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        device (str): Device to evaluate on
        class_names (list, optional): List of class names
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            probs = probabilities.cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)

    # Print metrics
    print("\nEvaluation Results:")
    print_metrics(metrics, class_names)

    return {
        "metrics": metrics,
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
    }


def predict_single_image(model, image_path, class_to_idx, img_size=224, device="cpu"):
    """
    Predict class for a single image
    Args:
        model (torch.nn.Module): Trained model
        image_path (str): Path to image file
        class_to_idx (dict): Class mapping
        img_size (int): Image size
        device (str): Device to run on
    Returns:
        tuple: (predicted_class, confidence, class_probabilities)
    """
    # Create reverse mapping
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = idx_to_class[prediction.item()]
    confidence = confidence.item()
    class_probabilities = {
        idx_to_class[i]: prob for i, prob in enumerate(probabilities[0].cpu().numpy())
    }

    return predicted_class, confidence, class_probabilities


def generate_visualizations(
    evaluation_results, model, dataloader, class_names, device, output_dir, model_type=None
):
    """
    Generate and save visualizations
    Args:
        evaluation_results (dict): Results from evaluate_model
        model (torch.nn.Module): Trained model
        dataloader (torch.utils.data.DataLoader): DataLoader
        class_names (list): List of class names
        device (str): Device to run on
        output_dir (str): Directory to save visualizations
        model_type (str, optional): Model type for GradCAM layer selection
    """
    # Check if output_dir already contains the model name
    model_name = os.path.basename(os.path.dirname(output_dir))
    base_dir = os.path.basename(output_dir)
    if base_dir == model_name:
        # The directory already includes the model name, keep as is
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Create directory structure without redundancy
        os.makedirs(output_dir, exist_ok=True)

    metrics = evaluation_results["metrics"]
    labels = evaluation_results["labels"]
    preds = evaluation_results["predictions"]

    # 1. Confusion Matrix
    fig = plot_confusion_matrix(labels, preds, class_names)
    fig.savefig(
        os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # 2. Classification Report
    fig = plot_classification_report(labels, preds, class_names)
    fig.savefig(
        os.path.join(output_dir, "classification_report.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Example Predictions
    fig = visualize_model_predictions(model, dataloader, class_names, device)
    fig.savefig(
        os.path.join(output_dir, "example_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. Misclassified Examples
    fig = plot_examples_of_misclassifications(model, dataloader, class_names, device)
    fig.savefig(
        os.path.join(output_dir, "misclassified_examples.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    
    # 5. GradCAM Visualizations
    try:
        # Create GradCAM directory
        gradcam_dir = os.path.join(output_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Try to determine model type from model if not provided
        if model_type is None:
            if hasattr(model, 'model_type'):
                model_type = model.model_type
            elif hasattr(model, 'name'):
                model_type = model.name
        
        # Find appropriate target layer
        try:
            target_layer = get_target_layer_for_model(model, model_type if model_type else "unknown")
            print(f"Using target layer for GradCAM: {target_layer}")
        except Exception as e:
            print(f"Could not identify target layer for GradCAM: {str(e)}")
            target_layer = None
        
        if target_layer is not None:
            # Initialize GradCAM
            grad_cam = GradCAM(model, target_layer)
            
            # Get sample images for each class
            sample_images = {}
            sample_labels = {}
            
            for inputs, labels in dataloader:
                for i, (img, label) in enumerate(zip(inputs, labels)):
                    label_idx = label.item()
                    if label_idx not in sample_images:
                        sample_images[label_idx] = img
                        sample_labels[label_idx] = label_idx
                    
                    if len(sample_images) == len(class_names):
                        break
                
                if len(sample_images) == len(class_names):
                    break
            
            # Generate and save GradCAM visualizations for each class
            for label_idx, img in sample_images.items():
                if label_idx >= len(class_names):
                    continue  # Skip if label index is out of bounds
                    
                class_name = class_names[label_idx]
                img_tensor = img.unsqueeze(0).to(device)
                
                try:
                    # Generate GradCAM
                    cam = grad_cam.generate_cam(img_tensor, target_class=label_idx)
                    
                    # Convert image for visualization
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    # Denormalize the image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    # Create a figure with two side-by-side subplots
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original image
                    axes[0].imshow(img_np)
                    axes[0].set_title(f"Original: {class_name}")
                    axes[0].axis("off")
                    
                    # GradCAM overlay
                    axes[1].imshow(img_np)
                    axes[1].imshow(cam, alpha=0.5, cmap='jet')
                    axes[1].set_title(f"GradCAM: {class_name}")
                    axes[1].axis("off")
                    
                    # Save the figure
                    safe_class_name = class_name.replace('/', '_').replace(' ', '_')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(gradcam_dir, f"gradcam_{safe_class_name}.png"),
                        dpi=300,
                        bbox_inches="tight"
                    )
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error generating GradCAM for class {class_name}: {str(e)}")
            
            # Clean up hooks
            grad_cam.remove_hooks()
            print(f"GradCAM visualizations saved to {gradcam_dir}")
            
    except Exception as e:
        print(f"Error generating GradCAM visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or directory",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to test data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/evaluations",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_mps", action="store_true", help="Disable MPS (Apple Silicon GPU)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    args = parser.parse_args()

    # Create output directory
    model_file = os.path.basename(args.model_path)

    # Handle case where model_path is a directory
    if os.path.isdir(args.model_path):
        model_file = os.path.basename(args.model_path)

    # Handle versioned model names
    if "_v" in model_file:
        # Extract base name before version
        experiment_name = model_file.split("_v")[0]
    else:
        # Remove extension if it's a file
        experiment_name = os.path.splitext(model_file)[0]

    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set device with priority: CUDA > MPS > CPU
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not args.no_mps
    ):
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Load model
    model, class_to_idx = load_model(args.model_path, device)

    # Reverse class mapping
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    class_names = [
        idx_to_class.get(i, f"Unknown-{i}") for i in range(len(idx_to_class))
    ]

    # Print model information
    print(f"Model loaded with {len(class_names)} classes:")
    for idx, name in sorted(idx_to_class.items()):
        print(f"  {idx}: {name}")

    # Create test dataset
    transform = get_transforms(args.img_size)

    # Check if data_dir is a directory with class folders or a csv file
    if os.path.isdir(args.data_dir):
        # Handle directory with class folders
        print(f"Creating dataset from directory: {args.data_dir}")
        dataset_info, _ = create_dataset_from_directory(
            args.data_dir, train_ratio=0, val_ratio=0, test_ratio=1.0, save_splits=False
        )
        test_dataset = CropDiseaseDataset(
            images_paths=dataset_info["test"]["images"],
            labels=dataset_info["test"]["labels"],
            transform=transform,
        )
    else:
        # Handle CSV file with image paths and labels
        print(f"Loading dataset from CSV: {args.data_dir}")
        test_df = pd.read_csv(args.data_dir)
        test_dataset = CropDiseaseDataset(
            images_paths=test_df["image_path"].tolist(),
            labels=test_df["label"].tolist(),
            transform=transform,
        )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test dataset created with {len(test_dataset)} images")

    # Evaluate model
    evaluation_results = evaluate_model(model, test_loader, device, class_names)

    # Save metrics to file
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Value": [
                evaluation_results["metrics"]["accuracy"],
                evaluation_results["metrics"]["precision"],
                evaluation_results["metrics"]["recall"],
                evaluation_results["metrics"]["f1"],
            ],
        }
    )
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.csv')}")

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "True_Label": [
                class_names[label] for label in evaluation_results["labels"]
            ],
            "Predicted_Label": [
                class_names[pred] for pred in evaluation_results["predictions"]
            ],
            "Correct": evaluation_results["labels"]
            == evaluation_results["predictions"],
        }
    )
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")

    # Generate visualizations if requested
    if args.visualize:
        # Try to determine model type for better GradCAM
        model_type = None
        # Try from the model name
        if "_" in experiment_name:
            # Handle names like "efficientnet_b0"
            model_type = experiment_name
        
        generate_visualizations(
            evaluation_results,
            model,
            test_loader,
            class_names,
            device,
            os.path.join(output_dir, "visualizations"),
            model_type=model_type
        )
        print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")

    # Save evaluation summary to JSON
    evaluation_summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "evaluated_at": datetime.now().isoformat(),
        "device": str(device),
        "metrics": {
            "accuracy": float(evaluation_results["metrics"]["accuracy"]),
            "precision": float(evaluation_results["metrics"]["precision"]),
            "recall": float(evaluation_results["metrics"]["recall"]),
            "f1": float(evaluation_results["metrics"]["f1"]),
        },
        "num_samples": len(test_dataset),
        "class_count": len(class_names),
    }

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print("Evaluation complete!")


if __name__ == "__main__":
    from datetime import datetime

    main()
