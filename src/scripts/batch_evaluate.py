# src/utils/batch_evaluate.py
import os
import argparse
import json
import glob
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import CropDiseaseDataset, create_dataset_from_directory
from src.training.evaluate import (
    get_transforms,
    load_model,
    evaluate_model,
    generate_visualizations,
)
from src.utils.model_comparison import ModelComparisonReport
from src.utils.logger import TrainingLogger
from src.utils.model_utils import safe_torch_load


class EvaluationLogger(TrainingLogger):
    """Extension of TrainingLogger for evaluation purposes"""

    def __init__(self, log_dir, experiment_name=None):
        super().__init__(log_dir, experiment_name)
        self.history = {}
        self.metrics = None

    def log_metrics(self, metrics):
        """Log evaluation metrics and save them"""
        self.metrics = metrics

    def generate_metrics_plot(self, class_names=None, confusion_matrix=None):
        """Create and save a visualization of the evaluation metrics"""
        if self.metrics is None:
            self.logger.warning("No metrics to plot")
            return

        # Create a figure with two parts:
        # 1. Bar chart of overall metrics (accuracy, precision, recall, f1)
        # 2. Heatmap of confusion matrix if provided

        if confusion_matrix is not None and class_names is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))

        # Plot metrics as bar chart
        metrics = ["accuracy", "precision", "recall", "f1"]
        values = [self.metrics[m] for m in metrics]

        ax1.bar(metrics, values, color="blue")
        ax1.set_ylim(0, 1.0)
        ax1.set_title("Evaluation Metrics")
        ax1.set_ylabel("Score")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Add values on top of bars
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f"{v:.4f}", ha="center")

        # Plot confusion matrix as heatmap if provided
        if confusion_matrix is not None and class_names is not None:
            # Normalize confusion matrix
            cm_norm = (
                confusion_matrix.astype("float")
                / confusion_matrix.sum(axis=1)[:, np.newaxis]
            )

            # Plot heatmap
            sns.heatmap(
                cm_norm,
                annot=False,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax2,
            )
            ax2.set_title("Confusion Matrix (Normalized)")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            plt.setp(
                ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

        plt.tight_layout()

        # Save the plot
        plt.savefig(
            os.path.join(
                self.log_dir, f"{self.experiment_name}_evaluation_metrics.png"
            ),
            dpi=100,
        )
        plt.close()

        self.logger.info(
            f"Metrics visualization saved to {os.path.join(self.log_dir, f'{self.experiment_name}_evaluation_metrics.png')}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate multiple models")
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Directory containing model files"
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
        "--report_dir",
        type=str,
        default="reports/comparisons",
        help="Path to save comparison report",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for data loading"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_mps", action="store_true", help="Disable MPS (Apple Silicon GPU)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--registry_path",
        type=str,
        default=None,
        help="Path to model registry file (if not in models_dir parent)",
    )
    parser.add_argument(
        "--use_registry",
        action="store_true",
        help="Use the model registry to find models instead of file pattern",
    )
    parser.add_argument(
        "--models_pattern",
        type=str,
        default="**/*_v*_*.pth",
        help="Pattern to match model files (e.g., '*_v*_*.pth' for versioned models)",
    )
    parser.add_argument(
        "--legacy_pattern",
        type=str,
        default="**/best_model.pth",
        help="Pattern to match legacy model files as fallback",
    )
    parser.add_argument(
        "--report_title",
        type=str,
        default="Model Comparison Report",
        help="Title for the generated report",
    )
    parser.add_argument(
        "--report_id",
        type=str,
        default=None,
        help="Custom ID for the report (default: timestamp)",
    )
    return parser.parse_args()


def find_latest_version_models(models_dir, experiment_dirs=None):
    """
    Find the latest version of each model from each experiment directory

    Args:
        models_dir (str): Base directory containing model experiments
        experiment_dirs (list, optional): List of specific experiment directories to check

    Returns:
        list: Paths to latest version models
    """
    latest_models = []

    # If no specific experiment dirs provided, get all subdirectories
    if experiment_dirs is None:
        experiment_dirs = [
            os.path.join(models_dir, d)
            for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]

    # Process each experiment directory
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)

        # First check for models directory
        models_subdir = os.path.join(exp_dir, "models")
        if os.path.isdir(models_subdir):
            target_dir = models_subdir
        else:
            target_dir = exp_dir

        # Find versioned models
        versioned_models = glob.glob(
            os.path.join(target_dir, f"{exp_name}_v*_best.pth")
        )
        if not versioned_models:
            versioned_models = glob.glob(
                os.path.join(target_dir, f"{exp_name}_v*_final.pth")
            )

        if versioned_models:
            # Sort by version number (highest first)
            try:
                versioned_models.sort(
                    key=lambda x: int(os.path.basename(x).split("_v")[1].split("_")[0]),
                    reverse=True,
                )
                latest_models.append(versioned_models[0])
                continue
            except (IndexError, ValueError):
                pass

        # Fallback to standard models if no versioned models found
        standard_models = ["best_model.pth", "final_model.pth"]
        for model_file in standard_models:
            model_path = os.path.join(target_dir, model_file)
            if os.path.exists(model_path):
                latest_models.append(model_path)
                break

    return latest_models


def find_models_from_registry(registry_path):
    """
    Find models using the model registry

    Args:
        registry_path (str): Path to model registry JSON file

    Returns:
        list: Paths to models from registry
    """
    if not os.path.exists(registry_path):
        print(f"Registry file not found: {registry_path}")
        return []

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding registry file: {registry_path}")
        return []

    base_dir = os.path.dirname(registry_path)
    model_paths = []

    for entry in registry:
        # First try best model path if available
        if "best_model_path" in entry and entry["best_model_path"]:
            best_path = os.path.join(base_dir, entry["best_model_path"])
            if os.path.exists(best_path):
                model_paths.append(best_path)
                continue

        # Then try regular path
        if "path" in entry and entry["path"]:
            model_path = os.path.join(base_dir, entry["path"])
            if os.path.exists(model_path):
                model_paths.append(model_path)

    return model_paths


def find_models(
    models_dir, version_pattern="**/*_v*_*.pth", legacy_pattern="**/best_model.pth"
):
    """
    Find all model files matching the patterns

    Args:
        models_dir (str): Directory to search for models
        version_pattern (str): Pattern to match versioned model files
        legacy_pattern (str): Pattern to match legacy model files

    Returns:
        list: Paths to model files
    """
    # First try to find versioned models
    models = []
    versioned_models = glob.glob(
        os.path.join(models_dir, version_pattern), recursive=True
    )

    # Group by base name to only keep latest versions
    model_groups = {}
    for model_path in versioned_models:
        model_file = os.path.basename(model_path)
        # Extract base name and version
        if "_v" in model_file:
            base_name = model_file.split("_v")[0]
            try:
                version = int(model_file.split("_v")[1].split("_")[0])

                # Check if it's a best or final model
                is_best = "_best.pth" in model_file

                # Add to group, preferring best models and higher versions
                key = f"{os.path.dirname(model_path)}/{base_name}"
                current = model_groups.get(
                    key, {"path": None, "version": -1, "is_best": False}
                )

                # Take this model if it's best and the current is not, or if it's the same type but higher version
                if (is_best and not current["is_best"]) or (
                    is_best == current["is_best"] and version > current["version"]
                ):
                    model_groups[key] = {
                        "path": model_path,
                        "version": version,
                        "is_best": is_best,
                    }
            except (ValueError, IndexError):
                # If can't parse version, just add the model
                models.append(model_path)

    # Add latest version from each group
    for group_info in model_groups.values():
        models.append(group_info["path"])

    # If no versioned models found, try legacy pattern
    if not models:
        models = glob.glob(os.path.join(models_dir, legacy_pattern), recursive=True)

    # Try another approach: find experiment directories and get latest model from each
    if not models:
        print("No models found with glob patterns, trying directory-based search...")
        models = find_latest_version_models(models_dir)

    return models


def get_model_display_name(model_path):
    """
    Get a clean display name for the model

    Args:
        model_path (str): Path to model file

    Returns:
        str: Display name for the model
    """
    model_file = os.path.basename(model_path)

    # Handle versioned model
    if "_v" in model_file:
        try:
            # Extract experiment name and version
            base_name = model_file.split("_v")[0]
            version = model_file.split("_v")[1].split("_")[0]

            # Check if it's best or final
            if "_best.pth" in model_file:
                return f"{base_name} v{version} (best)"
            elif "_final.pth" in model_file:
                return f"{base_name} v{version} (final)"
            else:
                return f"{base_name} v{version}"
        except (IndexError, ValueError):
            pass

    # Special case for best/final models
    if model_file == "best_model.pth" or model_file == "final_model.pth":
        # Try to get experiment name from parent directory
        parent_dir = os.path.basename(os.path.dirname(model_path))
        if parent_dir == "models":
            # Go up one more level
            exp_dir = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            return f"{exp_dir} ({model_file.replace('_model.pth', '')})"
        return f"{parent_dir} ({model_file.replace('_model.pth', '')})"

    # Default: just use filename without extension
    return os.path.splitext(model_file)[0]


def evaluate_models(args):
    """Evaluate multiple models and generate comparison report"""
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

    # Find model files based on method
    if args.use_registry:
        # Find registry file
        registry_path = args.registry_path
        if registry_path is None:
            # Try to find registry in parent of models_dir
            registry_path = os.path.join(
                os.path.dirname(args.models_dir), "model_registry.json"
            )

        if os.path.exists(registry_path):
            model_files = find_models_from_registry(registry_path)
            print(f"Found {len(model_files)} models from registry: {registry_path}")
        else:
            print(f"Registry file not found: {registry_path}")
            return
    else:
        # Use pattern-based approach
        model_files = find_models(
            args.models_dir,
            version_pattern=args.models_pattern,
            legacy_pattern=args.legacy_pattern,
        )
        print(f"Found {len(model_files)} model files")

    if not model_files:
        print(f"No model files found in {args.models_dir}")
        return

    # Create dataset
    transform = get_transforms(args.img_size)

    # Check if data_dir is a directory with class folders or a csv file
    if os.path.isdir(args.data_dir):
        # Handle directory with class folders - for simplicity, we'll use the evaluate.py's approach
        print(f"Creating dataset from directory: {args.data_dir}")
        dataset_info, class_info = create_dataset_from_directory(
            args.data_dir, train_ratio=0, val_ratio=0, test_ratio=1.0, save_splits=False
        )

        test_dataset = CropDiseaseDataset(
            images_paths=dataset_info["test"]["images"],
            labels=dataset_info["test"]["labels"],
            transform=transform,
        )

        # Get class names from class_info
        idx_to_class = class_info["idx_to_class"]
        num_classes = len(idx_to_class)
        class_names = [idx_to_class.get(i, f"Unknown_{i}") for i in range(num_classes)]
    else:
        # Handle CSV file with image paths and labels
        print(f"Loading dataset from CSV: {args.data_dir}")
        test_df = pd.read_csv(args.data_dir)

        test_dataset = CropDiseaseDataset(
            images_paths=test_df["image_path"].tolist(),
            labels=test_df["label"].tolist(),
            transform=transform,
        )

        # We'll need to load class names from elsewhere or infer from labels
        class_names = None

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test dataset created with {len(test_dataset)} images")

    # Initialize model comparison report
    report_id = args.report_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_report = ModelComparisonReport(args.report_dir, report_id=report_id)

    # Evaluate each model
    successful_models = []
    failed_models = []

    for model_path in tqdm(model_files, desc="Evaluating models"):
        # Get clean display name for the model
        model_display_name = get_model_display_name(model_path)
        print(f"\nEvaluating {model_display_name} from {model_path}")

        # Use model_display_name to create output directory name, removing any parenthetical parts
        output_name = model_display_name.split(" (")[0]
        # Replace spaces with underscores for filesystem compatibility
        output_name = output_name.replace(" ", "_")

        model_output_dir = os.path.join(args.output_dir, output_name)

        # Create logs directory inside the evaluations/model_name folder
        logs_dir = os.path.join(model_output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize logger with model name as experiment name
        logger = EvaluationLogger(logs_dir, os.path.basename(output_name))

        try:
            # Use evaluate.py's function to evaluate

            # Log start of evaluation
            logger.logger.info(logger.log_separator)
            logger.logger.info(f"Starting evaluation: {model_display_name}")
            logger.logger.info(f"Model path: {model_path}")
            logger.logger.info(f"Dataset size: {len(test_dataset)} images")
            logger.logger.info(logger.log_separator)

            # Load model
            logger.logger.info(f"Looking for model at {model_path}...")
            model, class_to_idx = load_model(model_path, device)
            logger.logger.info("Model loaded successfully")

            # If we didn't get class names from the dataset, try to get them from the model
            if class_names is None and class_to_idx:
                idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
                class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
                logger.logger.info(f"Found {len(class_names)} classes in model")

            # Evaluate model
            logger.logger.info("Starting model evaluation...")
            evaluation_results = evaluate_model(model, test_loader, device, class_names)
            logger.logger.info("Evaluation complete")

            # Create output directory
            os.makedirs(model_output_dir, exist_ok=True)

            # Log evaluation results
            logger.logger.info(logger.log_separator)
            logger.logger.info("Evaluation Results:")
            logger.logger.info(
                f"Accuracy: {evaluation_results['metrics']['accuracy']:.4f}"
            )
            logger.logger.info(
                f"Precision: {evaluation_results['metrics']['precision']:.4f}"
            )
            logger.logger.info(f"Recall: {evaluation_results['metrics']['recall']:.4f}")
            logger.logger.info(f"F1 Score: {evaluation_results['metrics']['f1']:.4f}")
            logger.logger.info(logger.log_separator)

            # Save metrics for plotting
            logger.log_metrics(evaluation_results["metrics"])
            # Extract confusion matrix once
            confusion_matrix = evaluation_results.get("confusion_matrix")
            if confusion_matrix is None:
                confusion_matrix = evaluation_results["metrics"].get("confusion_matrix")

            # Generate and save metrics plot
            logger.generate_metrics_plot(class_names, confusion_matrix)

            # Log confusion matrix
            logger.logger.info("Confusion Matrix:")
            if confusion_matrix is not None:
                for i, row in enumerate(confusion_matrix):
                    logger.logger.info(f"{class_names[i]}: {row}")
            else:
                logger.logger.info("Confusion matrix not available")
            logger.logger.info(logger.log_separator)

            # Save metrics to file
            metrics = ["accuracy", "precision", "recall", "f1"]
            metrics_df = pd.DataFrame(
                {
                    "Metric": [m.capitalize() for m in metrics],
                    "Value": [evaluation_results["metrics"][m] for m in metrics],
                }
            )

            metrics_path = os.path.join(model_output_dir, "metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            logger.logger.info(f"Metrics saved to {metrics_path}")

            # Save evaluation summary with model info
            model_info = {
                "model_path": model_path,
                "model_name": model_display_name,
                "evaluated_at": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": float(evaluation_results["metrics"]["accuracy"]),
                    "precision": float(evaluation_results["metrics"]["precision"]),
                    "recall": float(evaluation_results["metrics"]["recall"]),
                    "f1": float(evaluation_results["metrics"]["f1"]),
                },
            }

            with open(
                os.path.join(model_output_dir, "evaluation_summary.json"), "w"
            ) as f:
                json.dump(model_info, f, indent=2)
            logger.logger.info(
                f"Evaluation summary saved to {os.path.join(model_output_dir, 'evaluation_summary.json')}"
            )

            # Generate visualizations if requested
            if args.visualize:
                logger.logger.info("Generating visualizations...")
                # Try to determine model type from checkpoint
                model_type = None
                try:
                    # Check if checkpoint has model_type information
                    checkpoint = safe_torch_load(
                        model_path,
                        map_location="cpu",
                        weights_only=True,
                        fallback_to_unsafe=True,
                    )
                    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
                        model_type = checkpoint["model_type"]
                        logger.logger.info(
                            f"Found model type in checkpoint: {model_type}"
                        )
                    elif (
                        isinstance(checkpoint, dict) and "training_params" in checkpoint
                    ):
                        model_type = checkpoint["training_params"].get("model_type")
                        logger.logger.info(
                            f"Found model type in training params: {model_type}"
                        )
                except Exception as e:
                    logger.logger.warning(
                        f"Error loading checkpoint for model type detection: {str(e)}"
                    )

                visualizations_dir = os.path.join(model_output_dir, "visualizations")
                generate_visualizations(
                    evaluation_results,
                    model,
                    test_loader,
                    class_names,
                    device,
                    visualizations_dir,
                    model_type=model_type,
                )
                logger.logger.info(f"Visualizations saved to {visualizations_dir}")

            # Add model results to comparison
            comparison_report.add_model_results(
                model_display_name, metrics_path, model_info=model_info
            )
            successful_models.append(
                (model_display_name, model_path, evaluation_results["metrics"]["f1"])
            )

            # Log completion
            logger.logger.info(logger.log_separator)
            logger.logger.info(
                f"Evaluation of {model_display_name} completed successfully"
            )
            logger.logger.info(f"F1 Score: {evaluation_results['metrics']['f1']:.4f}")
            logger.logger.info(logger.log_separator)

        except Exception as e:
            error_msg = f"Error evaluating {model_display_name}: {str(e)}"
            print(error_msg)

            # Log the error
            logger.logger.error(logger.log_separator)
            logger.logger.error(error_msg)

            error_traceback = traceback.format_exc()
            logger.logger.error(error_traceback)
            logger.logger.error(logger.log_separator)

            failed_models.append((model_display_name, model_path, str(e)))
            traceback.print_exc()

    # Generate comparison report
    report_path = comparison_report.generate_report(title=args.report_title)

    print(f"\nAll evaluations complete!")
    print(f"Successfully evaluated: {len(successful_models)} models")
    print(f"Failed to evaluate: {len(failed_models)} models")

    if failed_models:
        print("\nFailed models:")
        for name, path, error in failed_models:
            print(f"  - {name}: {error}")

    print(f"\nComparison report generated: {report_path}")

    # Return best model based on F1 score
    if successful_models:
        best_model = max(successful_models, key=lambda x: x[2])
        print(
            f"\nBest model based on F1 score: {best_model[0]} (F1 = {best_model[2]:.4f})"
        )
        print(f"Path: {best_model[1]}")
        return best_model[1]
    else:
        print("No successful evaluations to determine best model")
        return None


def main():
    args = parse_args()
    evaluate_models(args)


if __name__ == "__main__":
    main()
