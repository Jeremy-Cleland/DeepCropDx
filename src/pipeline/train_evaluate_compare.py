# src/pipeline/train_evaluate_compare.py
import os
import argparse
import subprocess
import json
from datetime import datetime
import glob
import logging
from src.utils.log_config import configure_logger, get_level_from_string

# Apple Silicon optimization - prevent thread contention
os.environ["OMP_NUM_THREADS"] = "16"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and compare multiple models"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Path to save models and results",
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Path to save reports"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="JSON configuration file with model specifications",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for training and evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64 for M-series Macs)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_mps", action="store_true", help="Disable MPS (Apple Silicon GPU)"
    )
    parser.add_argument(
        "--use_mps",
        action="store_true",
        help="Enable MPS (Apple Silicon GPU) - default for Apple Silicon",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of data loading workers (0 for no multiprocessing)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only evaluate existing models",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="crop_disease_detection",
        help="Project name for the report",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use Automatic Mixed Precision training (faster with minimal accuracy impact)",
    )
    parser.add_argument(
        "--mps_fallback",
        action="store_true",
        help="Use CPU fallback for operations not supported in MPS",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient operations (especially helpful for Apple Silicon)",
    )
    parser.add_argument(
        "--cache_dataset",
        action="store_true",
        help="Cache processed dataset in memory for faster training",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for faster data transfer to GPU (helps on M-series chips)",
    )
    parser.add_argument(
        "--mps_graph",
        action="store_true",
        help="Enable MPS graph mode for faster training on Apple Silicon",
    )
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Run learning rate finder before training to find optimal learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=3,
        help="Keep only top K checkpoints to save disk space",
    )
    parser.add_argument(
        "--optimize_for_m_series",
        action="store_true",
        help="Apply specific optimizations for M-series Apple Silicon chips (M1/M2/M3/M4)",
    )
    return parser.parse_args()


def load_model_configs(config_file=None):
    """
    Load model configurations from JSON file or use default configurations
    """
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)

    # Default model configurations
    return [
        {
            "name": "efficientnet_b0",
            "model": "efficientnet",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": True,
        },
        {
            "name": "efficientnet_b3",
            "model": "efficientnet_b3",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": True,
        },
        {
            "name": "mobilenet_v2",
            "model": "mobilenet",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": True,
        },
        {
            "name": "resnet18",
            "model": "resnet18",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": True,
        },
        {
            "name": "resnet_attention",
            "model": "resnet_attention",
            "img_size": 224,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": False,
        },
        {
            "name": "mobilenet_v3_small",
            "model": "mobilenet_v3_small",
            "img_size": 224,
            "batch_size": 64,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "use_weights": True,
            "freeze_backbone": True,
        },
    ]


def train_model(config, args):
    """
    Train a model with the specified configuration
    """
    print(f"\n{'='*80}")
    print(f"Training model: {config['name']}")
    print(f"{'='*80}")

    # Find existing versions of this model to create a versioned experiment directory
    model_name = config["name"]
    model_dirs = glob.glob(os.path.join(args.output_dir, f"{model_name}_v*"))

    if model_dirs:
        # Find the highest version number
        versions = []
        for dir_path in model_dirs:
            try:
                # Extract version number from directory name
                dirname = os.path.basename(dir_path)
                version_part = dirname.split("_v")[1]
                version = int(version_part)
                versions.append(version)
            except (IndexError, ValueError):
                continue

        next_version = max(versions) + 1 if versions else 1
    else:
        # No existing versions
        next_version = 1

    # Create versioned experiment directory
    versioned_model_name = f"{model_name}_v{next_version}"
    experiment_dir = os.path.join(args.output_dir, versioned_model_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Creating new model version: {versioned_model_name}")

    # Build command
    cmd = [
        "python",
        "-m",
        "src.training.train",
        "--data_dir",
        args.data_dir,
        "--output_dir",
        experiment_dir,
        "--model",
        config["model"],
        "--img_size",
        str(config.get("img_size", args.img_size)),
        "--batch_size",
        str(config.get("batch_size", args.batch_size)),
        "--num_workers",
        str(config.get("num_workers", args.num_workers)),
        "--epochs",
        str(config.get("epochs", args.epochs)),
        "--lr",
        str(config.get("lr", 0.001)),
        "--weight_decay",
        str(config.get("weight_decay", 1e-4)),
        "--experiment_name",
        model_name,  # Use base model name to avoid nesting
    ]

    # Add optional flags
    if config.get("use_weights", True):
        cmd.append("--use_weights")

    if config.get("freeze_backbone", True):
        cmd.append("--freeze_backbone")

    if args.no_cuda:
        cmd.append("--no_cuda")

    if args.no_mps:
        cmd.append("--no_mps")
    elif args.use_mps:  # Use the new argument
        cmd.append("--use_mps")

    # Add MPS fallback if enabled
    if args.mps_fallback and not args.no_mps:
        cmd.append("--mps_fallback")

    if args.use_amp:
        cmd.append("--use_amp")

    if args.memory_efficient:
        cmd.append("--memory_efficient")

    if args.cache_dataset:
        cmd.append("--cache_dataset")

    if args.mps_graph and not args.no_mps:
        cmd.append("--mps_graph")

    if args.pin_memory:
        cmd.append("--pin_memory")

    if args.optimize_for_m_series:
        cmd.append("--optimize_for_m_series")

    if args.find_lr:
        cmd.append("--find_lr")

    cmd.extend(["--patience", str(args.patience)])

    if args.keep_top_k > 0:
        cmd.extend(["--keep_top_k", str(args.keep_top_k)])

    # Add M4 optimizations if running on Apple Silicon
    try:
        import platform

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # This is Apple Silicon - apply specific optimizations
            print("Detected Apple Silicon - applying M-series optimizations")
            if "--optimize_for_m_series" not in cmd:
                cmd.append("--optimize_for_m_series")
    except:
        # If check fails, don't add the flag
        pass

    # Execute command
    result = subprocess.run(cmd)

    # Return success/failure and model path
    if result.returncode == 0:
        # The models are saved in experiment_dir/models
        # Try finding the best model file first
        best_model_path = os.path.join(
            experiment_dir, "models", f"{versioned_model_name}_best.pth"
        )

        if os.path.exists(best_model_path):
            print(f"Found best model: {best_model_path}")
            return True, best_model_path, versioned_model_name

        # Fallback to best_model.pth
        best_model_path = os.path.join(experiment_dir, "models", "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Found best model: {best_model_path}")
            return True, best_model_path, versioned_model_name

        # If all else fails, search for any pth files
        model_files = glob.glob(os.path.join(experiment_dir, "models", "*.pth"))
        if model_files:
            print(f"Found model file: {model_files[0]}")
            return True, model_files[0], versioned_model_name

        print(
            f"Warning: No model files found in {os.path.join(experiment_dir, 'models')}"
        )
        return True, None, versioned_model_name
    else:
        print(f"Training failed for {versioned_model_name}")
        return False, None, versioned_model_name


def evaluate_all_models(models_dir, data_dir, args):
    """
    Evaluate all trained models and generate comparison report
    """
    print(f"\n{'='*80}")
    print(f"Evaluating all models")
    print(f"{'='*80}")

    # Build command
    cmd = [
        "python",
        "-m",
        "src.scripts.batch_evaluate",
        "--models_dir",
        models_dir,
        "--data_dir",
        data_dir,
        "--output_dir",
        os.path.join(args.report_dir, "evaluations"),
        "--report_dir",
        os.path.join(args.report_dir, "comparisons"),
        "--img_size",
        str(args.img_size),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--report_title",
        f"{args.project_name} - Model Comparison Report",
    ]

    # Add optional flags
    if args.visualize:
        cmd.append("--visualize")

    if args.no_cuda:
        cmd.append("--no_cuda")

    if args.no_mps:
        cmd.append("--no_mps")

    # Execute command
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Evaluation failed")
        return False

    return True


def run_model_comparison(args, trained_model_names=None):
    """
    Run the model comparison script to generate comprehensive comparison reports
    """
    print(f"\n{'='*80}")
    print(f"Generating model comparison reports")
    print(f"{'='*80}")

    evals_dir = os.path.join(args.report_dir, "evaluations")
    output_dir = os.path.join(args.report_dir, "comparisons")

    # Build command
    cmd = [
        "python",
        "-m",
        "src.scripts.compare_models",
        "--evaluations_dir",
        evals_dir,
        "--output_dir",
        output_dir,
        "--report_title",
        f"{args.project_name} - Model Comparison Report",
    ]

    # Add specific models to compare if provided
    if trained_model_names:
        cmd.extend(["--models"] + trained_model_names)

    # Execute command
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"Model comparison completed successfully")
        print(f"Reports available at: {output_dir}")
        return True
    else:
        print(f"Model comparison failed with exit code {result.returncode}")
        return False


def display_model_registry(registry_path):
    """
    Display the contents of the model registry
    """
    if not os.path.exists(registry_path):
        print("Model registry not found.")
        return

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)

        if not registry:
            print("Model registry is empty.")
            return

        print(f"\n{'='*80}")
        print(f"MODEL REGISTRY ({len(registry)} models)")
        print(f"{'='*80}")

        # Sort models by creation date (newest first)
        registry.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        for i, model in enumerate(registry):
            print(
                f"{i+1}. {model.get('id', 'Unknown')} ({model.get('model_type', 'Unknown')})"
            )
            print(f"   Created: {model.get('created_at', 'Unknown')}")
            print(f"   Path: {model.get('path', 'Unknown')}")
            metrics = model.get("metrics", {})
            print(
                f"   Metrics: Accuracy={metrics.get('val_accuracy', 0):.4f}, F1={metrics.get('val_f1', 0):.4f}"
            )
            print()

    except Exception as e:
        print(f"Error reading model registry: {e}")


def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    # Ensure model registry exists at the root of the output directory
    registry_path = os.path.join(args.output_dir, "model_registry.json")
    if not os.path.exists(registry_path):
        print(f"Creating new model registry at {registry_path}")
        with open(registry_path, "w") as f:
            json.dump([], f)

    # Load model configurations
    model_configs = load_model_configs(args.config_file)

    # Save run configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = {"timestamp": timestamp, "args": vars(args), "models": model_configs}

    with open(os.path.join(args.report_dir, f"run_config_{timestamp}.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Train models
    trained_model_names = []
    successful_models = []

    if not args.skip_training:
        for config in model_configs:
            success, model_path, versioned_name = train_model(config, args)
            if success:
                successful_models.append((versioned_name, model_path))
                trained_model_names.append(versioned_name)

        print(f"\nSuccessfully trained {len(successful_models)} models")

    # Collect paths of successfully trained models
    model_paths = []
    if successful_models:
        for model_name, model_path in successful_models:
            if model_path:
                model_paths.append(model_path)
                print(f"Adding model for evaluation: {model_name} at {model_path}")

    # Find the test data
    test_data_dir = os.path.join(args.data_dir, "processed_data", "test_set.csv")
    if not os.path.exists(test_data_dir):
        # Try processed directory
        test_data_dir = os.path.join(args.data_dir, "processed", "test_set.csv")
        if not os.path.exists(test_data_dir):
            # Try raw data directory directly
            test_data_dir = args.data_dir

    # Create a temporary file with model paths
    if model_paths:
        models_list_file = os.path.join(args.report_dir, "models_list.txt")
        with open(models_list_file, "w") as f:
            for path in model_paths:
                f.write(f"{path}\n")
        print(
            f"Saved list of {len(model_paths)} models to evaluate at: {models_list_file}"
        )

    # Evaluate all models
    eval_success = evaluate_all_models(args.output_dir, test_data_dir, args)

    # Run model comparison
    if eval_success:
        # If models were trained in this run, only compare those models
        # Otherwise, compare all models in the evaluations directory
        comp_success = run_model_comparison(
            args, trained_model_names if trained_model_names else None
        )
        if comp_success:
            print(f"Model comparison completed successfully")

    print(f"\nTraining and evaluation pipeline complete!")
    print(
        f"Check {os.path.join(args.report_dir, 'comparisons')} for comparison reports"
    )

    # Display the model registry
    registry_path = os.path.join(args.output_dir, "model_registry.json")
    display_model_registry(registry_path)


if __name__ == "__main__":
    main()
