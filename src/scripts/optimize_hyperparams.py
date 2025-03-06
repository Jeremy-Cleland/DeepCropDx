#!/usr/bin/env python3
# src/scripts/optimize_hyperparams.py

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.device_utils import get_device
from src.data.dataset import process_data
from src.utils.logger import TrainingLogger
from src.utils.optuna_utils import run_optuna_study


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization using Optuna"
    )

    # Required arguments
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save models and results"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "efficientnet",
            "efficientnet_b3",
            "mobilenet",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "mobilenet_attention",
            "resnet18",
            "resnet50",
            "resnet_attention",
        ],
        help="Model architecture to use",
    )

    # Optuna parameters
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout for optimization in seconds"
    )
    parser.add_argument(
        "--optimize_augmentation",
        action="store_true",
        help="Optimize data augmentation parameters",
    )
    parser.add_argument(
        "--optimize_architecture",
        action="store_true",
        help="Optimize model architecture parameters",
    )
    parser.add_argument(
        "--optimize_training",
        action="store_true",
        help="Optimize training dynamics parameters",
    )
    parser.add_argument(
        "--pruning_percentile",
        type=int,
        default=75,
        help="Percentile for median pruner (higher = more aggressive pruning)",
    )

    # Training parameters
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Base batch size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of data loading workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--use_weights", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone layers"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--keep_top_k", type=int, default=1, help="Keep top K checkpoints"
    )

    # Hardware options
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS")
    parser.add_argument("--use_mps", action="store_true", help="Enable MPS explicitly")
    parser.add_argument(
        "--use_amp", action="store_true", help="Use Automatic Mixed Precision"
    )

    # Other options
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--resnet_version",
        type=int,
        default=50,
        help="ResNet version (18, 34, 50, 101, 152) - only used for 'resnet_attention' model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = get_device(no_cuda=args.no_cuda, no_mps=args.no_mps)
    print(f"Using device: {device}")

    from src.utils.model_tester import test_model_forward_backward

    print(f"\nVerifying {args.model} implementation before optimization...")
    test_success = test_model_forward_backward(
        model_type=args.model,
        num_classes=39,  # Use a reasonable default
        batch_size=args.batch_size,
    )

    if not test_success:
        print(f"\nWARNING: Model {args.model} verification failed!")
        user_response = input("Continue with optimization anyway? (y/n): ")
        if user_response.lower() != "y":
            print("Exiting due to model verification failure")
            return 1

    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.model}_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(logs_dir, args.experiment_name)
    logger.logger.info(
        f"Starting Optuna hyperparameter optimization for {args.experiment_name}"
    )
    logger.logger.info(f"Command line arguments: {args}")

    # Process data
    logger.logger.info("Processing dataset...")

    # Create processed_data directory
    processed_data_dir = os.path.join(args.output_dir, "processed_data")
    os.makedirs(processed_data_dir, exist_ok=True)

    dataloaders, class_info = process_data(
        args.data_dir,
        processed_data_dir,
        args.img_size,
        args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Print class distribution
    logger.logger.info("Class distribution:")
    for class_name, count in class_info["class_counts"].items():
        logger.logger.info(f"  {class_name}: {count} images")

    # Run Optuna study
    study, best_params = run_optuna_study(
        args,
        dataloaders,
        class_info,
        device,
        logger,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Log results
    logger.logger.info("Hyperparameter optimization completed!")
    logger.logger.info(f"Best hyperparameters: {best_params}")

    # Print command to train with best parameters
    cmd = f"\nTo train with the best parameters, run:\n"
    cmd += f"python -m src.training.train \\\n"
    cmd += f"  --data_dir {args.data_dir} \\\n"
    cmd += f"  --output_dir {args.output_dir}/best_model \\\n"
    cmd += f"  --model {args.model} \\\n"

    # Add best hyperparameters to command
    param_mapping = {
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "dropout_rate": "dropout_rate",
    }

    for optuna_param, cmd_param in param_mapping.items():
        if optuna_param in best_params:
            cmd += f"  --{cmd_param} {best_params[optuna_param]} \\\n"

    # Add other relevant parameters
    if "freeze_backbone" in best_params and best_params["freeze_backbone"]:
        cmd += f"  --freeze_backbone \\\n"
    if args.use_weights:
        cmd += f"  --use_weights \\\n"
    if args.use_amp:
        cmd += f"  --use_amp \\\n"

    # Add a few more standard parameters
    cmd += f"  --img_size {args.img_size} \\\n"
    cmd += f"  --epochs 30 \\\n"  # Recommend slightly longer training for final model
    cmd += f"  --patience {args.patience}"

    logger.logger.info(cmd)
    print(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
