import os
import time
import argparse
import glob
import json
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
import matplotlib.pyplot as plt

from src.data.dataset import process_data
from src.models.model_factory import create_model
from src.utils.metrics import calculate_metrics
from src.utils.logger import TrainingLogger
from src.utils.visualization import GradCAM, get_target_layer_for_model


def update_model_registry(registry_path, model_info):
    """
    Update the model registry with information about a newly trained model

    Args:
        registry_path (str): Path to the registry JSON file
        model_info (dict): Information about the model to register
    """
    registry = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted registry file
            pass

    registry.append(model_info)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return True


def cleanup_checkpoints(checkpoint_dir, keep_top_k=3, pattern="checkpoint_epoch_*.pth"):
    """
    Keep only the top K checkpoints based on modification time

    Args:
        checkpoint_dir (str): Directory containing checkpoints
        keep_top_k (int): Number of checkpoints to keep
        pattern (str): Glob pattern to match checkpoint files
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(checkpoints) <= keep_top_k:
        return

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Remove older checkpoints beyond the keep limit
    for checkpoint in checkpoints[keep_top_k:]:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {checkpoint}")


def get_next_version(model_dir, model_name):
    """
    Get the next version number for a model

    Args:
        model_dir (str): Directory containing model versions
        model_name (str): Base name of the model

    Returns:
        int: Next version number
    """
    # Look for model files with pattern: model_name_v*_best.pth or model_name_v*_final.pth
    pattern1 = os.path.join(model_dir, f"{model_name}_v*_best.pth")
    pattern2 = os.path.join(model_dir, f"{model_name}_v*_final.pth")
    existing_models = glob.glob(pattern1) + glob.glob(pattern2)

    print(
        f"Looking for existing versions in {model_dir} with patterns: {pattern1}, {pattern2}"
    )

    if not existing_models:
        print(f"No existing versions found for {model_name}, starting with version 1")
        return 1

    versions = []
    for model_path in existing_models:
        try:
            # Extract version number from filename
            filename = os.path.basename(model_path)
            # Split by '_v' to get the part after the version marker
            parts = filename.split("_v")
            if len(parts) < 2:
                continue

            # Get the version number (part before the next underscore)
            version_part = parts[1].split("_")[0]
            version = int(version_part)
            versions.append(version)
            print(f"Found existing version: {version} in {filename}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing version from {model_path}: {str(e)}")
            continue

    next_version = max(versions) + 1 if versions else 1
    print(f"Next version for {model_name}: {next_version}")
    return next_version


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_dir,
    logger=None,
    patience=10,
    use_amp=False,
    keep_top_k=3,
    model_name=None,
    class_info=None,
    training_params=None,
    optuna_trial=None,
):
    """
    Train the model with early stopping and optional mixed precision
    """
    since = time.time()

    # Initialize scaler for AMP
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    # Create model directory structure
    if model_name:
        # Create base model directory if needed
        model_base_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_base_dir):
            os.makedirs(model_base_dir, exist_ok=True)

        # Get the next version
        version = get_next_version(save_dir, model_name)
        versioned_model_name = f"{model_name}_v{version}"

        # Create full versioned directory
        versioned_dir = os.path.join(save_dir, versioned_model_name)
        os.makedirs(versioned_dir, exist_ok=True)

        # Create subdirectories
        models_dir = os.path.join(versioned_dir, "models")
        logs_dir = os.path.join(versioned_dir, "logs")
        viz_dir = os.path.join(versioned_dir, "visualizations")

        for dir_path in [models_dir, logs_dir, viz_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Update model save directory to the versioned models folder
        model_save_dir = models_dir

        # If logger is provided, create a new one with the correct path
        if logger:
            # First, remove all handlers from the original logger to avoid duplicate logging
            if hasattr(logger, "logger"):
                original_logger = logger.logger
                for handler in list(original_logger.handlers):
                    original_logger.removeHandler(handler)

            # Create new logger with the versioned logs directory
            logger = TrainingLogger(logs_dir, model_name)
    else:
        # If no model_name, just use save_dir directly
        model_save_dir = save_dir
        versioned_model_name = None
        version = 1

    # Initialize history dictionary
    if logger is None:
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
    else:
        history = logger.history
        logger.start_training(
            num_epochs,
            {"train": len(dataloaders["train"]), "val": len(dataloaders["val"])},
        )

    best_model_wts = None
    best_f1 = 0.0
    counter = 0  # Counter for early stopping
    early_stop = False  # Flag for early stopping
    best_metrics = None  # Store best metrics
    best_model_path = None  # Path to best model

    # Ensure all necessary directories exist
    os.makedirs(model_save_dir, exist_ok=True)

    if logger:
        logger.logger.info(f"Training model: {versioned_model_name or model_name}")

    for epoch in range(num_epochs):
        if logger:
            logger.start_epoch(epoch + 1)
        else:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            all_preds = []
            all_labels = []

            # Iterate over data
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                if logger:
                    logger.batch_start_time = time.time()

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    # Use autocast for mixed precision where appropriate
                    with (
                        autocast()
                        if use_amp and phase == "train" and device.type == "cuda"
                        else nullcontext()
                    ):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        if use_amp and device.type == "cuda":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Log batch progress
                if logger:
                    logger.log_batch(
                        phase=phase,
                        batch_idx=batch_idx,
                        loss=batch_loss / inputs.size(0),
                        progress=100.0 * (batch_idx + 1) / len(dataloaders[phase]),
                        log_every=max(
                            1, len(dataloaders[phase]) // 10
                        ),  # Log about 10 times per epoch
                    )

            if phase == "train" and scheduler is not None:
                scheduler.step()

            # Calculate metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            metrics = calculate_metrics(all_labels, all_preds)
            metrics["loss"] = epoch_loss

            # Update history
            if phase == "train":
                train_metrics = metrics
            else:
                val_metrics = metrics

            if not logger:
                print(
                    f'{phase} Loss: {epoch_loss:.4f} Acc: {metrics["accuracy"]:.4f} F1: {metrics["f1"]:.4f}'
                )

            # Early stopping check - only on validation phase
            if phase == "val":
                if metrics["f1"] > best_f1:
                    counter = 0  # Reset counter
                    best_f1 = metrics["f1"]
                    best_model_wts = model.state_dict().copy()
                    best_metrics = metrics.copy()

                    # Create checkpoint with enhanced metadata
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss,
                        "metrics": metrics,
                        "model_type": training_params.get("model_type"),
                        "class_to_idx": (
                            class_info.get("class_to_idx") if class_info else None
                        ),
                        "version": version,
                        "created_at": datetime.now().isoformat(),
                        "training_params": training_params,
                    }

                    # Save versioned best model
                    best_versioned_path = os.path.join(
                        model_save_dir, f"{versioned_model_name or model_name}_best.pth"
                    )
                    torch.save(checkpoint, best_versioned_path)
                    best_model_path = best_versioned_path

                    # Also save as best_model.pth for backward compatibility
                    best_model_path_compat = os.path.join(save_dir, "best_model.pth")
                    torch.save(checkpoint, best_model_path_compat)

                    if logger:
                        logger.log_checkpoint(
                            epoch + 1, f"{versioned_model_name}_best.pth"
                        )
                else:
                    counter += 1
                    if counter >= patience:
                        if logger:
                            logger.logger.info(
                                f"Early stopping triggered after {epoch+1} epochs"
                            )
                        else:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                        early_stop = True

                # Add Optuna pruning support here
                if optuna_trial is not None:
                    # Report validation metrics for pruning
                    optuna_trial.report(metrics["f1"], epoch)

                    # Handle pruning based on reported value
                    if optuna_trial.should_prune():
                        message = (
                            f"Trial {optuna_trial.number} pruned at epoch {epoch+1}"
                        )
                        if logger:
                            logger.logger.info(message)
                        else:
                            print(message)

                        # If we have a best model, load it before pruning
                        if best_model_wts is not None:
                            model.load_state_dict(best_model_wts)

                        # Return early with current best results instead of raising exception
                        return model, history, best_model_path, version
                        # If we have a best model, load it before pruning

        # Log epoch results
        if logger:
            logger.end_epoch(train_metrics, val_metrics)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                model_save_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "metrics": metrics,
                    "model_type": training_params.get("model_type"),
                    "class_to_idx": (
                        class_info.get("class_to_idx") if class_info else None
                    ),
                    "version": version,
                    "created_at": datetime.now().isoformat(),
                },
                checkpoint_path,
            )

            if logger:
                logger.log_checkpoint(epoch + 1, f"checkpoint_epoch_{epoch+1}.pth")

        # Break out of epoch loop if early stopping triggered
        if early_stop:
            break

    time_elapsed = time.time() - since
    if logger:
        logger.end_training()
    else:
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val F1: {best_f1:4f}")

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Save training history if not using logger
    if not logger:
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    # Cleanup old checkpoints
    if keep_top_k > 0:
        cleanup_checkpoints(model_save_dir, keep_top_k)

    return model, history, best_model_path, version


def find_learning_rate(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    start_lr=1e-7,
    end_lr=10,
    num_steps=100,
):
    """
    Implements a learning rate finder to find optimal learning rate

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_steps: Number of steps for the search

    Returns:
        lrs: List of learning rates
        losses: List of corresponding losses
    """
    # Save current model state
    model_state = model.state_dict().copy()
    optim_state = optimizer.state_dict().copy()

    # Setup
    lrs = []
    losses = []
    best_loss = float("inf")

    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    # Exponentially increasing factor
    mult_factor = (end_lr / start_lr) ** (1 / num_steps)

    # Training loop
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_steps:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)

        # Stop if loss explodes
        if loss_value < best_loss:
            best_loss = loss_value
        if loss_value > 4 * best_loss or torch.isnan(loss):
            break

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= mult_factor

    # Restore model state
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)

    return lrs, losses


def benchmark_device(device):
    """Simple benchmark to verify MPS/CUDA acceleration"""
    print(f"Running benchmark on {device}...")
    x = torch.randn(1000, 1000).to(device)
    start = time.time()
    for _ in range(100):
        torch.matmul(x, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    end = time.time()
    print(f"Benchmark time: {end - start:.4f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train deep learning models for crop disease detection"
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

    # Training parameters
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of data loading workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--use_weights", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone layers"
    )

    # Hardware options
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS")
    parser.add_argument("--use_mps", action="store_true", help="Enable MPS explicitly")

    # Optimization options
    parser.add_argument(
        "--use_amp", action="store_true", help="Use Automatic Mixed Precision"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient operations",
    )
    parser.add_argument(
        "--cache_dataset", action="store_true", help="Cache processed dataset in memory"
    )
    parser.add_argument(
        "--mps_graph", action="store_true", help="Enable MPS graph mode"
    )
    parser.add_argument(
        "--mps_fallback",
        action="store_true",
        help="Use CPU fallback for unsupported MPS ops",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for faster data transfer to GPU",
    )
    parser.add_argument(
        "--optimize_for_m_series",
        action="store_true",
        help="Apply specific optimizations for M-series Apple Silicon chips (M1/M2/M3/M4)",
    )

    # Early stopping and checkpoints
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--keep_top_k", type=int, default=3, help="Keep top K checkpoints"
    )

    # Versioning
    parser.add_argument(
        "--version", type=int, default=None, help="Explicitly set model version"
    )

    # Other options
    parser.add_argument(
        "--find_lr", action="store_true", help="Run learning rate finder"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--resnet_version",
        type=int,
        default=50,
        help="ResNet version (18, 34, 50, 101, 152) - only used for 'resnet_attention' model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    from src.utils.device_utils import get_device

    # Check for user preference to disable CUDA or MPS
    device = get_device()
    if args.no_cuda and device.type == "cuda":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() and not args.no_mps else "cpu"
        )
    elif args.no_mps and device.type == "mps":
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Set environment variables for better performance on Apple Silicon
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Use float32 for better compatibility with MPS
        torch.set_default_dtype(torch.float32)

        # Apply M-series specific optimizations if requested
        if args.optimize_for_m_series:
            print("Applying advanced optimizations for M-series chips")
            # For latest M-series chips (M2/M3/M4), optimize memory usage
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
                "0.0"  # Maximize GPU memory usage
            )

            # Thread optimizations for M4 Max which has more cores
            num_cores = min(16, os.cpu_count() or 8)  # M4 Max has up to 16 cores
            os.environ["OMP_NUM_THREADS"] = str(min(8, num_cores))
            os.environ["MKL_NUM_THREADS"] = str(min(8, num_cores))

            # Enable graph mode if supported and not already enabled
            if hasattr(torch.backends.mps, "enable_graph_mode"):
                torch.backends.mps.enable_graph_mode = True
                print("MPS graph mode enabled for M-series optimization")

            # Check PyTorch version for additional optimizations
            if hasattr(torch, "__version__") and torch.__version__ >= "2.2.0":
                print("Enabling PyTorch 2.2+ specific MPS optimizations")
                if hasattr(torch.backends.mps, "set_benchmark_mode"):
                    torch.backends.mps.set_benchmark_mode(True)
                    print("MPS benchmark mode enabled")

    # Run benchmark
    benchmark_device(device)

    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create output directories with improved structure
    root_dir = args.output_dir

    # Check if the output directory already has a version suffix (_v1, _v2, etc.)
    # If it does, use it directly to avoid nesting
    if os.path.basename(root_dir).startswith(f"{args.experiment_name}_v"):
        # We're already in a versioned directory, don't add another level
        experiment_dir = root_dir
        print(f"Detected versioned directory structure. Using: {experiment_dir}")
    else:
        # Standard case - create experiment directory inside root
        experiment_dir = os.path.join(root_dir, args.experiment_name)

    # Create subdirectories
    processed_data_dir = os.path.join(experiment_dir, "processed_data")
    model_save_dir = os.path.join(experiment_dir, "models")
    logs_dir = os.path.join(experiment_dir, "logs")
    viz_dir = os.path.join(experiment_dir, "visualizations")

    # Print directory structure for debugging
    print(f"Output directory structure:")
    print(f"  Root dir: {root_dir}")
    print(f"  Experiment dir: {experiment_dir}")
    print(f"  Model save dir: {model_save_dir}")
    print(f"  Visualizations dir: {viz_dir}")

    # Create all directories
    for directory in [processed_data_dir, model_save_dir, logs_dir, viz_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(logs_dir, args.experiment_name)
    logger.logger.info(f"Starting experiment: {args.experiment_name}")
    logger.logger.info(f"Command line arguments: {args}")

    # Process data
    logger.logger.info("Processing dataset...")
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

    # Create model
    num_classes = len(class_info["class_to_idx"])
    logger.logger.info(f"Creating model: {args.model} with {num_classes} classes")

    # Create model using the factory pattern
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        use_weights=args.use_weights,
        freeze_backbone=args.freeze_backbone,
        resnet_version=args.resnet_version,
    )

    model = model.to(device)

    # Log model architecture
    logger.logger.info(f"Model architecture:\n{model}")

    # Create criterion and optimizer
    class_weights = None
    if num_classes > 1:
        # Calculate class weights to handle imbalance
        class_counts = np.array(list(class_info["class_counts"].values()))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / np.sum(class_weights) * num_classes
        class_weights = torch.FloatTensor(class_weights).to(device)
        logger.logger.info(f"Using class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only train the classifier parameters if freezing the backbone
    if args.freeze_backbone:
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        logger.logger.info(
            f"Training only {len(params_to_update)} parameters (classifier)"
        )
    else:
        params_to_update = model.parameters()
        logger.logger.info("Training all parameters (full model)")

    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)

    # Run learning rate finder if requested
    if args.find_lr:
        logger.logger.info("Running learning rate finder...")
        lrs, losses = find_learning_rate(
            model, dataloaders["train"], criterion, optimizer, device
        )

        # Plot learning rate vs. loss
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.savefig(os.path.join(logs_dir, f"{args.experiment_name}_lr_finder.png"))
        plt.close()

        # Find the optimal learning rate (typically where the slope is steepest)
        min_grad_idx = None
        try:
            loss_diff = np.gradient(np.array(losses))
            min_grad_idx = np.argmin(loss_diff)
            suggested_lr = lrs[min_grad_idx] / 10  # Divide by 10 as a rule of thumb
            logger.logger.info(f"Suggested learning rate: {suggested_lr:.6f}")

            # Update the learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = suggested_lr
        except Exception as e:
            logger.logger.warning(
                f"Could not determine optimal learning rate: {str(e)}"
            )

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Save class mapping
    with open(os.path.join(model_save_dir, "class_mapping.txt"), "w") as f:
        for class_name, idx in class_info["class_to_idx"].items():
            f.write(f"{class_name},{idx}\n")

    # Prepare training parameters for model metadata
    training_params = {
        "model_type": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
        "use_weights": args.use_weights,
        "freeze_backbone": args.freeze_backbone,
        "device": str(device),
        "use_amp": args.use_amp,
    }

    # Train model
    logger.logger.info(f"Starting training for {args.epochs} epochs")
    logger.logger.info(f"Using Automatic Mixed Precision: {args.use_amp}")
    logger.logger.info(f"Early stopping patience: {args.patience}")

    model, history, best_model_path, version = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
        model_save_dir,
        logger=logger,
        patience=args.patience,
        use_amp=args.use_amp,
        keep_top_k=args.keep_top_k,
        model_name=args.experiment_name,
        class_info=class_info,
        training_params=training_params,
    )

    versioned_model_name = f"{args.experiment_name}_v{version}"
    final_model_path = os.path.join(model_save_dir, f"{versioned_model_name}_final.pth")

    # Create rich checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_info["class_to_idx"],
        "model_type": args.model,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "training_params": training_params,
    }

    # Save versioned final model
    torch.save(checkpoint, final_model_path)

    # Also save as final_model.pth for backward compatibility
    torch.save(checkpoint, os.path.join(model_save_dir, "final_model.pth"))

    logger.logger.info(f"Final model saved to {final_model_path}")

    # Update model registry
    # Make sure we're using the correct root directory for the model registry
    # If we're in a versioned directory, the registry should be in its parent
    if os.path.basename(root_dir).startswith(f"{args.experiment_name}_v"):
        registry_dir = os.path.dirname(root_dir)
    else:
        registry_dir = root_dir

    registry_path = os.path.join(registry_dir, "model_registry.json")

    # Load best metrics, if available
    from src.utils.model_utils import safe_torch_load

    best_checkpoint = safe_torch_load(
        best_model_path, map_location="cpu", weights_only=True, fallback_to_unsafe=True
    )

    best_metrics = best_checkpoint.get("metrics", {})

    model_info = {
        "id": versioned_model_name,
        "experiment": args.experiment_name,
        "model_type": args.model,
        "path": os.path.relpath(final_model_path, root_dir),
        "best_model_path": os.path.relpath(best_model_path, root_dir),
        "num_classes": num_classes,
        "created_at": datetime.now().isoformat(),
        "version": version,
        "metrics": {
            "val_accuracy": best_metrics.get("accuracy", 0),
            "val_precision": best_metrics.get("precision", 0),
            "val_recall": best_metrics.get("recall", 0),
            "val_f1": best_metrics.get("f1", 0),
        },
        "training_params": training_params,
    }

    update_model_registry(registry_path, model_info)
    logger.logger.info(f"Model registered in {registry_path}")

    # Get class names from class_to_idx mapping
    class_names = [None] * len(class_info["class_to_idx"])
    for name, idx in class_info["class_to_idx"].items():
        class_names[idx] = name

    logger.logger.info("Generating visualizations...")

    # Generate all standard visualizations including GradCAM
    logger.logger.info("Generating standard visualizations and GradCAM")
    from src.utils.visualization import save_all_visualizations

    save_all_visualizations(
        model, dataloaders, class_names, device, viz_dir, model_type=args.model
    )

    # t-SNE visualization of feature space
    try:
        from src.utils.visualization import visualize_tsne

        # Function to extract features
        def extract_features(model, dataloader, device):
            model.eval()
            features = []
            labels = []

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(device)

                    # For most models, we need to extract features before the final classification layer
                    if hasattr(model, "features") and hasattr(model, "classifier"):
                        # For EfficientNet, MobileNet
                        x = model.features(inputs)
                        if len(x.shape) > 2:  # If not flattened yet
                            x = (
                                model.avgpool(x)
                                if hasattr(model, "avgpool")
                                else nn.functional.adaptive_avg_pool2d(x, 1)
                            )
                            x = torch.flatten(x, 1)
                    elif hasattr(model, "layer4") and hasattr(model, "fc"):
                        # For ResNet
                        x = model.layer4(
                            model.layer3(
                                model.layer2(
                                    model.layer1(
                                        model.maxpool(
                                            model.relu(model.bn1(model.conv1(inputs)))
                                        )
                                    )
                                )
                            )
                        )
                        x = model.avgpool(x)
                        x = torch.flatten(x, 1)
                    else:
                        # Fallback
                        logger.logger.warning(
                            "Unknown model architecture for feature extraction. Using basic approach."
                        )
                        # Remove the last layer
                        modules = list(model.children())[:-1]
                        feature_extractor = nn.Sequential(*modules)
                        x = feature_extractor(inputs)
                        if len(x.shape) > 2:  # If not flattened yet
                            x = nn.functional.adaptive_avg_pool2d(x, 1)
                            x = torch.flatten(x, 1)

                    features.append(x.cpu().numpy())
                    labels.append(targets.cpu().numpy())

            return np.vstack(features), np.concatenate(labels)

        # Extract features from a subset of the test set (t-SNE can be slow for large datasets)
        subset_size = min(500, len(dataloaders["test"].dataset))
        subset_indices = np.random.choice(
            len(dataloaders["test"].dataset), subset_size, replace=False
        )
        subset_dataset = torch.utils.data.Subset(
            dataloaders["test"].dataset, subset_indices
        )
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset, batch_size=32, shuffle=False
        )

        features, labels = extract_features(model, subset_loader, device)

        # Visualize t-SNE
        fig = visualize_tsne(features, labels, class_names)
        fig.savefig(
            os.path.join(viz_dir, "tsne_visualization.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.logger.info(f"t-SNE visualization saved to {viz_dir}")
    except Exception as e:
        logger.logger.error(f"Error generating t-SNE visualization: {str(e)}")

    logger.logger.info("Training and visualization finished!")


if __name__ == "__main__":
    main()
