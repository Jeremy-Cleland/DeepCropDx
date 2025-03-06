import os
import torch
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import your existing safe_torch_load
from src.utils.model_utils import safe_torch_load


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    model_dir,
    model_name,
    is_best=False,
    class_info=None,
    training_params=None,
    keep_top_k=3,
):
    """
    Standard checkpoint saving function to use across project.

    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        model_dir: Directory to save checkpoint
        model_name: Base name for the model
        is_best: Whether this is the best model so far
        class_info: Optional class information dictionary
        training_params: Optional training parameters
        keep_top_k: Number of best checkpoints to keep

    Returns:
        checkpoint_path: Path to the saved checkpoint
    """
    os.makedirs(model_dir, exist_ok=True)

    # Get model type
    model_type = model.__class__.__name__
    if hasattr(model, "model_type"):
        model_type = model.model_type

    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_type": model_type,
        "created_at": datetime.now().isoformat(),
    }

    # Add class info if provided
    if class_info is not None and "class_to_idx" in class_info:
        checkpoint["class_to_idx"] = class_info["class_to_idx"]

    # Add training params if provided
    if training_params is not None:
        checkpoint["training_params"] = training_params

    # Save regular checkpoint
    checkpoint_path = os.path.join(model_dir, f"{model_name}_checkpoint_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Save best model if indicated
    if is_best:
        best_path = os.path.join(model_dir, f"{model_name}_best.pth")
        torch.save(checkpoint, best_path)

        # Also save a versioned best model that won't be overwritten
        best_versioned_path = os.path.join(
            model_dir, f"{model_name}_best_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, best_versioned_path)

        # For compatibility with older code
        compat_path = os.path.join(model_dir, "best_model.pth")
        torch.save(checkpoint, compat_path)

    # Clean up old checkpoints if needed
    if keep_top_k > 0:
        cleanup_old_checkpoints(model_dir, model_name, keep_top_k)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path, model=None, optimizer=None, device="cpu", strict=True
):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load the model to
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        model: Loaded model (or None if not provided)
        optimizer: Loaded optimizer (or None if not provided)
        checkpoint: Dictionary containing checkpoint data
    """
    # Use the existing safe_torch_load from model_utils.py
    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        fallback_to_unsafe=True,
    )

    # Load model if provided
    if model is not None and "model_state_dict" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except Exception as e:
            logger.warning(f"Error loading model state dict: {str(e)}")
            logger.warning("Attempting to adjust state dict keys...")

            adjusted_state_dict = {}
            for k, v in checkpoint["model_state_dict"].items():
                if k.startswith("module."):
                    adjusted_state_dict[k[7:]] = v
                else:
                    adjusted_state_dict[k] = v

            model.load_state_dict(adjusted_state_dict, strict=strict)
            logger.info("Model weights loaded successfully after adjustment")

    # Load optimizer if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {str(e)}")

    return model, optimizer, checkpoint


def cleanup_old_checkpoints(checkpoint_dir, model_name, keep_top_k=3):
    """
    Remove old checkpoints, keeping only the latest few.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Base name of the model
        keep_top_k: Number of checkpoints to keep

    Returns:
        removed_count: Number of checkpoints removed
    """
    # Find regular checkpoints (not best or final)
    checkpoint_pattern = f"{model_name}_checkpoint_*.pth"
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith(f"{model_name}_checkpoint_") and f.endswith(".pth")
    ]

    # Skip if no need to clean up
    if len(checkpoints) <= keep_top_k:
        return 0

    # Extract checkpoint epochs
    checkpoint_info = []
    for checkpoint in checkpoints:
        try:
            # Extract epoch number from filename
            epoch = int(
                checkpoint.replace(f"{model_name}_checkpoint_", "").replace(".pth", "")
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            mod_time = os.path.getmtime(checkpoint_path)
            checkpoint_info.append((checkpoint, epoch, mod_time, checkpoint_path))
        except ValueError:
            continue

    # Sort by modification time, most recent first
    checkpoint_info.sort(key=lambda x: x[2], reverse=True)

    # Keep the top K checkpoints, remove the rest
    removed_count = 0
    for checkpoint, epoch, mod_time, checkpoint_path in checkpoint_info[keep_top_k:]:
        try:
            os.remove(checkpoint_path)
            removed_count += 1
            logger.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            logger.warning(f"Error removing checkpoint {checkpoint}: {str(e)}")

    return removed_count


def export_model_info(checkpoint_path, output_dir=None, format="json"):
    """
    Export model information from checkpoint to a readable format.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save output (defaults to checkpoint directory)
        format: Output format ('json' or 'text')

    Returns:
        output_path: Path to the output file
    """
    # Load checkpoint using safe_torch_load
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)

    # Define output directory
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)

    # Extract info
    model_info = {
        "model_type": checkpoint.get("model_type", "Unknown"),
        "created_at": checkpoint.get("created_at", "Unknown"),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "training_params": checkpoint.get("training_params", {}),
        "num_classes": len(checkpoint.get("class_to_idx", {})),
        "checkpoint_path": checkpoint_path,
    }

    # Generate output filename
    base_name = os.path.basename(checkpoint_path).replace(".pth", "")

    if format == "json":
        output_path = os.path.join(output_dir, f"{base_name}_info.json")
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)
    else:
        output_path = os.path.join(output_dir, f"{base_name}_info.txt")
        with open(output_path, "w") as f:
            f.write(f"Model Information for {os.path.basename(checkpoint_path)}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Model Type: {model_info['model_type']}\n")
            f.write(f"Created: {model_info['created_at']}\n")
            f.write(f"Epoch: {model_info['epoch']}\n")
            f.write(f"Number of Classes: {model_info['num_classes']}\n\n")

            f.write("Metrics:\n")
            for metric, value in model_info.get("metrics", {}).items():
                f.write(f"  {metric}: {value}\n")

            f.write("\nTraining Parameters:\n")
            for param, value in model_info.get("training_params", {}).items():
                f.write(f"  {param}: {value}\n")

    return output_path
