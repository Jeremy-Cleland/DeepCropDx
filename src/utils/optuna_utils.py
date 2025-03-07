# src/utils/optuna_utils.py

import os
import torch
import optuna
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import json


from src.data.dataset import (
    process_data,
    get_data_transforms,
    recreate_dataloaders,
)
from src.models.model_factory import create_model
from src.training.train import train_model


def objective(trial, args, dataloaders, class_info, device, logger):
    """Objective function for Optuna to optimize."""

    # Basic hyperparameters (already included)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Model architecture choices based on model type
    model_specific_params = {}

    if args.model in ["efficientnet", "efficientnet_b3"]:
        # EfficientNet specific hyperparameters
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        add_hidden_layer = trial.suggest_categorical("add_hidden_layer", [True, False])
        hidden_layer_size = trial.suggest_categorical(
            "hidden_layer_size", [256, 512, 1024]
        )
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "silu"]
        )

        model_specific_params.update(
            {
                "freeze_backbone": freeze_backbone,
                "dropout_rate": dropout_rate,
                "add_hidden_layer": add_hidden_layer,
                "hidden_layer_size": hidden_layer_size,
                "activation": activation,
            }
        )

    elif args.model in ["resnet18", "resnet50", "resnet_attention"]:
        # ResNet specific hyperparameters
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        add_hidden_layer = trial.suggest_categorical("add_hidden_layer", [True, False])
        hidden_layer_size = trial.suggest_categorical(
            "hidden_layer_size", [256, 512, 1024]
        )
        attention_reduction = (
            trial.suggest_int("attention_reduction", 8, 32)
            if "attention" in args.model
            else None
        )

        model_specific_params.update(
            {
                "freeze_backbone": freeze_backbone,
                "dropout_rate": dropout_rate,
                "add_hidden_layer": add_hidden_layer,
                "hidden_layer_size": hidden_layer_size,
                "attention_reduction": attention_reduction,
            }
        )

    elif "mobilenet" in args.model:
        # MobileNet specific hyperparameters
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        width_multiplier = trial.suggest_categorical(
            "width_multiplier", [0.75, 1.0, 1.4]
        )
        add_hidden_layer = trial.suggest_categorical("add_hidden_layer", [True, False])
        hidden_layer_size = trial.suggest_categorical(
            "hidden_layer_size", [256, 512, 1024]
        )

        model_specific_params.update(
            {
                "freeze_backbone": freeze_backbone,
                "dropout_rate": dropout_rate,
                "width_multiplier": width_multiplier,
                "add_hidden_layer": add_hidden_layer,
                "hidden_layer_size": hidden_layer_size,
            }
        )
    else:
        # Default parameters for other model types
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

        model_specific_params.update(
            {
                "freeze_backbone": freeze_backbone,
                "dropout_rate": dropout_rate,
            }
        )

    # Data augmentation hyperparameters
    augmentation_params = {
        "rotation_degrees": trial.suggest_int("rotation_degrees", 5, 45),
        "brightness_factor": trial.suggest_float("brightness_factor", 0.05, 0.3),
        "contrast_factor": trial.suggest_float("contrast_factor", 0.05, 0.3),
        "saturation_factor": trial.suggest_float("saturation_factor", 0.05, 0.3),
        "random_erase_prob": trial.suggest_float("random_erase_prob", 0.0, 0.3),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.5),
    }

    # Optimizer hyperparameters
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "SGD", "AdamW", "RMSprop"]
    )
    optimizer_params = {}

    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer_params.update({"momentum": momentum, "nesterov": nesterov})
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        alpha = trial.suggest_float("alpha", 0.9, 0.99)
        optimizer_params.update({"momentum": momentum, "alpha": alpha})

    # Learning rate scheduler parameters
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_params = {}
    scheduler_type = None

    if use_scheduler:
        scheduler_type = trial.suggest_categorical(
            "scheduler_type", ["step", "cosine", "plateau", "onecycle"]
        )
        if scheduler_type == "step":
            step_size = trial.suggest_int("step_size", 2, 10)
            gamma = trial.suggest_float("gamma", 0.1, 0.7)
            scheduler_params.update(
                {
                    "step_size": step_size,
                    "gamma": gamma,
                }
            )
        elif scheduler_type == "plateau":
            factor = trial.suggest_float("factor", 0.1, 0.5)
            patience = trial.suggest_int("scheduler_patience", 2, 8)
            scheduler_params.update(
                {
                    "factor": factor,
                    "patience": patience,
                }
            )
        elif scheduler_type == "onecycle":
            max_lr = trial.suggest_float("max_lr", lr, lr * 10)
            pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
            scheduler_params.update(
                {
                    "max_lr": max_lr,
                    "pct_start": pct_start,
                }
            )

    # Normalization strategy
    normalization = trial.suggest_categorical(
        "normalization", ["batch", "instance", "layer", "none"]
    )
    model_specific_params["normalization"] = normalization

    # Loss function and parameters
    loss_params = {}
    if len(class_info["class_to_idx"]) > 2:  # Multiclass case
        loss_function = trial.suggest_categorical(
            "loss_function", ["cross_entropy", "focal"]
        )
        if loss_function == "focal":
            gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
            loss_params["gamma"] = gamma
    else:
        loss_function = "cross_entropy"  # Default for binary classification

    # Create model with the architecture hyperparameters
    num_classes = len(class_info["class_to_idx"])

    # Create model with the specified hyperparameters
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        use_weights=args.use_weights,
        resnet_version=getattr(args, "resnet_version", 50),
        **model_specific_params,
    )
    model = model.to(device)

    # Add this debugging code
    print(f"Model architecture:\n{model}")
    print(f"Classifier structure:\n{model.classifier}")

    try:
        dummy_input = torch.rand(1, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {dummy_output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        # Continue with training anyway

    # Define optimizer based on the suggested hyperparameter
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            params_to_update,
            lr=lr,
            momentum=optimizer_params.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_params.get("nesterov", False),
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params_to_update, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params_to_update,
            lr=lr,
            momentum=optimizer_params.get("momentum", 0),
            alpha=optimizer_params.get("alpha", 0.99),
            weight_decay=weight_decay,
        )

    # Define criterion (loss function)
    class_weights = None
    if num_classes > 1:
        # Calculate class weights to handle imbalance
        class_counts = np.array(list(class_info["class_counts"].values()))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / np.sum(class_weights) * num_classes
        class_weights = torch.FloatTensor(class_weights).to(device)

    if loss_function == "focal":
        try:
            from torch.nn.functional import (
                binary_cross_entropy_with_logits,
                cross_entropy,
            )

            # Simple implementation of Focal Loss
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, weight=None, reduction="mean"):
                    super(FocalLoss, self).__init__()
                    self.gamma = gamma
                    self.weight = weight
                    self.reduction = reduction

                def forward(self, input, target):
                    ce_loss = cross_entropy(
                        input, target, weight=self.weight, reduction="none"
                    )
                    pt = torch.exp(-ce_loss)
                    focal_loss = ((1 - pt) ** self.gamma) * ce_loss

                    if self.reduction == "mean":
                        return focal_loss.mean()
                    elif self.reduction == "sum":
                        return focal_loss.sum()
                    else:
                        return focal_loss

            criterion = FocalLoss(
                gamma=loss_params.get("gamma", 2.0), weight=class_weights
            )
        except Exception as e:
            logger.logger.warning(
                f"Failed to create Focal Loss, fallback to CrossEntropyLoss: {str(e)}"
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define scheduler
    scheduler = None
    if use_scheduler:
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 7),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 5),
                verbose=True,
            )
        elif scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_params.get("max_lr", lr * 10),
                total_steps=args.epochs * len(dataloaders["train"]),
                pct_start=scheduler_params.get("pct_start", 0.3),
            )
    # Create a unique directory for this trial
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    model_save_dir = os.path.join(trial_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # Update logger to log to the trial directory
    trial_log_dir = os.path.join(trial_dir, "logs")
    os.makedirs(trial_log_dir, exist_ok=True)
    trial_logger = logger.__class__(
        trial_log_dir, f"{args.experiment_name}_trial_{trial.number}"
    )

    # Log hyperparameters
    trial_logger.logger.info(f"Trial {trial.number} hyperparameters:")
    trial_logger.logger.info(f"Learning rate: {lr}")
    trial_logger.logger.info(f"Weight decay: {weight_decay}")
    trial_logger.logger.info(f"Batch size: {batch_size}")
    trial_logger.logger.info(f"Optimizer: {optimizer_name}")
    for param, value in model_specific_params.items():
        trial_logger.logger.info(f"{param}: {value}")
    if use_scheduler:
        trial_logger.logger.info(f"Scheduler: {scheduler_type}")
        for param, value in scheduler_params.items():
            trial_logger.logger.info(f"Scheduler {param}: {value}")
    trial_logger.logger.info(f"Loss function: {loss_function}")
    for param, value in loss_params.items():
        trial_logger.logger.info(f"Loss {param}: {value}")
    trial_logger.logger.info(f"Augmentation parameters: {augmentation_params}")

    # Prepare training parameters for model metadata
    training_params = {
        "model_type": args.model,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "img_size": args.img_size,
        "use_weights": args.use_weights,
        "optimizer": optimizer_name,
        "device": str(device),
        "use_amp": getattr(args, "use_amp", False),
        "trial_number": trial.number,
        "model_specific_params": model_specific_params,
        "augmentation_params": augmentation_params,
        "loss_function": loss_function,
        "loss_params": loss_params,
        "scheduler_type": scheduler_type,
        "scheduler_params": scheduler_params,
    }

    if hasattr(args, "optimize_augmentation") and args.optimize_augmentation:
        try:
            # Create transforms with the optimized augmentation parameters
            trial_logger.logger.info("Applying optimized augmentation parameters")
            optimized_transforms = get_data_transforms(
                args.img_size, augmentation_params
            )

            # Recreate dataloaders with the optimized transforms and batch size
            optimized_dataloaders, updated_class_info = recreate_dataloaders(
                args.data_dir,
                optimized_transforms,
                batch_size,
                args.num_workers,
                device,
            )

            # Use the optimized dataloaders for training
            dataloaders = optimized_dataloaders

            # Update class_info if needed
            if updated_class_info:
                # Merge the updated class info with the original, keeping original values for missing keys
                for key, value in updated_class_info.items():
                    if key not in class_info:
                        class_info[key] = value

            trial_logger.logger.info(
                f"Successfully created optimized dataloaders with batch size {batch_size}"
            )
        except Exception as e:
            trial_logger.logger.warning(
                f"Failed to create optimized dataloaders: {str(e)}"
            )
            trial_logger.logger.warning("Continuing with original dataloaders")

            from src.data.dataset_utils import dataset_quality_check

            # Create a quality check output directory for this trial
            quality_dir = os.path.join(trial_dir, "data_quality")
            os.makedirs(quality_dir, exist_ok=True)

            # Run quality check and log any issues
            quality_report = dataset_quality_check(
                dataloaders, class_info, output_dir=quality_dir
            )

            # Log significant issues
            for phase in ["train", "val", "test"]:
                if phase in quality_report:
                    if quality_report[phase]["imbalance_ratio"] > 10:
                        trial_logger.logger.warning(
                            f"SEVERE class imbalance in {phase} set (ratio: {quality_report[phase]['imbalance_ratio']:.2f})"
                        )

                    if quality_report[phase]["missing_classes"]:
                        trial_logger.logger.warning(
                            f"Missing classes in {phase} set: {quality_report[phase]['missing_classes']}"
                        )

            # If severe quality issues are found, consider affecting the trial score
            # For example, penalize trials with severe class imbalance
        except Exception as e:
            trial_logger.logger.warning(f"Data quality check failed: {str(e)}")

    # Train model with early stopping
    try:
        model, history, best_model_path, version = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            args.epochs,
            device,
            model_save_dir,
            logger=trial_logger,
            patience=args.patience if hasattr(args, "patience") else 10,
            use_amp=getattr(args, "use_amp", False),
            keep_top_k=args.keep_top_k if hasattr(args, "keep_top_k") else 1,
            model_name=f"{args.experiment_name}_trial_{trial.number}",
            class_info=class_info,
            training_params=training_params,
            optuna_trial=trial,  # Pass the trial object for pruning
        )

        # Get the best validation F1 score from the history
        best_val_f1 = max(history.get("val_f1", [0]))

        # Log the best validation score achieved
        trial_logger.logger.info(
            f"Trial {trial.number} completed with best val F1: {best_val_f1:.4f}"
        )

        return best_val_f1

    except Exception as e:
        # Log any exception that occurred during training
        trial_logger.logger.error(f"Error in trial {trial.number}: {str(e)}")
        # Reraise the exception to be caught by study.optimize
        raise e


def run_optuna_study(
    args, dataloaders, class_info, device, logger, n_trials=50, timeout=None
):
    """
    Run an Optuna study to find optimal hyperparameters.

    Args:
        args: Command line arguments
        dataloaders: Dictionary of DataLoaders
        class_info: Class information dictionary
        device: Device to run on
        logger: Logger instance
        n_trials: Number of trials to run
        timeout: Timeout in seconds (optional)

    Returns:
        optuna.Study: The completed study object
        dict: Best hyperparameters
    """
    # Create study output directory
    study_dir = os.path.join(args.output_dir, "optuna_study")
    os.makedirs(study_dir, exist_ok=True)

    # Set up study database
    storage_name = os.path.join(study_dir, f"{args.experiment_name}_study.db")
    storage = f"sqlite:///{storage_name}"

    # Define pruner and sampler
    pruner_percentile = getattr(args, "pruning_percentile", 75)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(
        seed=getattr(args, "seed", 42)
    )  # Using TPE (Tree-structured Parzen Estimator)

    # Create or load study
    study_name = f"{args.experiment_name}_optuna_study"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",  # We want to maximize F1 score
        pruner=pruner,
        sampler=sampler,
    )

    # Log study start
    logger.logger.info(f"Starting Optuna study '{study_name}' with {n_trials} trials")
    logger.logger.info(
        f"Using pruner: MedianPruner with percentile={pruner_percentile}%"
    )
    logger.logger.info(f"Storage: {storage}")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, dataloaders, class_info, device, logger),
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),
        show_progress_bar=True,
    )

    # Get best trial information
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Log best results
    logger.logger.info(f"Best trial: {best_trial.number}")
    logger.logger.info(f"Best F1 score: {best_value:.4f}")
    logger.logger.info("Best hyperparameters:")
    for param, value in best_params.items():
        logger.logger.info(f"  {param}: {value}")

    # Create visualization of the optimization history
    try:
        fig_history = plot_optimization_history(study)
        fig_history.savefig(
            os.path.join(study_dir, "optimization_history.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_history)
    except Exception as e:
        logger.logger.warning(f"Failed to create optimization history plot: {str(e)}")

    # Create parameter importance plot
    try:
        fig_importance = plot_param_importances(study)
        fig_importance.savefig(
            os.path.join(study_dir, "param_importances.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_importance)
    except Exception as e:
        logger.logger.warning(f"Failed to create parameter importance plot: {str(e)}")

    # Create parallel coordinate plot for hyperparameters
    try:
        fig_parallel = plot_parallel_coordinate(study)
        fig_parallel.savefig(
            os.path.join(study_dir, "parallel_coordinate.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_parallel)
    except Exception as e:
        logger.logger.warning(f"Failed to create parallel coordinate plot: {str(e)}")

    # Create contour plot for parameter relationships
    try:
        # Find the most important parameters
        importances = optuna.importance.get_param_importances(study)
        top_params = list(importances.keys())[: min(len(importances), 2)]

        if len(top_params) >= 2:
            fig_contour = optuna.visualization.matplotlib.plot_contour(
                study, params=top_params
            )
            fig_contour.savefig(
                os.path.join(study_dir, "contour_plot.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_contour)
    except Exception as e:
        logger.logger.warning(f"Failed to create contour plot: {str(e)}")

    # Create slice plots for top parameters
    try:
        importances = optuna.importance.get_param_importances(study)
        top_params = list(importances.keys())[: min(len(importances), 4)]

        for param in top_params:
            fig_slice = optuna.visualization.matplotlib.plot_slice(
                study, params=[param]
            )
            fig_slice.savefig(
                os.path.join(study_dir, f"slice_plot_{param}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_slice)
    except Exception as e:
        logger.logger.warning(f"Failed to create slice plots: {str(e)}")

    # Save study results to JSON
    import json

    study_results = {
        "best_trial": best_trial.number,
        "best_f1_score": best_value,
        "best_params": best_params,
        "completed_trials": len(study.trials),
        "datetime": datetime.now().isoformat(),
        "study_name": study_name,
        "pruner": f"MedianPruner(percentile={pruner_percentile}%)",
        "n_trials": n_trials,
    }

    with open(os.path.join(study_dir, "study_results.json"), "w") as f:
        json.dump(study_results, f, indent=2)

    # Create a summary file for quick reference
    with open(os.path.join(study_dir, "best_params_summary.txt"), "w") as f:
        f.write(f"Best Trial: {best_trial.number}\n")
        f.write(f"Best F1 Score: {best_value:.4f}\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nCompleted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total trials: {len(study.trials)}\n")

    return study, best_params


def plot_optimization_history(study):
    """Plot the optimization history of the study."""
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value (F1 Score)")
    plt.tight_layout()
    return fig


def plot_param_importances(study):
    """Plot the parameter importances."""
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.tight_layout()
    return fig


def plot_parallel_coordinate(study):
    """Plot the parallel coordinate plot."""
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinate Plot")
    plt.tight_layout()
    return fig


def get_best_hyperparameters(args):
    """
    Load the best hyperparameters from a previous Optuna study.

    Args:
        args: Command line arguments with output_dir and experiment_name

    Returns:
        dict: Best hyperparameters or None if not found
    """
    study_dir = os.path.join(args.output_dir, "optuna_study")
    results_path = os.path.join(study_dir, "study_results.json")

    if not os.path.exists(results_path):
        return None

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        return results.get("best_params")
    except Exception as e:
        print(f"Error loading best hyperparameters: {str(e)}")
        return None


def apply_best_hyperparameters(args, best_params):
    """
    Apply the best hyperparameters from Optuna to the command line arguments.

    Args:
        args: Command line arguments
        best_params: Best hyperparameters from Optuna

    Returns:
        args: Updated command line arguments
    """
    if best_params is None:
        return args

    # Map Optuna parameter names to argument names
    param_mapping = {
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "freeze_backbone": "freeze_backbone",
        "dropout_rate": "dropout_rate",
    }

    # Update args with best parameters
    for optuna_param, arg_name in param_mapping.items():
        if optuna_param in best_params:
            setattr(args, arg_name, best_params[optuna_param])

    return args


if __name__ == "__main__":
    print(
        "Optuna utilities module loaded. Use run_optuna_study() to start optimization."
    )
