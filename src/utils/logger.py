import os
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize training logger

        Args:
            log_dir (str): Directory to save logs
            experiment_name (str, optional): Name of the experiment
        """
        os.makedirs(log_dir, exist_ok=True)

        # Set up experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Set up file logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Initialize metrics tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        # Track timing information
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None

        self.total_epochs = 0
        self.current_epoch = 0
        self.batches_per_epoch = {"train": 0, "val": 0}

        self.log_separator = "-" * 80

    def start_training(self, total_epochs, batches_per_epoch):
        """
        Log start of training

        Args:
            total_epochs (int): Total number of epochs
            batches_per_epoch (dict): Number of batches per epoch for each phase
        """
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.start_time = time.time()

        self.logger.info(self.log_separator)
        self.logger.info(f"Starting training: {self.experiment_name}")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Training batches per epoch: {batches_per_epoch['train']}")
        self.logger.info(f"Validation batches per epoch: {batches_per_epoch['val']}")
        self.logger.info(self.log_separator)

    def start_epoch(self, epoch):
        """
        Log start of epoch

        Args:
            epoch (int): Current epoch number
        """
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

        self.logger.info(f"Epoch {epoch}/{self.total_epochs}")
        self.logger.info("-" * 40)

    def log_batch(
        self, phase, batch_idx, loss, progress=None, metrics=None, log_every=10
    ):
        """
        Log batch information

        Args:
            phase (str): Training phase ('train' or 'val')
            batch_idx (int): Current batch index
            loss (float): Current batch loss
            progress (float, optional): Percentage of completion
            metrics (dict, optional): Additional metrics
            log_every (int): Log frequency in number of batches
        """
        if (
            batch_idx % log_every != 0
            and batch_idx != self.batches_per_epoch[phase] - 1
        ):
            return

        if progress is None:
            progress = 100.0 * batch_idx / self.batches_per_epoch[phase]

        batch_time = time.time() - self.batch_start_time if self.batch_start_time else 0
        self.batch_start_time = time.time()

        message = (
            f"[{phase.upper()}] Epoch: {self.current_epoch}/{self.total_epochs} | "
        )
        message += (
            f"Batch: {batch_idx}/{self.batches_per_epoch[phase]} ({progress:.1f}%) | "
        )
        message += f"Loss: {loss:.4f} | Batch time: {batch_time:.2f}s"

        if metrics:
            for key, value in metrics.items():
                message += f" | {key}: {value:.4f}"

        self.logger.info(message)

    def end_epoch(self, train_metrics, val_metrics):
        """
        Log end of epoch with metrics

        Args:
            train_metrics (dict): Training metrics
            val_metrics (dict): Validation metrics
        """
        epoch_time = time.time() - self.epoch_start_time

        # Update history
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["train_acc"].append(train_metrics["accuracy"])
        self.history["train_f1"].append(train_metrics["f1"])

        self.history["val_loss"].append(val_metrics["loss"])
        self.history["val_acc"].append(val_metrics["accuracy"])
        self.history["val_f1"].append(val_metrics["f1"])

        self.logger.info(self.log_separator)
        self.logger.info(
            f"Epoch {self.current_epoch}/{self.total_epochs} completed in {epoch_time:.2f}s"
        )
        self.logger.info(
            f"Training   - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}"
        )
        self.logger.info(
            f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}"
        )

        # Check for improvement
        if len(self.history["val_f1"]) > 1:
            prev_best_f1 = max(self.history["val_f1"][:-1])
            current_f1 = self.history["val_f1"][-1]

            if current_f1 > prev_best_f1:
                self.logger.info(
                    f"Validation F1 improved from {prev_best_f1:.4f} to {current_f1:.4f}"
                )

        self.logger.info(self.log_separator)

    def end_training(self):
        """Log end of training with summary"""
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        best_epoch = np.argmax(self.history["val_f1"]) + 1
        best_val_f1 = max(self.history["val_f1"])

        self.logger.info(self.log_separator)
        self.logger.info(
            f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s"
        )
        self.logger.info(f"Best validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        self.logger.info(self.log_separator)

        # Plot training curves
        self._plot_training_curves()

    def log_checkpoint(self, epoch, filename):
        """Log checkpoint saving"""
        self.logger.info(f"Checkpoint saved: {filename} (Epoch {epoch})")

    def _plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot loss
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Training Loss")
        ax1.plot(epochs, self.history["val_loss"], "r-", label="Validation Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot F1 score
        ax2.plot(epochs, self.history["train_f1"], "b-", label="Training F1")
        ax2.plot(epochs, self.history["val_f1"], "r-", label="Validation F1")
        ax2.set_title("F1 Score")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("F1 Score")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(self.log_dir, f"{self.experiment_name}_training_curves.png")
        )
        plt.close()
