import torch
import torch.serialization
import numpy as np


def load_checkpoint(checkpoint_path, model, optimizer=None):
    # Add numpy.ndarray to safe globals to prevent the unpickling error
    torch.serialization.add_safe_globals([np.ndarray])

    # Load checkpoint with weights_only=True
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Apply checkpoint to model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
