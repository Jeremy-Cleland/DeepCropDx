# Model utilities for loading, saving, and manipulating models
import torch
import logging

logger = logging.getLogger(__name__)

def safe_torch_load(model_path, map_location="cpu", weights_only=True, fallback_to_unsafe=True):
    """
    Safely load a PyTorch model with weights_only=True by adding necessary globals
    
    Args:
        model_path (str): Path to the model file
        map_location (str or torch.device): Device to map model to
        weights_only (bool): Whether to load only weights (True) or allow code execution (False)
        fallback_to_unsafe (bool): Whether to fallback to weights_only=False if weights_only=True fails
        
    Returns:
        dict or nn.Module: The loaded checkpoint
    """
    try:
        # Add numpy.core.multiarray._reconstruct to safe globals
        import torch.serialization
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
        
        # Try loading with weights_only=True
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=weights_only)
        return checkpoint
    except Exception as e:
        if fallback_to_unsafe and weights_only:
            logger.warning(f"Failed to load with weights_only=True: {str(e)}")
            logger.warning("Falling back to weights_only=False, which allows arbitrary code execution")
            return torch.load(model_path, map_location=map_location, weights_only=False)
        else:
            # Re-raise the exception if we don't want to fallback
            raise