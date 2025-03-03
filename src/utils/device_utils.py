"""
Device utilities for handling different compute devices (CPU, CUDA, MPS)
"""

import os
import torch


def get_device(no_cuda=False, no_mps=False, verbose=True):
    """
    Get the best available device with priority: CUDA > MPS > CPU

    Args:
        no_cuda (bool): If True, avoid using CUDA even if available
        no_mps (bool): If True, avoid using MPS even if available
        verbose (bool): If True, print device information

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using CUDA device: {gpu_name} with {gpu_mem:.1f} GB memory")
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not no_mps
    ):
        device = torch.device("mps")
        if verbose:
            print("Using MPS (Apple Silicon GPU)")
            # Configure MPS environment for better performance
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("MPS fallback enabled for unsupported operations")
    else:
        device = torch.device("cpu")
        if verbose:
            import multiprocessing

            num_cores = multiprocessing.cpu_count()
            print(f"Using CPU with {num_cores} cores")

    return device


def is_apple_silicon():
    """Check if running on Apple Silicon"""
    import platform

    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
    )


def optimize_for_device(device, model=None):
    """
    Apply device-specific optimizations

    Args:
        device (torch.device): Device to optimize for
        model (nn.Module, optional): Model to optimize

    Returns:
        model: Optimized model (if provided)
    """
    if device.type == "cuda":
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            # Set memory strategy for better performance
            torch.cuda.empty_cache()
            # Use TF32 precision if available (Ampere+ GPUs)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                print("TF32 precision enabled for faster training")

    elif device.type == "mps":
        # Import MPS optimizations and apply them
        try:
            from src.utils.mps_optimizations import configure_mps_environment
            
            # M4 Max specific optimizations
            if is_apple_silicon():
                import platform
                mac_model = platform.machine()
                # Check Apple hardware model if possible
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', 'hw.model'], capture_output=True, text=True)
                    if result.returncode == 0:
                        hardware_model = result.stdout.strip().split(': ')[1]
                        print(f"Detected Apple hardware: {hardware_model}")
                        
                        # M4 Max-specific optimizations
                        if "Mac" in hardware_model and any(x in hardware_model for x in ["M4", "M3", "M2"]):
                            print("Enabling advanced Apple Silicon optimizations for M-series chips")
                            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # Use more GPU memory
                            # Check if we have PyTorch 2.2+
                            if torch.__version__ >= "2.2.0":
                                print("Enabling PyTorch 2.2+ MPS optimizations")
                                # PyTorch 2.2+ specific optimizations
                                if hasattr(torch.backends.mps, "set_benchmark_mode"):
                                    torch.backends.mps.set_benchmark_mode(True)
                except:
                    print("Could not determine exact Apple hardware model")
                    
            configure_mps_environment()
        except ImportError:
            # MPS optimizations module not available
            print("MPS optimizations module not available, using basic settings")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Model-specific optimizations
    if model is not None:
        if device.type == "mps":
            # Use float32 for best performance on MPS
            model = model.float()
            
            # For PyTorch 2.0+, attempt model compilation with optimizations for Apple Silicon
            if hasattr(torch, "compile"):
                try:
                    # Check PyTorch version to use appropriate compile options
                    if hasattr(torch, "__version__") and torch.__version__ >= "2.2.0":
                        # PyTorch 2.2+ has better MPS compilation support
                        print("Using advanced torch.compile() for PyTorch 2.2+ on Apple Silicon")
                        # Use "reduce-overhead" mode which works well on Apple Silicon
                        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                    else:
                        # Basic compilation for older PyTorch versions
                        print("Applied torch.compile() optimization for MPS")
                        model = torch.compile(model)
                    
                    print("Model compilation successful")
                except Exception as e:
                    print(f"Could not apply torch.compile(): {e}")
                    print("Continuing with uncompiled model")
            
            # For M-series chips, apply additional optimization: precompute constants
            if is_apple_silicon():
                def apply_mps_optimizations(module):
                    # Convert all parameters to float32 for best MPS performance
                    for param in module.parameters():
                        if param.requires_grad:
                            param.data = param.data.float()
                
                # Apply optimizations recursively to all modules
                model.apply(apply_mps_optimizations)

    return model if model is not None else None


if __name__ == "__main__":
    # Test device detection
    device = get_device()
    print(f"Best available device: {device}")

    # Test Apple Silicon detection
    if is_apple_silicon():
        print("Running on Apple Silicon")
    else:
        print("Not running on Apple Silicon")
