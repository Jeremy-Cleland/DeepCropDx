"""
Optimizations specifically for MPS (Metal Performance Shaders) on Apple Silicon
"""

import os
import torch
import numpy as np
import time


def configure_mps_environment():
    """Configure environment variables for optimal MPS performance"""
    if not is_mps_available():
        print("MPS is not available on this system")
        return False

    # Enable fallback for operations not supported by MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # For latest M-series chips (M2/M3/M4), optimize memory usage
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Maximize GPU memory usage
    
    # Thread optimizations
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))  # Limit MKL threads
    
    # For PyTorch 2.0+
    if hasattr(torch.backends.mps, "enable_graph_mode"):
        torch.backends.mps.enable_graph_mode = True
        print("MPS graph mode enabled")
    
    # For PyTorch 2.2+
    if hasattr(torch, "__version__") and torch.__version__ >= "2.2.0":
        if hasattr(torch.backends.mps, "set_benchmark_mode"):
            torch.backends.mps.set_benchmark_mode(True)
            print("MPS benchmark mode enabled (PyTorch 2.2+)")
            
    print("MPS environment configured for optimal performance on Apple Silicon")
    return True


def is_mps_available():
    """Check if MPS is available on the current system"""
    return (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def get_optimal_mps_batch_size(
    model, input_size=(3, 224, 224), start_batch=32, device=None
):
    """
    Find the optimal batch size for MPS by testing memory usage

    Args:
        model: The PyTorch model
        input_size: Input tensor shape (channels, height, width)
        start_batch: Starting batch size to test
        device: Device to use (defaults to mps if available)

    Returns:
        int: Optimal batch size
    """
    if not is_mps_available():
        print("MPS not available, returning default batch size")
        return start_batch

    if device is None:
        device = torch.device("mps")

    model = model.to(device)
    model.eval()

    # Try decreasing batch sizes
    batch_size = start_batch
    while batch_size > 1:
        try:
            # Create a dummy input
            dummy_input = torch.rand(batch_size, *input_size, device=device)

            # Test forward pass
            with torch.no_grad():
                _ = model(dummy_input)

            # If successful, break
            print(f"Batch size {batch_size} works on MPS")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce batch size and try again
                batch_size //= 2
                print(f"Reducing batch size to {batch_size}")
                # Clear memory
                torch.mps.empty_cache()
            else:
                # If it's not a memory error, raise it
                raise e

    return max(1, batch_size)


def benchmark_mps_performance(
    model, input_size=(3, 224, 224), batch_size=16, iterations=100
):
    """
    Benchmark model performance on MPS

    Args:
        model: PyTorch model to benchmark
        input_size: Input tensor shape
        batch_size: Batch size for testing
        iterations: Number of iterations to run

    Returns:
        float: Average inference time per batch in milliseconds
    """
    if not is_mps_available():
        print("MPS not available for benchmarking")
        return None

    device = torch.device("mps")
    model = model.to(device)
    model.eval()

    dummy_input = torch.rand(batch_size, *input_size, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize before timing
    torch.mps.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize before stopping timer
    torch.mps.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) * 1000 / iterations  # convert to ms
    print(f"MPS inference time: {avg_time:.2f} ms per batch (batch size: {batch_size})")
    return avg_time


def optimize_mps_model(model):
    """
    Apply optimizations to improve model performance on MPS

    Args:
        model: PyTorch model

    Returns:
        model: Optimized model
    """
    if not is_mps_available():
        return model

    # For PyTorch 2.0+, use torch.compile if available
    if hasattr(torch, "compile"):
        try:
            print("Applying torch.compile() optimization")
            model = torch.compile(model)
        except Exception as e:
            print(f"Could not apply torch.compile(): {e}")

    # Ensure model is using float32 for best MPS compatibility
    model = model.float()

    return model


def convert_convolutions_to_mps_friendly(model):
    """
    Convert model convolutions to formats that work better with MPS

    Some specific convolution types or parameters may not be optimally
    supported by MPS. This function replaces them with MPS-friendly alternatives.

    Args:
        model: PyTorch model

    Returns:
        model: Modified model with MPS-friendly convolutions
    """
    if not is_mps_available():
        return model

    # This is a placeholder for more specific optimizations
    # In a real implementation, you would identify problematic layers
    # and replace them with more MPS-friendly alternatives

    return model


if __name__ == "__main__":
    # Simple test to verify MPS is working
    if is_mps_available():
        print("MPS is available!")
        device = torch.device("mps")

        # Create a simple test tensor and verify it works on MPS
        test_tensor = torch.rand(1000, 1000, device=device)
        result = torch.mm(test_tensor, test_tensor)

        print(f"Test tensor shape: {result.shape}")
        print("MPS test successful!")

        # Configure MPS for optimal performance
        configure_mps_environment()
    else:
        print("MPS is not available on this system.")
