import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# Import create_model from your model factory
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.models.model_factory import create_model


def test_model_forward_backward(model_type, num_classes=39, batch_size=2, img_size=224):
    """
    Test complete forward and backward pass for a model architecture.

    Args:
        model_type: Type of model to test
        num_classes: Number of output classes
        batch_size: Batch size for testing
        img_size: Input image size

    Returns:
        bool: Whether the test was successful
    """
    print(f"\n====== Testing model: {model_type} ======")

    # Create model
    try:
        model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            test_forward=False,  # We'll do our own testing
        )
        print(f"✓ Model creation successful")
    except Exception as e:
        print(f"✗ ERROR: Model creation failed: {str(e)}")
        return False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params:.1%})"
    )

    # Move to CPU for testing
    device = torch.device("cpu")
    model = model.to(device)

    # Test forward pass
    try:
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
        dummy_target = torch.randint(0, num_classes, (batch_size,), device=device)

        # Test with tracing
        with torch.autograd.detect_anomaly():
            outputs = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {outputs.shape}")

        if outputs.shape != (batch_size, num_classes):
            print(
                f"✗ WARNING: Output shape mismatch: got {outputs.shape}, expected {(batch_size, num_classes)}"
            )
            return False

    except Exception as e:
        print(f"✗ ERROR: Forward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    # Test backward pass
    try:
        # Loss calculation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_target)
        print(f"  Initial loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print(f"✓ Backward pass successful")

        # Verify gradients exist for trainable parameters
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                    grad_norm = torch.norm(param.grad).item()
                    if grad_norm > 0:
                        print(f"  Gradient norm for {name}: {grad_norm:.6f}")
                    else:
                        print(f"  WARNING: Zero gradient for {name}")

        if not has_grad:
            print("✗ ERROR: No gradients computed during backward pass")
            return False

    except Exception as e:
        print(f"✗ ERROR: Backward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    # Test optimization step
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.step()
        print(f"✓ Optimization step successful")
    except Exception as e:
        print(f"✗ ERROR: Optimization step failed: {str(e)}")
        return False

    # Test second forward-backward pass
    try:
        outputs2 = model(dummy_input)
        loss2 = criterion(outputs2, dummy_target)
        print(f"  Loss after one step: {loss2.item():.4f}")

        if loss2.item() == loss.item():
            print("✗ WARNING: Loss didn't change after optimization step")

        loss2.backward()
        optimizer.step()
        print(f"✓ Second forward-backward pass successful")
    except Exception as e:
        print(f"✗ ERROR: Second forward-backward pass failed: {str(e)}")
        return False

    # Test inference mode
    try:
        model.eval()
        with torch.no_grad():
            inference_start = time.time()
            inference_output = model(dummy_input)
            inference_time = time.time() - inference_start

        print(
            f"✓ Inference successful ({inference_time*1000:.2f} ms per batch of {batch_size})"
        )
    except Exception as e:
        print(f"✗ ERROR: Inference failed: {str(e)}")
        return False

    # Test with different batch sizes
    for test_batch in [1, 4, 8]:
        try:
            test_input = torch.randn(test_batch, 3, img_size, img_size, device=device)
            with torch.no_grad():
                test_output = model(test_input)
            print(f"✓ Batch size {test_batch} test passed")
        except Exception as e:
            print(f"✗ ERROR: Batch size {test_batch} test failed: {str(e)}")
            return False

    print(f"\n✅ Model {model_type} passed all tests!\n")
    return True


def test_all_models():
    """Test all model architectures supported by the factory."""
    model_types = [
        "efficientnet",
        "efficientnet_b3",
        "mobilenet",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "mobilenet_attention",
        "resnet18",
        "resnet50",
        "resnet_attention",
    ]

    results = {}

    for model_type in model_types:
        results[model_type] = test_model_forward_backward(model_type)

    # Print summary
    print("\n===== MODEL TEST SUMMARY =====")
    for model_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_type}: {status}")

    # Return overall success
    return all(results.values())


if __name__ == "__main__":
    # If file is run directly, test all models
    test_all_models()
