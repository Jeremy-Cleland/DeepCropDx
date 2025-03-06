# src/scripts/verify_models.py
import argparse
import sys
import os

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.model_tester import test_all_models, test_model_forward_backward


def main():
    parser = argparse.ArgumentParser(description="Test model implementations")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to test, or all models if not specified",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=39,
        help="Number of classes to use for testing",
    )
    args = parser.parse_args()

    if args.model:
        success = test_model_forward_backward(args.model, num_classes=args.num_classes)
        sys.exit(0 if success else 1)
    else:
        success = test_all_models()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
