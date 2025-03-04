#!/usr/bin/env python3
"""
Script to compare multiple model evaluation results and generate a comprehensive report.


```
python src/scripts/compare_models.py --models efficientnet_b0_v1 efficientnet_b3_v1 mobilenet_v2 mobilenet_v3
```

"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.model_comparison import ModelComparisonReport

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("compare_models")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model evaluation results.")
    parser.add_argument(
        "--evaluations_dir",
        type=str,
        default="reports/evaluations",
        help="Directory containing model evaluation results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/comparisons",
        help="Directory to save comparison results.",
    )
    parser.add_argument(
        "--report_id",
        type=str,
        default=None,
        help="Custom ID for the report (defaults to timestamp).",
    )
    parser.add_argument(
        "--report_title",
        type=str,
        default="Crop Disease Detection Model Comparison",
        help="Title for the comparison report.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Specific model directories to include (defaults to all).",
    )
    return parser.parse_args()


def main():
    """Main function to run the model comparison."""
    args = parse_arguments()

    # Log execution parameters
    logger.info(f"Starting model comparison with the following parameters:")
    logger.info(f"- Evaluations directory: {args.evaluations_dir}")
    logger.info(f"- Output directory: {args.output_dir}")
    logger.info(f"- Report ID: {args.report_id or 'auto-generated'}")
    logger.info(f"- Report title: {args.report_title}")
    logger.info(f"- Specific models: {args.models or 'all available'}")

    # Validate directories
    eval_dir = Path(args.evaluations_dir)
    if not eval_dir.exists():
        logger.error(f"Evaluations directory not found: {eval_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison report
    report = ModelComparisonReport(output_dir=str(output_dir), report_id=args.report_id)

    # Add specified models or scan all
    if args.models:
        for model in args.models:
            model_dir = eval_dir / model
            if model_dir.exists():
                report.add_model_from_evaluation_dir(model, str(model_dir))
            else:
                logger.warning(f"Model directory not found: {model_dir}")
    else:
        report.scan_evaluation_directories(base_dir=str(eval_dir))

    # Generate the report
    report_path = report.generate_report(title=args.report_title)

    if report_path:
        logger.info(f"Report generated successfully at: {report_path}")
        return 0
    else:
        logger.error("Failed to generate comparison report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
