#!/usr/bin/env python3
"""
Launch script for the Crop Disease Diagnosis web application.
"""

import os
import argparse
from src.app.flask_app import start_app


def main():
    parser = argparse.ArgumentParser(
        description="Start the Crop Disease Diagnosis web application"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model checkpoint"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to run the server on (default: 127.0.0.1, use 0.0.0.0 to make publicly accessible)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Run the app in debug mode"
    )

    args = parser.parse_args()

    # Ensure model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Starting Crop Disease Diagnosis app with model: {args.model}")
    print(f"Server will be accessible at: http://{args.host}:{args.port}")

    # Start the Flask app
    start_app(model_path=args.model, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
