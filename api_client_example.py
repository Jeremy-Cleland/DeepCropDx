#!/usr/bin/env python3
"""
Demo client for the Crop Disease Detection API.
This script shows how to use the API programmatically.
"""

import os
import sys
import json
import argparse
import requests
from PIL import Image
from datetime import datetime


def get_available_models(base_url):
    """Get list of available models"""
    url = f"{base_url}/api/models"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting models: {response.status_code}")
        print(response.text)
        return None


def set_current_model(base_url, model_id):
    """Set the current model"""
    url = f"{base_url}/api/set-model/{model_id}"
    response = requests.post(url)

    if response.status_code == 200:
        result = response.json()
        print(f"Current model set to: {result['model_name']}")
        return True
    else:
        print(f"Error setting model: {response.status_code}")
        print(response.text)
        return False


def get_diagnosis_history(base_url):
    """Get diagnosis history"""
    url = f"{base_url}/api/history"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting history: {response.status_code}")
        print(response.text)
        return None


def diagnose_image(base_url, image_path, model_id=None, crop_type=None):
    """Submit an image for diagnosis"""
    url = f"{base_url}/api/diagnose"

    # Verify file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None

    # Prepare the multipart form data
    files = {"file": open(image_path, "rb")}
    data = {}

    if model_id:
        data["model_id"] = model_id

    if crop_type:
        data["crop_type"] = crop_type

    # Send request
    try:
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error diagnosing image: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None
    finally:
        # Close file
        files["file"].close()


def display_diagnosis(diagnosis):
    """Display diagnosis results in a nice format"""
    if not diagnosis:
        return

    if "status" in diagnosis and diagnosis["status"] == "error":
        print(f"Error: {diagnosis['error']}")
        return

    if "prediction" in diagnosis:
        # API v2 format
        prediction = diagnosis["prediction"]
        metadata = diagnosis.get("metadata", {})
        disease_info = diagnosis.get("disease_info", {})

        print("\n" + "=" * 60)
        print(f"DIAGNOSIS RESULT")
        print("=" * 60)
        print(f"Prediction: {prediction['class']}")
        print(f"Confidence: {prediction['confidence']*100:.1f}%")
        print(f"Model: {metadata.get('model_name', 'Unknown')}")
        print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
        if metadata.get("crop_type"):
            print(f"Crop Type: {metadata['crop_type']}")

        print("\nProbabilities:")
        for disease, prob in sorted(
            prediction["probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  - {disease}: {prob*100:.1f}%")

        if disease_info:
            print("\nDisease Information:")
            print("-" * 40)
            print(disease_info.get("description", "No description available"))

            if "treatment" in disease_info:
                print("\nTreatment Recommendations:")
                print("-" * 40)
                print(disease_info["treatment"])
    else:
        # API v1 format or other
        prediction = diagnosis.get("prediction", "Unknown")
        confidence = diagnosis.get("confidence", 0) * 100

        print("\n" + "=" * 60)
        print(f"DIAGNOSIS RESULT")
        print("=" * 60)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.1f}%")

        if "probabilities" in diagnosis:
            print("\nProbabilities:")
            for prob in diagnosis["probabilities"]:
                print(f"  - {prob['class']}: {prob['percentage']}")

        if "disease_info" in diagnosis:
            print("\nDisease Information:")
            print("-" * 40)
            print(
                diagnosis["disease_info"].get("description", "No description available")
            )

            if "treatment" in diagnosis["disease_info"]:
                print("\nTreatment Recommendations:")
                print("-" * 40)
                print(diagnosis["disease_info"]["treatment"])

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Crop Disease Detection API Client")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:5000",
        help="Base URL of the API (default: http://127.0.0.1:5000)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")

    # Set model command
    set_model_parser = subparsers.add_parser("set-model", help="Set current model")
    set_model_parser.add_argument("model_id", type=str, help="ID of the model to use")

    # History command
    history_parser = subparsers.add_parser("history", help="Get diagnosis history")

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose an image")
    diagnose_parser.add_argument("image", type=str, help="Path to image file")
    diagnose_parser.add_argument("--model", type=str, help="Specific model to use")
    diagnose_parser.add_argument(
        "--crop", type=str, help="Crop type (e.g., rice, tomato)"
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "models":
        models = get_available_models(args.url)
        if models:
            print("\nAvailable Models:")
            print("-" * 40)
            for model in models:
                current = " (current)" if model["is_current"] else ""
                print(f"- {model['name']}: {model['num_classes']} classes{current}")
            print()

    elif args.command == "set-model":
        set_current_model(args.url, args.model_id)

    elif args.command == "history":
        history = get_diagnosis_history(args.url)
        if history:
            print("\nDiagnosis History:")
            print("-" * 60)
            for entry in history:
                confidence = entry.get("confidence", 0) * 100
                print(
                    f"- {entry['date']}: {entry['prediction']} ({confidence:.1f}%) [Model: {entry['model_id']}]"
                )
            print()

    elif args.command == "diagnose":
        print(f"Diagnosing image: {args.image}")
        if args.model:
            print(f"Using model: {args.model}")
        if args.crop:
            print(f"Crop type: {args.crop}")

        diagnosis = diagnose_image(args.url, args.image, args.model, args.crop)
        display_diagnosis(diagnosis)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
