import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from datetime import datetime

# Import project modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.training.evaluate import load_model, predict_single_image
from src.utils.device_utils import get_device

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration
app.config["UPLOAD_FOLDER"] = "src/app/static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables for models and device
DEVICE = get_device()
MODELS = {}  # Dictionary to store multiple loaded models
CURRENT_MODEL_ID = None
IMG_SIZE = 224

# Directory for storing diagnosis history
HISTORY_DIR = os.path.join("src", "app", "static", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def load_model_for_inference(model_path, model_id=None):
    """
    Load model for inference

    Args:
        model_path (str): Path to model checkpoint
        model_id (str, optional): Identifier for the model. If None, uses the filename.

    Returns:
        tuple: (model, class_to_idx, model_id)
    """
    global MODELS, CURRENT_MODEL_ID, DEVICE

    # Generate model_id from filename if not provided
    if model_id is None:
        model_id = os.path.basename(model_path)

    # Check if model is already loaded
    if model_id not in MODELS:
        model, class_to_idx = load_model(model_path, DEVICE)
        model.eval()

        # Store model info
        MODELS[model_id] = {
            "model": model,
            "class_to_idx": class_to_idx,
            "path": model_path,
            "name": model_id,
            "num_classes": len(class_to_idx),
        }

        print(f"Model '{model_id}' loaded successfully on {DEVICE}")
        print(f"Detected {len(class_to_idx)} classes")

    # Set as current model if this is the first one or explicitly requested
    if CURRENT_MODEL_ID is None:
        CURRENT_MODEL_ID = model_id

    return MODELS[model_id]["model"], MODELS[model_id]["class_to_idx"], model_id


def save_to_history(result, image_path):
    """Save diagnosis result to history"""
    history_file = os.path.join(HISTORY_DIR, "diagnosis_history.json")

    # Create history entry
    history_entry = {
        "id": f"{result['timestamp']}_{result['prediction']}",
        "timestamp": result["timestamp"],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "model_id": result["model_id"],
        "image_path": result["image_path"],
    }

    # Load existing history or create new
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except:
            history = []

    # Add new entry
    history.append(history_entry)

    # Save updated history
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

    return True


@app.route("/")
def index():
    """Render the main page"""
    # Create a dictionary of class descriptions and treatments if available
    class_info = {}
    try:
        with open("src/app/static/class_info.json", "r") as f:
            class_info = json.load(f)
    except FileNotFoundError:
        # Create default class info if file doesn't exist
        if CURRENT_MODEL_ID in MODELS:
            class_to_idx = MODELS[CURRENT_MODEL_ID]["class_to_idx"]
            class_info = {
                cls: {
                    "description": f"Information about {cls}",
                    "treatment": "Generic treatment information",
                }
                for cls in class_to_idx.keys()
            }

            # Save default class info
            with open("src/app/static/class_info.json", "w") as f:
                json.dump(class_info, f, indent=4)

    # Get available models for the dropdown
    available_models = [
        {"id": model_id, "name": info["name"]} for model_id, info in MODELS.items()
    ]

    return render_template(
        "index.html",
        class_info=class_info,
        available_models=available_models,
        current_model=CURRENT_MODEL_ID,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get model_id parameter if it's provided, otherwise use current model
    model_id = request.form.get("model_id", CURRENT_MODEL_ID)

    # Check if the requested model exists
    if model_id not in MODELS:
        return jsonify({"error": f"Model {model_id} not found"}), 404

    if file and allowed_file(file.filename):
        # Create a timestamp-based filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        base, ext = os.path.splitext(original_filename)
        filename = f"{base}_{timestamp}{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Make prediction
        try:
            model = MODELS[model_id]["model"]
            class_to_idx = MODELS[model_id]["class_to_idx"]

            predicted_class, confidence, class_probabilities = predict_single_image(
                model, file_path, class_to_idx, img_size=IMG_SIZE, device=DEVICE
            )

            # Sort probabilities for display
            sorted_probs = sorted(
                [(cls, prob) for cls, prob in class_probabilities.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            # Format probabilities for display
            formatted_probs = [
                {
                    "class": cls,
                    "probability": float(prob),
                    "percentage": f"{prob*100:.2f}%",
                }
                for cls, prob in sorted_probs
            ]

            # Prepare response
            result = {
                "prediction": predicted_class,
                "confidence": float(confidence),
                "confidence_percentage": f"{confidence*100:.2f}%",
                "probabilities": formatted_probs,
                "image_path": url_for("static", filename=f"uploads/{filename}"),
                "model_id": model_id,
                "model_name": MODELS[model_id]["name"],
                "timestamp": timestamp,
            }

            # Save diagnosis to history if enabled
            save_to_history(result, file_path)

            # Get disease info if available
            try:
                with open("src/app/static/class_info.json", "r") as f:
                    class_info = json.load(f)

                if predicted_class in class_info:
                    result["disease_info"] = class_info[predicted_class]
            except Exception as e:
                print(f"Error loading disease info: {str(e)}")

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format"}), 400


@app.route("/api/models", methods=["GET"])
def get_models():
    """API endpoint to get available models"""
    models_list = [
        {
            "id": model_id,
            "name": info["name"],
            "num_classes": info["num_classes"],
            "is_current": model_id == CURRENT_MODEL_ID,
        }
        for model_id, info in MODELS.items()
    ]
    return jsonify(models_list)


@app.route("/api/set-model/<model_id>", methods=["POST"])
def set_current_model(model_id):
    """API endpoint to set the current model"""
    global CURRENT_MODEL_ID

    if model_id not in MODELS:
        return jsonify({"error": f"Model {model_id} not found"}), 404

    CURRENT_MODEL_ID = model_id
    return jsonify(
        {
            "success": True,
            "current_model": model_id,
            "model_name": MODELS[model_id]["name"],
        }
    )


@app.route("/api/history", methods=["GET"])
def get_history():
    """API endpoint to get diagnosis history"""
    history_file = os.path.join(HISTORY_DIR, "diagnosis_history.json")

    if not os.path.exists(history_file):
        return jsonify([])

    with open(history_file, "r") as f:
        history = json.load(f)

    # Return most recent entries first
    return jsonify(sorted(history, key=lambda x: x["timestamp"], reverse=True))


@app.route("/history")
def history_page():
    """Render the history page"""
    return render_template("history.html")


@app.route("/generate-report/<history_id>", methods=["GET"])
def generate_report(history_id):
    """Generate a PDF report for a diagnosis"""
    try:
        # Find the diagnosis in history
        history_file = os.path.join(HISTORY_DIR, "diagnosis_history.json")
        if not os.path.exists(history_file):
            return jsonify({"error": "No history found"}), 404

        with open(history_file, "r") as f:
            history = json.load(f)

        # Find the specific diagnosis
        diagnosis = None
        for entry in history:
            if entry.get("id") == history_id:
                diagnosis = entry
                break

        if not diagnosis:
            return jsonify({"error": "Diagnosis not found"}), 404

        # Get disease information
        disease_info = {}
        try:
            with open("src/app/static/class_info.json", "r") as f:
                class_info = json.load(f)

            if diagnosis["prediction"] in class_info:
                disease_info = class_info[diagnosis["prediction"]]
        except:
            pass

        # Generate HTML report
        report_html = render_template(
            "report.html",
            diagnosis=diagnosis,
            disease_info=disease_info,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # For now, return the HTML report directly
        # In a production system, you'd convert this to PDF
        return report_html

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    """API endpoint for programmatic crop disease diagnosis"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get model_id parameter if it's provided, otherwise use current model
    model_id = request.form.get("model_id", CURRENT_MODEL_ID)
    crop_type = request.form.get("crop_type", None)

    # Check if the requested model exists
    if model_id not in MODELS:
        return jsonify({"error": f"Model {model_id} not found"}), 404

    if file and allowed_file(file.filename):
        # Create a timestamp-based filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        base, ext = os.path.splitext(original_filename)
        filename = f"{base}_{timestamp}{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Make prediction
        try:
            model = MODELS[model_id]["model"]
            class_to_idx = MODELS[model_id]["class_to_idx"]

            predicted_class, confidence, class_probabilities = predict_single_image(
                model, file_path, class_to_idx, img_size=IMG_SIZE, device=DEVICE
            )

            # Format probabilities
            formatted_probs = {
                cls: float(prob) for cls, prob in class_probabilities.items()
            }

            # Prepare API response
            result = {
                "status": "success",
                "prediction": {
                    "class": predicted_class,
                    "confidence": float(confidence),
                    "probabilities": formatted_probs,
                },
                "metadata": {
                    "model_id": model_id,
                    "model_name": MODELS[model_id]["name"],
                    "timestamp": timestamp,
                    "crop_type": crop_type,
                    "image_filename": filename,
                },
            }

            # Get disease info if available
            try:
                with open("src/app/static/class_info.json", "r") as f:
                    class_info = json.load(f)

                if predicted_class in class_info:
                    result["disease_info"] = class_info[predicted_class]
            except Exception as e:
                print(f"Error loading disease info: {str(e)}")

            return jsonify(result)

        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    return jsonify({"status": "error", "error": "Invalid file format"}), 400


def start_app(model_path, host="0.0.0.0", port=5000, debug=False, model_id=None):
    """Start the Flask application"""
    # Load model before starting app
    load_model_for_inference(model_path, model_id)

    # Load additional models if specified in the models directory
    models_dir = os.path.dirname(model_path)
    for filename in os.listdir(models_dir):
        if (
            filename.endswith(".pth")
            and os.path.join(models_dir, filename) != model_path
        ):
            try:
                additional_model_path = os.path.join(models_dir, filename)
                load_model_for_inference(additional_model_path)
                print(f"Loaded additional model: {filename}")
            except Exception as e:
                print(f"Error loading additional model {filename}: {str(e)}")

    # Run the Flask app
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Start the crop disease diagnosis web app"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model checkpoint"
    )
    parser.add_argument("--model-id", type=str, help="Identifier for the model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_app(args.model, args.host, args.port, args.debug, args.model_id)
