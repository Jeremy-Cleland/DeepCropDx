![Crop Disease Detection Banner](.github/assets/crop_disease_detection_01.jpg)

# **DeepCropDX | Deep Neural Network Plan Disease Diganostic System**

This repository presents a deep neural network pipeline combining EfficientNet, MobileNet, and ResNet variants with bespoke attention mechanisms, designed to accurately diagnose plant diseases from images. The integrated Flask-based web interface allows users to submit crop images and instantly receive diagnostic feedback.

## 📋 Table of Contents

- [**DeepCropDX | Deep Neural Network Plan Disease Diganostic System**](#deepcropdx--deep-neural-network-plan-disease-diganostic-system)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🚀 Installation](#-installation)
  - [📁 Project Structure](#-project-structure)
  - [📊 Usage](#-usage)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Starting with a Clean Environment](#starting-with-a-clean-environment)
    - [Full Pipeline Execution](#full-pipeline-execution)
    - [Visualization](#visualization)
  - [🧠 Model Architectures](#-model-architectures)
  - [🔬 Examples](#-examples)
    - [Training with EfficientNet-B3](#training-with-efficientnet-b3)
    - [Fine-tuning a pre-trained ResNet with attention](#fine-tuning-a-pre-trained-resnet-with-attention)
    - [Evaluating a trained model](#evaluating-a-trained-model)
    - [Running the Model Evaluation Pipeline](#running-the-model-evaluation-pipeline)
  - [🔧 Advanced Features](#-advanced-features)
    - [Device Support](#device-support)
    - [Class Imbalance](#class-imbalance)
    - [Experiment Tracking](#experiment-tracking)
    - [Model Registry](#model-registry)
  - [🚢 Deployment](#-deployment)
    - [Web Application](#web-application)
    - [Mobile Deployment](#mobile-deployment)
  - [📱 Web Interface](#-web-interface)
    - [Running the Web App](#running-the-web-app)
    - [Features](#features)
    - [Screenshots](#screenshots)
  - [🔌 API Integration](#-api-integration)
    - [API Endpoints](#api-endpoints)
    - [Using the API Client](#using-the-api-client)
    - [API Documentation](#api-documentation)
  - [📊 Evaluation Framework](#-evaluation-framework)
    - [Features](#features-1)
    - [Running Evaluations](#running-evaluations)
    - [Batch Evaluation](#batch-evaluation)
    - [Model Comparison Reports](#model-comparison-reports)
  - [🧩 Extending the Project](#-extending-the-project)
    - [Adding New Models](#adding-new-models)
    - [Custom Datasets](#custom-datasets)
  - [📈 Experimental Results](#-experimental-results)
  - [📚 Citation](#-citation)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## ✨ Features

- **Multiple Model Architectures**: Support for EfficientNet (B0, B3), MobileNet (V2, V3), and ResNet (18, 50) with custom modifications
- **Attention Mechanisms**: Implementation of attention modules to focus on disease-relevant features
- **Robust Data Processing**: Comprehensive data augmentation and stratified splitting
- **Transfer Learning**: Leveraging pre-trained models with customizable backbone freezing
- **Visualization Tools**: Extensive visualization capabilities including GradCAM, t-SNE, confusion matrices
- **Training Infrastructure**: Support for multiple devices (CPU, CUDA, MPS), checkpointing, and metrics tracking
- **Class Imbalance Handling**: Weighted loss functions to handle imbalanced datasets
- **Deployment Options**: Modern web application with user-friendly interface and mobile export capabilities
- **Apple Silicon Optimization**: Specific optimizations for Apple M-series chips using the MPS backend
- **Multi-model Support**: Load and switch between different models in the web interface
- **Diagnosis History**: Track and view historical diagnoses with detailed information
- **Report Generation**: Create and print comprehensive diagnosis reports
- **API Access**: RESTful API and Python client for programmatic access
- **Model Registry**: Centralized tracking of model information and performance metrics
- **Batch Evaluation**: Comprehensive evaluation of multiple models with comparison reports

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/jeremy-cleland/crop-disease-detection.git
cd crop-disease-detection

# Method 1: Using pip and virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Method 2: Using conda
conda env create -f environment.yml
conda activate crop-disease
```

## 📁 Project Structure

```
crop_disease_detection/
├── README.md                     # Project documentation
├── api_client_example.py         # Example API client
├── data/                         # Data directory
│   ├── external/                 # External data sources
│   ├── raw/                      # Raw, immutable data
│   └── processed/                # Processed dataset splits
├── docs/                         # Documentation
│   └── API_Documentation.md      # API documentation
├── environment.yml               # Conda environment specification
├── models/                       # Saved model checkpoints
├── notebooks/                    # Jupyter notebooks
│   ├── evaluation.ipynb          # Model evaluation examples
│   ├── exploratory_analysis.ipynb # Dataset exploration
│   └── training_demo.ipynb       # Training demonstration
├── reports/                      # Generated reports and visualizations
├── requirements.txt              # Python dependencies
├── run_app.py                    # Web application launcher
├── setup.py                      # Package installation script
├── train_all_models.sh           # Script to train all models in sequence
└── src/                          # Source code
    ├── __init__.py
    ├── app/                      # Application deployment code
    │   ├── __init__.py
    │   ├── flask_app.py          # Web application 
    │   ├── templates/            # HTML templates for web interface
    │   ├── static/               # Static assets (CSS, JS, images)
    │   └── mobile/               # Mobile deployment utilities
    ├── data/                     # Data processing modules
    │   ├── __init__.py
    │   └── dataset.py            # Dataset handling and augmentation
    ├── models/                   # Model architectures
    │   ├── __init__.py
    │   ├── efficientnet.py       # EfficientNet implementations
    │   ├── mobilenet.py          # MobileNet implementations
    │   ├── model_factory.py      # Factory for creating models
    │   └── resnet.py             # ResNet implementations
    ├── pipeline/                 # Pipeline modules
    │   └── train_evaluate_compare.py # End-to-end pipeline
    ├── training/                 # Training infrastructure
    │   ├── __init__.py
    │   ├── evaluate.py           # Evaluation utilities
    │   └── train.py              # Training script
    └── utils/                    # Utility functions
        ├── __init__.py
        ├── batch_evaluate.py     # Batch evaluation of models
        ├── benchmark.py          # Performance benchmarking
        ├── device_utils.py       # Device optimization utilities
        ├── metrics.py            # Evaluation metrics
        ├── model_comparison.py   # Model comparison reporting
        ├── logger.py             # Experiment logging
        └── visualization.py      # Visualization utilities
```

## 📊 Usage

### Data Preparation

Prepare your dataset with the following structure:

```
data/raw/
├── class1/                  # e.g., "healthy"
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/                  # e.g., "bacterial_blight"
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

The dataset processing utilities will automatically:

- Split data into training, validation, and test sets
- Generate class mappings
- Apply appropriate augmentations

### Training

Basic training command:

```bash
python -m src.training.train \
  --data_dir data/raw \
  --output_dir models/experiment1 \
  --model efficientnet \
  --img_size 224 \
  --batch_size 32 \
  --epochs 50 \
  --pretrained \
  --freeze_backbone
```

### Starting with a Clean Environment

Before running a new training process, you can clean up all previous model files, logs, and reports:

```bash
# Clean everything (will prompt for confirmation)
./cleanup.py

# Perform a dry run (show what would be removed without deleting)
./cleanup.py --dry-run

# Clean only specific categories
./cleanup.py --models    # Only remove model files
./cleanup.py --reports   # Only remove report files
./cleanup.py --logs      # Only remove log files

# Keep the model registry while cleaning models
./cleanup.py --models --keep-registry

# Skip confirmation prompt (use with caution)
./cleanup.py --confirm
```

A shell script version is also available with similar functionality:

```bash
./cleanup.sh             # Clean everything
./cleanup.sh --dry-run   # Perform a dry run
./cleanup.sh --models    # Only remove model files
```

### Full Pipeline Execution

To run the entire pipeline (training, evaluation, comparison):

```bash
python -m src.pipeline.train_evaluate_compare \
  --data_dir data/raw \
  --output_dir models \
  --report_dir reports \
  --visualize \
  --img_size 224 \
  --batch_size 32 \
  --epochs 30 \
  --patience 10
```

For Apple Silicon (M-series) optimized training:

```bash
python -m src.pipeline.train_evaluate_compare \
  --data_dir data/raw \
  --output_dir models \
  --report_dir reports \
  --visualize \
  --use_mps \
  --mps_graph \
  --mps_fallback \
  --memory_efficient \
  --cache_dataset \
  --pin_memory \
  --optimize_for_m_series \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 16 \
  --patience 30 \ 
  --epochs 30
```

python -m src.pipeline.train_evaluate_compare \
  --data_dir data/raw \
  --output_dir models \
  --report_dir reports \
  --epochs 1 \
  --batch_size 8 \
  --img_size 64 \
  --patience 30 \
  --visualize \
  --use_mps \
  --mps_graph \
  --mps_fallback \
  --memory_efficient \
  --cache_dataset \
  --pin_memory \
  --optimize_for_m_series

For a quick test run with minimal time:

```bash
python -m src.pipeline.train_evaluate_compare \
  --data_dir data/raw \
  --output_dir models \
  --report_dir reports \
  --epochs 1 \
  --batch_size 8 \
  --img_size 64 \
  --patience 1 \
  --visualize
```

For a quick test run with minimal time on Apple Silicon:

```bash
python -m src.pipeline.train_evaluate_compare \
  --data_dir data/raw \
  --output_dir models \
  --report_dir reports \
  --epochs 1 \
  --batch_size 8 \
  --img_size 64 \
  --patience 1 \
  --visualize \
  --use_mps \
  --mps_graph \
  --mps_fallback \
  --memory_efficient \
  --cache_dataset \
  --pin_memory \
  --optimize_for_m_series
```

```bash
### Model Evaluation

Evaluate a single model:

```bash
python -m src.training.evaluate \
  --model_path models/experiment1/best_model.pth \
  --data_dir data/raw \
  --output_dir reports/evaluations \
  --visualize
```

Batch evaluate multiple models:

```bash
python -m src.utils.batch_evaluate \
  --models_dir models \
  --data_dir data/raw \
  --output_dir reports/evaluations \
  --report_dir reports/comparisons \
  --visualize
```

### Visualization

The training process automatically generates visualizations in two locations:

1. During training: `models/[model_name]_v[version]/visualizations/`
2. During evaluation: `reports/evaluations/[model_name]_v[version]/visualizations/`

Key visualizations include:

- Confusion matrices
- Classification reports
- Example predictions
- Misclassified examples
- GradCAM heatmaps (in the `gradcam/` subdirectory) for model interpretability

The GradCAM visualizations are especially useful for understanding which parts of the image the model focuses on when making predictions.

You can also generate visualizations separately:

```bash
python -m src.utils.visualization \
  --model_path models/experiment1/best_model.pth \
  --data_dir data/processed/test \
  --output_dir reports/visualizations
```

## 🧠 Model Architectures

| Architecture | Variants | Special Features |
|--------------|----------|-----------------|
| EfficientNet | B0, B3 | Compound scaling, advanced classifier |
| MobileNet | V2, V3 (Small, Large) | Depthwise separable convolutions, attention mechanisms |
| ResNet | 18, 50 | Residual connections, attention mechanisms |

All models support:

- Transfer learning from ImageNet
- Selective freezing of backbone layers
- Custom classifier heads
- Attention mechanisms to highlight disease-specific features

## 🔬 Examples

### Training with EfficientNet-B3

```bash
python -m src.training.train \
  --data_dir data/raw \
  --output_dir models/efficientnet_b3_run1 \
  --model efficientnet_b3 \
  --img_size 300 \
  --batch_size 16 \
  --epochs 30 \
  --lr 0.001 \
  --pretrained \
  --freeze_backbone
```

### Fine-tuning a pre-trained ResNet with attention

```bash
python -m src.training.train \
  --data_dir data/raw \
  --output_dir models/resnet_attention_run1 \
  --model resnet_attention \
  --img_size 224 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0005 \
  --weight_decay 1e-5 \
  --pretrained
```

### Evaluating a trained model

```bash
python -m src.training.evaluate \
  --model_path models/efficientnet_b3_run1/best_model.pth \
  --data_dir data/processed/test \
  --batch_size 32 \
  --visualize
```

The evaluation script provides comprehensive model assessment with:

- Automatic model architecture detection
- Calculation of accuracy, precision, recall, and F1 score
- Confusion matrix generation
- GradCAM visualizations of model attention
- Identification of misclassified examples
- Export of results to CSV files

### Running the Model Evaluation Pipeline

```bash
python -m src.utils.batch_evaluate \
  --models_dir models \
  --data_dir data/raw \
  --output_dir reports/evaluations \
  --report_dir reports/comparisons \
  --use_registry \
  --report_id custom_evaluation_run \
  --visualize
```

## 🔧 Advanced Features

### Device Support

The code automatically selects the best available device (CUDA > MPS > CPU). You can force specific devices:

```bash
# Disable CUDA
python -m src.training.train --no_cuda ...

# Disable MPS (Apple Silicon)
python -m src.training.train --no_mps ...

# Explicitly enable MPS (Apple Silicon)
python -m src.training.train --use_mps ...
```

### Class Imbalance

The training script automatically handles class imbalance by calculating class weights:

```python
# Calculate class weights to handle imbalance
class_counts = np.array(list(class_info["class_counts"].values()))
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / np.sum(class_weights) * num_classes
```

### Experiment Tracking

Each run creates a comprehensive log with:

- Training and validation metrics
- Model architecture details
- Training curves
- Checkpoint information

Logs are stored in `reports/logs/[experiment_name]`.

### Model Registry

The project maintains a central model registry (`models/model_registry.json`) that tracks:

- Model version and architecture
- Performance metrics
- Creation date and paths
- Training parameters

The registry is automatically updated during training and helps with model versioning. Each new training run creates a new versioned directory (e.g., `efficientnet_b0_v1`, `efficientnet_b0_v2`) instead of overwriting previous results.

The pipeline displays the model registry at the end of each run, showing all registered models with their metrics.

## 🚢 Deployment

### Web Application

The project includes a modern Flask web application for model deployment:

```bash
# Start the web server with the launch script
python run_app.py --model models/best_model.pth --host 127.0.0.1 --port 5000
```

See the [Web Interface](#-web-interface) section below for more details.

### Mobile Deployment

For mobile deployment, the project provides utilities to convert PyTorch models to TorchScript or ONNX format:

```bash
python -m src.app.mobile.export \
  --model_path models/best_model.pth \
  --output_path models/mobile/model.pt \
  --format torchscript
```

## 📱 Web Interface

The project includes a comprehensive web interface for crop disease diagnosis, allowing users to upload images and receive immediate analysis results.

### Running the Web App

```bash
# Basic usage
python run_app.py --model models/best_model.pth

# Advanced options - specify model ID and make accessible from other devices
python run_app.py --model models/best_model.pth --model-id "EfficientNet-B0" --host 0.0.0.0
```

<a id="web-features"></a>

### Features

- **User-friendly Interface**: Intuitive drag-and-drop image upload
- **Real-time Analysis**: Immediate disease detection and confidence scores
- **Detailed Results**: Probability breakdown for all potential diseases
- **Disease Information**: Descriptions and treatment recommendations for identified diseases
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Visual Feedback**: Color-coded confidence levels and probabilities
- **Multi-model Support**: Switch between different models through dropdown menu
- **Diagnosis History**: View and manage past diagnoses through dedicated history page
- **Report Generation**: Create detailed PDF reports with diagnosis information
- **Crop Type Selection**: Specify crop type for more targeted diagnoses

### Screenshots

*[Screenshots of the web interface would be placed here]*

## 🔌 API Integration

The system provides a RESTful API for programmatic access to the crop disease detection functionality.

### API Endpoints

- **GET /api/models**: List all available models
- **POST /api/set-model/{model_id}**: Change the active model
- **GET /api/history**: Retrieve diagnosis history
- **POST /api/diagnose**: Submit an image for diagnosis

### Using the API Client

The project includes an example API client (`api_client_example.py`) that demonstrates how to interact with the API:

```bash
# List available models
python api_client_example.py --url http://localhost:5000 models

# Diagnose an image using the default model
python api_client_example.py --url http://localhost:5000 diagnose path/to/image.jpg

# Diagnose an image with a specific model and crop type
python api_client_example.py --url http://localhost:5000 diagnose path/to/image.jpg --model model_name.pth --crop rice

# View diagnosis history
python api_client_example.py --url http://localhost:5000 history
```

### API Documentation

Comprehensive API documentation is available in the `docs/API_Documentation.md` file, which includes:

- Detailed endpoint descriptions
- Request and response formats
- Example requests using curl
- Error handling information

## 📊 Evaluation Framework

The project includes a comprehensive model evaluation framework that assesses model performance and generates insightful visualizations.

<a id="evaluation-features"></a>

### Features

- **Automated Model Loading**: Automatically detects model architecture from the checkpoint
- **Multiple Evaluation Metrics**: Calculates accuracy, precision, recall, F1 score, and confusion matrices
- **Advanced Visualizations**:
  - Confusion matrices
  - Classification report heatmaps
  - Example predictions
  - Analysis of misclassifications
  - GradCAM visualizations showing regions of interest for model interpretability
- **Flexible Data Input**: Compatible with directory structures or CSV listings
- **Single Image Inference**: Dedicated function for individual image predictions
- **Hardware Optimization**: Automatic detection and utilization of available hardware (CUDA, MPS, CPU)

### Running Evaluations

Basic evaluation:

```bash
python -m src.training.evaluate \
  --model_path models/best_model.pth \
  --data_dir data/processed/test
```

With visualizations:

```bash
python -m src.training.evaluate \
  --model_path models/best_model.pth \
  --data_dir data/processed/test \
  --output_dir reports/evaluations \
  --visualize
```

Hardware-specific options:

```bash
# Force CPU usage
python -m src.training.evaluate --model_path models/best_model.pth --data_dir data/test --no_cuda --no_mps

# Use Apple Silicon GPU
python -m src.training.evaluate --model_path models/best_model.pth --data_dir data/test --no_cuda
```

### Batch Evaluation

The project supports batch evaluation of multiple models with enhanced comparison features:

```bash
python -m src.utils.batch_evaluate \
  --models_dir models \
  --data_dir data/raw \
  --output_dir reports/evaluations \
  --report_dir reports/comparisons \
  --visualize
```

New Options for Batch Evaluation:

- `--use_registry`: Use the model registry for finding models
- `--registry_path`: Custom path to the model registry file
- `--models_pattern`: Pattern to match versioned model files (default: `**/*_v*_*.pth`)
- `--legacy_pattern`: Pattern to match legacy model files (default: `**/best_model.pth`)
- `--report_id`: Custom ID for the report (defaults to timestamp)

### Model Comparison Reports

The batch evaluation generates comprehensive comparison reports that:

- Compare performance metrics across all models
- Identify the best model for each metric
- Visualize performance differences
- Provide detailed analysis of the best-performing model
- Include interactive HTML reports with visualizations

## 🧩 Extending the Project

### Adding New Models

1. Create a new file in the `src/models` directory
2. Implement model creation functions that follow this template:

```python
def create_your_model(num_classes, pretrained=True, freeze_backbone=True):
    # Initialize model
    # Modify as needed
    # Return model
    pass
```

3. Add the model to the model selection logic in `src/models/model_factory.py`

### Custom Datasets

Modify the `create_dataset_from_directory` function in `src/data/dataset.py` to handle your specific dataset format. For specialized data loading needs, inherit from the `CropDiseaseDataset` class.

## 📈 Experimental Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| EfficientNet-B0 | 94.3% | 0.942 | 0.945 | 0.939 |
| MobileNetV3-Large | 93.1% | 0.930 | 0.937 | 0.925 |
| ResNet50+Attention | 95.7% | 0.956 | 0.958 | 0.954 |

*For full benchmark results, see the [reports/benchmarks](reports/benchmarks) directory.*

## 📚 Citation

If you use this code in your research, please cite:

```
@software{crop_disease_detection,
  author = {Jeremy Cleland},
  title = {{Crop Disease Detection: A Deep Learning Approach}},
  year = {2025},
  url = {https://github.com/jeremy-cleland/crop-disease-detection},
  version = {1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The attention mechanisms are inspired by "Residual Attention Network for Image Classification" (Wang et al., 2017)
- EfficientNet implementations based on "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (Tan & Le, 2019)
- The GradCAM visualization is based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (Selvaraju et al., 2017)
- Pre-trained models provided by the PyTorch torchvision library
- Dataset processing utilities adapted from PyTorch's ImageFolder
