#!/bin/bash

# Script to train all model architectures for crop disease classification in parallel
# Author: Jeremy
# Date: 2025-03-01

# Set up error handling
set -e  # Exit on any error

# Create logs directory for training outputs
mkdir -p logs

# Print start message
echo "Starting parallel model training sequence at $(date)"
echo "=============================================="

# Function to train a model and log output
train_model() {
    model_name=$1
    output_dir=$2
    model_type=$3
    experiment_name=$4
    extra_args=${5:-""}  # Optional extra arguments

    echo "Starting training for $model_name at $(date)" > "logs/${experiment_name}.log"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Run the training command and log output
    (
        echo ""
        echo "Training $model_name..."
        echo "Started at $(date)"
        
        PYTHONPATH=. python src/training/train.py \
            --data_dir data/raw \
            --output_dir "$output_dir" \
            --model "$model_type" \
            --use_weights \
            --patience 7 \
            --find_lr \
            --experiment_name "$experiment_name" \
            $extra_args >> "logs/${experiment_name}.log" 2>&1
        
        # Record completion status
        if [ $? -eq 0 ]; then
            echo "$model_name training completed successfully at $(date)" | tee -a "logs/${experiment_name}.log"
            echo "$model_name: SUCCESS" >> logs/training_summary.txt
        else
            echo "ERROR: $model_name training failed at $(date)" | tee -a "logs/${experiment_name}.log"
            echo "$model_name: FAILED" >> logs/training_summary.txt
        fi
    ) &  # Run in background
    
    # Store the process ID
    echo "Started $model_name (PID: $!)"
}

# Clear the summary file
echo "Training Summary ($(date))" > logs/training_summary.txt
echo "------------------------------" >> logs/training_summary.txt

# Train models in parallel
# Adjust the grouping based on your Mac's resources (groups of 2-3 is usually good)

# Group 1
train_model "EfficientNet" "models/efficientnet_v1" "efficientnet" "efficientnet_run_v1"
train_model "MobileNet" "models/mobilenet_v1" "mobilenet" "mobilenet_run_v1"

# Wait for Group 1 to finish before starting Group 2
# This prevents overloading your system
wait

# Group 2
train_model "EfficientNet-B3" "models/efficientnet_b3_v1" "efficientnet_b3" "efficientnet_b3_run_v1"
train_model "MobileNetV3 Small" "models/mobilenet_v3_small_v1" "mobilenet_v3_small" "mobilenet_v3_small_run_v1"

wait

# Group 3
train_model "MobileNetV3 Large" "models/mobilenet_v3_large_v1" "mobilenet_v3_large" "mobilenet_v3_large_run_v1"
train_model "MobileNet with Attention" "models/mobilenet_attention_v1" "mobilenet_attention" "mobilenet_attention_run_v1"

wait

# Group 4
train_model "ResNet18" "models/resnet18_v1" "resnet18" "resnet18_run_v1"
train_model "ResNet50" "models/resnet50_v1" "resnet50" "resnet50_run_v1"
train_model "ResNet with Attention" "models/resnet_attention_v1" "resnet_attention" "resnet_attention_run_v1" "--resnet_version 50"

# Wait for all remaining jobs to complete
wait

# Print completion message
echo ""
echo "=============================================="
echo "All model training jobs completed at $(date)"
echo "Check logs directory for training outputs"
echo "Models are saved in their respective directories"
echo ""
echo "Training Summary:"
cat logs/training_summary.txt