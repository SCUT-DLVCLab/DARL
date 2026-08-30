#!/bin/bash

# DARL Training Script
# Launches distributed training with DeepSpeed and Accelerate

echo "============================================"
echo "DARL Training - Starting..."
echo "============================================"

# Check if config file exists
if [ ! -f "configs/deep_config.yaml" ]; then
    echo "Error: configs/deep_config.yaml not found!"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Display GPU information
echo ""
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# Set environment variables
export WANDB_PROJECT="darl_training"
export WANDB_MODE="offline"  # Change to "online" for cloud logging

# Launch training
echo "Launching training with accelerate..."
echo ""

accelerate launch \
    --config_file configs/deep_config.yaml \
    train.py \
    "$@"

echo ""
echo "============================================"
echo "Training completed or terminated"
echo "============================================"
