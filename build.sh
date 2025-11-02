#!/usr/bin/env bash
# build.sh - Render build script

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p models

# Set TensorFlow to use less memory
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Build completed successfully!"
