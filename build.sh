#!/usr/bin/env bash
# build.sh - Render build script

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p models

echo "Build completed successfully!"
