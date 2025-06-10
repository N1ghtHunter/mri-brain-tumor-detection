#!/bin/bash

# Brain Tumor Analysis API Startup Script

echo "Starting Brain Tumor Analysis API..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/Scripts/activate

# Install/upgrade requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if models exist
if [ ! -f "yolov8_model.pt" ]; then
    echo "Error: yolov8_model.pt not found! Please ensure the YOLO model is available."
    exit 1
fi

echo "Starting Flask server..."
python app.py
