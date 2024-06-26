#!/bin/bash

# Create the model directory
mkdir -p /app/model

# Define the URL prefix for the model
MODEL_URL="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02/resolve/main/"

# Download the model files
wget -P /app/model ${MODEL_URL}model-00001-of-00003.safetensors
wget -P /app/model ${MODEL_URL}model-00002-of-00003.safetensors
wget -P /app/model ${MODEL_URL}model-00003-of-00003.safetensors
wget -P /app/model ${MODEL_URL}model.safetensors.index.json

# Start the application
exec python3 main.py
