#!/bin/bash

# Script to create OpenTelemetry Lambda layer with proper Python packages
# This resolves the "No module named 'opentelemetry'" errors

set -e

echo "Creating OpenTelemetry Lambda layer..."

# Create temporary layer directory
LAYER_DIR="/tmp/otel-python-layer"
rm -rf "$LAYER_DIR"
mkdir -p "$LAYER_DIR/python"

# Install OpenTelemetry packages into the layer
echo "Installing OpenTelemetry packages..."
pip3 install --target "$LAYER_DIR/python" -r infrastructure/otel-layer-requirements.txt

# Create the layer zip file
echo "Creating layer zip file..."
cd "$LAYER_DIR"
zip -r otel-python-layer.zip python/

# Upload the layer to AWS Lambda
echo "Uploading layer to AWS Lambda..."
aws lambda publish-layer-version \
    --layer-name "stock-analytics-otel-python-complete" \
    --description "Complete OpenTelemetry Python instrumentation for Stock Analytics" \
    --zip-file "fileb://otel-python-layer.zip" \
    --compatible-runtimes python3.11 \
    --profile stock-analytics-admin

echo "OTEL layer created successfully!"

# Cleanup
rm -rf "$LAYER_DIR"

echo "Layer ARN will be displayed above. Use this ARN to update Lambda functions."