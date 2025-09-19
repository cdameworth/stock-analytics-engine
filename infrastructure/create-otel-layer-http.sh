#!/bin/bash

# Script to create HTTP-only OpenTelemetry Lambda layer
# This avoids gRPC cygrpc compatibility issues

set -e

echo "Creating HTTP-only OpenTelemetry Lambda layer..."

# Create temporary layer directory
LAYER_DIR="/tmp/otel-python-http-layer"
rm -rf "$LAYER_DIR"
mkdir -p "$LAYER_DIR/python"

# Install OpenTelemetry packages into the layer (HTTP only)
echo "Installing OpenTelemetry HTTP packages..."
pip3 install --target "$LAYER_DIR/python" -r infrastructure/otel-layer-requirements.txt

# Create the layer zip file
echo "Creating layer zip file..."
cd "$LAYER_DIR"
zip -r otel-python-http-layer.zip python/

# Upload the layer to AWS Lambda
echo "Uploading HTTP layer to AWS Lambda..."
aws lambda publish-layer-version \
    --layer-name "stock-analytics-otel-python-http" \
    --description "HTTP-only OpenTelemetry Python instrumentation for Stock Analytics" \
    --zip-file "fileb://otel-python-http-layer.zip" \
    --compatible-runtimes python3.11 \
    --profile stock-analytics-admin

echo "HTTP OTEL layer created successfully!"

# Cleanup
rm -rf "$LAYER_DIR"

echo "Layer ARN will be displayed above. Use this ARN to update Lambda functions."