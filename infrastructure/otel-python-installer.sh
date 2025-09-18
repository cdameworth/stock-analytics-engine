#!/bin/bash
# OpenTelemetry Python Layer Build Script for AWS Lambda
# This script creates a Lambda layer with OpenTelemetry instrumentation

set -e

echo "Building OpenTelemetry Python Layer for AWS Lambda..."

# Create layer directory structure
mkdir -p python/lib/python3.11/site-packages

# Install dependencies
pip install -r requirements.txt -t python/lib/python3.11/site-packages/

# Remove unnecessary files to reduce layer size
find python/ -type d -name "__pycache__" -exec rm -rf {} + || true
find python/ -type f -name "*.pyc" -delete || true
find python/ -type f -name "*.pyo" -delete || true
find python/ -name "*.dist-info" -exec rm -rf {} + || true

# Create bootstrap script for OpenTelemetry auto-instrumentation
cat > python/otel-instrument << 'EOF'
#!/opt/python/bin/python3.11
import os
import sys

# Add the layer path to Python path
sys.path.insert(0, '/opt/python/lib/python3.11/site-packages')

# Import and configure OpenTelemetry
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize

# Set up environment for auto-instrumentation
os.environ.setdefault('OTEL_PYTHON_DISTRO', 'aws_distro')
os.environ.setdefault('OTEL_PYTHON_CONFIGURATOR', 'aws_lambda_configurator')

# Execute the original Lambda handler
import runpy
sys.argv[0] = sys.argv[1]
sys.argv[1:] = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
EOF

chmod +x python/otel-instrument

echo "OpenTelemetry Python layer build completed."