#!/bin/bash

echo "Creating minimal ML layer for dual prediction system..."

# Create layer directory structure
mkdir -p layer_dual_prediction/python

# Create requirements.txt with minimal dependencies
cat > requirements_dual.txt << EOF
yfinance==0.2.18
pandas==2.0.3
numpy==1.24.3
requests==2.31.0
EOF

# Install dependencies
pip install -r requirements_dual.txt -t layer_dual_prediction/python/

# Create layer package
cd layer_dual_prediction
zip -r ../dual_prediction_layer.zip .
cd ..

echo "Dual prediction layer created: dual_prediction_layer.zip"
echo "Size: $(du -h dual_prediction_layer.zip | cut -f1)"

# Deploy layer to AWS
aws lambda publish-layer-version \
    --layer-name dual-prediction-ml-layer \
    --description "ML dependencies for dual prediction system (yfinance, pandas, numpy)" \
    --zip-file fileb://dual_prediction_layer.zip \
    --compatible-runtimes python3.11 \
    --profile stock-analytics-admin

echo "Layer deployed successfully!"