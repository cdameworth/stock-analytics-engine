#!/bin/bash
# Create Lambda layer for sentiment analysis dependencies using Docker for correct architecture

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-stock-analytics-admin}"
REGION="us-east-1"
LAYER_NAME="sentiment-analysis-dependencies"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_layer_with_docker() {
    print_status "Creating Lambda layer using Docker for x86_64 architecture..."

    # Create Dockerfile for building dependencies
    cat > Dockerfile.lambda-layer << 'EOF'
FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY lambda_requirements/requirements_sentiment.txt /tmp/
RUN pip install -r /tmp/requirements_sentiment.txt -t /opt/python/

# Create the layer structure
CMD ["echo", "Layer dependencies installed"]
EOF

    # Create docker-compose for easier management
    cat > docker-compose.lambda-layer.yml << 'EOF'
version: '3.8'
services:
  lambda-layer:
    build:
      context: .
      dockerfile: Dockerfile.lambda-layer
    volumes:
      - ./layer-output:/output
    command: cp -r /opt/python /output/
EOF

    # Create output directory
    mkdir -p layer-output

    print_status "Building dependencies with Docker..."
    docker build -f Dockerfile.lambda-layer -t lambda-layer-builder .

    print_status "Extracting dependencies..."
    docker run --rm -v $(pwd)/layer-output:/output lambda-layer-builder cp -r /opt/python /output/

    # Create the layer zip
    print_status "Creating layer zip file..."
    cd layer-output
    zip -r ../sentiment-layer-docker.zip python/ -q
    cd ..

    print_success "Layer package created: sentiment-layer-docker.zip"

    # Clean up Docker resources
    docker rmi lambda-layer-builder 2>/dev/null || true
    rm -rf layer-output Dockerfile.lambda-layer docker-compose.lambda-layer.yml
}

create_layer_with_pip_target() {
    print_status "Creating Lambda layer using pip with platform targeting..."

    # Create temporary directory for layer building
    TEMP_DIR=$(mktemp -d)
    LAYER_DIR="$TEMP_DIR/python"

    print_status "Using temporary directory: $TEMP_DIR"

    # Create python directory for Lambda layer structure
    mkdir -p "$LAYER_DIR"

    # Install dependencies with platform targeting for Linux x86_64
    print_status "Installing Python dependencies for Linux x86_64..."
    pip3 install \
        --platform linux_x86_64 \
        --target "$LAYER_DIR" \
        --implementation cp \
        --python-version 3.11 \
        --only-binary=:all: \
        --upgrade \
        requests boto3 pandas numpy

    # Create deployment package
    print_status "Creating deployment package..."
    cd "$TEMP_DIR"
    zip -r "../sentiment-layer-targeted.zip" python/ -q

    # Move to project root
    mv "../sentiment-layer-targeted.zip" "$OLDPWD/"

    print_success "Layer package created: sentiment-layer-targeted.zip"

    # Clean up
    cd "$OLDPWD"
    rm -rf "$TEMP_DIR"
}

deploy_layer() {
    local layer_file="$1"
    print_status "Deploying Lambda layer to AWS using $layer_file..."

    # Delete existing layer version if needed
    EXISTING_VERSION=$(aws lambda list-layer-versions \
        --layer-name "$LAYER_NAME" \
        --region "$REGION" \
        --profile "$AWS_PROFILE" \
        --query 'LayerVersions[0].Version' \
        --output text 2>/dev/null || echo "None")

    if [[ "$EXISTING_VERSION" != "None" ]]; then
        print_status "Found existing layer version $EXISTING_VERSION"
    fi

    # Create or update the layer
    LAYER_VERSION=$(aws lambda publish-layer-version \
        --layer-name "$LAYER_NAME" \
        --description "Dependencies for sentiment analysis (requests, boto3, pandas, numpy) - x86_64" \
        --zip-file "fileb://$layer_file" \
        --compatible-runtimes python3.11 python3.10 python3.9 \
        --compatible-architectures x86_64 \
        --region "$REGION" \
        --profile "$AWS_PROFILE" \
        --query 'Version' \
        --output text)

    if [[ $? -eq 0 ]]; then
        print_success "Layer version $LAYER_VERSION created successfully"

        # Get layer ARN with version
        LAYER_ARN="arn:aws:lambda:$REGION:$(aws sts get-caller-identity --query Account --output text --profile $AWS_PROFILE):layer:$LAYER_NAME:$LAYER_VERSION"

        echo ""
        echo "==============================================="
        echo "🎉 LAMBDA LAYER DEPLOYMENT COMPLETE"
        echo "==============================================="
        echo "Layer Name: $LAYER_NAME"
        echo "Layer Version: $LAYER_VERSION"
        echo "Layer ARN: $LAYER_ARN"
        echo "Compatible Runtimes: python3.9, python3.10, python3.11"
        echo "Architecture: x86_64"
        echo "==============================================="
        echo ""

        # Save layer ARN for later use
        echo "$LAYER_ARN" > layer-arn-fixed.txt
        print_success "Layer ARN saved to layer-arn-fixed.txt"

        return 0
    else
        print_error "Failed to create Lambda layer"
        return 1
    fi
}

update_lambda_functions() {
    print_status "Updating Lambda functions to use the new layer..."

    LAYER_ARN=$(cat layer-arn-fixed.txt)

    if [[ -z "$LAYER_ARN" ]]; then
        print_error "Layer ARN not found. Deploy layer first."
        return 1
    fi

    # Update news-sentiment-analyzer function
    print_status "Updating news-sentiment-analyzer function..."
    aws lambda update-function-configuration \
        --function-name news-sentiment-analyzer \
        --layers "$LAYER_ARN" \
        --region "$REGION" \
        --profile "$AWS_PROFILE" > /dev/null

    if [[ $? -eq 0 ]]; then
        print_success "news-sentiment-analyzer updated with layer"
    else
        print_error "Failed to update news-sentiment-analyzer"
    fi

    # Update enhanced-feature-extractor function
    print_status "Updating enhanced-feature-extractor function..."
    aws lambda update-function-configuration \
        --function-name enhanced-feature-extractor \
        --layers "$LAYER_ARN" \
        --region "$REGION" \
        --profile "$AWS_PROFILE" > /dev/null

    if [[ $? -eq 0 ]]; then
        print_success "enhanced-feature-extractor updated with layer"
    else
        print_error "Failed to update enhanced-feature-extractor"
    fi
}

test_functions() {
    print_status "Testing Lambda functions with x86_64 dependencies..."

    # Test sentiment analyzer
    print_status "Testing news-sentiment-analyzer..."
    aws lambda invoke \
        --function-name news-sentiment-analyzer \
        --cli-binary-format raw-in-base64-out \
        --payload '{"symbol": "AAPL", "lookback_hours": 24}' \
        /tmp/sentiment-arch-test.json \
        --region "$REGION" \
        --profile "$AWS_PROFILE" > /dev/null

    if grep -q "errorMessage" /tmp/sentiment-arch-test.json; then
        print_warning "Sentiment analyzer test showed errors:"
        cat /tmp/sentiment-arch-test.json
    else
        print_success "Sentiment analyzer test completed successfully"
        SENTIMENT_SCORE=$(cat /tmp/sentiment-arch-test.json | jq -r '.body.sentiment_metrics.overall_sentiment // "N/A"')
        print_status "Sentiment score for AAPL: $SENTIMENT_SCORE"
    fi

    # Test enhanced feature extractor
    print_status "Testing enhanced-feature-extractor..."
    aws lambda invoke \
        --function-name enhanced-feature-extractor \
        --cli-binary-format raw-in-base64-out \
        --payload '{"symbol": "AAPL", "current_data": {"close": 175.23, "volume": 50000000, "moving_avg_5": 172.45, "moving_avg_20": 170.12, "volatility": 0.25}}' \
        /tmp/feature-arch-test.json \
        --region "$REGION" \
        --profile "$AWS_PROFILE" > /dev/null

    if grep -q "errorMessage" /tmp/feature-arch-test.json; then
        print_warning "Enhanced feature extractor test showed errors:"
        cat /tmp/feature-arch-test.json
    else
        print_success "Enhanced feature extractor test completed successfully"
        FEATURE_COUNT=$(cat /tmp/feature-arch-test.json | jq -r '.feature_count // "N/A"')
        print_status "Feature count for AAPL: $FEATURE_COUNT"
    fi

    # Clean up test files
    rm -f /tmp/sentiment-arch-test.json /tmp/feature-arch-test.json
}

# Main execution
main() {
    echo "================================================"
    echo "🔧 Lambda Layer (x86_64) for Sentiment Analysis"
    echo "================================================"
    echo ""

    # Check prerequisites
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 not found. Please install pip3 first."
        exit 1
    fi

    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install AWS CLI first."
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        print_error "jq not found. Please install jq for JSON processing."
        exit 1
    fi

    # Check if Docker is available
    if command -v docker &> /dev/null; then
        print_status "Docker found - using Docker for cross-platform builds"
        USE_DOCKER=true
    else
        print_warning "Docker not found - using pip platform targeting"
        USE_DOCKER=false
    fi

    # Verify AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        print_error "AWS credentials not configured for profile: $AWS_PROFILE"
        exit 1
    fi

    print_success "Prerequisites check passed"

    # Create layer using appropriate method
    if [[ "$USE_DOCKER" == "true" ]]; then
        create_layer_with_docker
        LAYER_FILE="sentiment-layer-docker.zip"
    else
        create_layer_with_pip_target
        LAYER_FILE="sentiment-layer-targeted.zip"
    fi

    # Deploy and test
    if deploy_layer "$LAYER_FILE"; then
        update_lambda_functions
        test_functions

        # Clean up
        rm -f "$LAYER_FILE"

        print_success "Lambda layer with robust pandas/numpy support deployed successfully!"
        echo ""
        echo "🎯 Benefits of pandas/numpy:"
        echo "  • Robust data manipulation and analysis"
        echo "  • Efficient numerical computations"
        echo "  • Better sentiment analysis with statistical methods"
        echo "  • Enhanced feature engineering capabilities"
        echo "  • Professional-grade time series analysis"
    else
        print_error "Layer deployment failed"
        exit 1
    fi
}

# Run main function
main "$@"