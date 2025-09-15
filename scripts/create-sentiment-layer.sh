#!/bin/bash
# Create Lambda layer for sentiment analysis dependencies

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

create_layer() {
    print_status "Creating Lambda layer for sentiment analysis dependencies..."

    # Create temporary directory for layer building
    TEMP_DIR=$(mktemp -d)
    LAYER_DIR="$TEMP_DIR/python"

    print_status "Using temporary directory: $TEMP_DIR"

    # Create python directory for Lambda layer structure
    mkdir -p "$LAYER_DIR"

    # Install dependencies using pip
    print_status "Installing Python dependencies..."
    pip3 install -r lambda_requirements/requirements_sentiment.txt -t "$LAYER_DIR" --no-deps

    # Install core dependencies manually to ensure compatibility
    print_status "Installing core dependencies with proper versions..."
    pip3 install requests==2.31.0 -t "$LAYER_DIR"
    pip3 install urllib3==1.26.18 -t "$LAYER_DIR"  # Compatible with requests
    pip3 install charset-normalizer==3.3.2 -t "$LAYER_DIR"
    pip3 install idna==3.4 -t "$LAYER_DIR"
    pip3 install certifi==2023.11.17 -t "$LAYER_DIR"

    # Create deployment package
    print_status "Creating deployment package..."
    cd "$TEMP_DIR"
    zip -r "../sentiment-layer.zip" python/ -q

    # Move to project root
    mv "../sentiment-layer.zip" "$OLDPWD/"

    print_success "Layer package created: sentiment-layer.zip"

    # Clean up
    cd "$OLDPWD"
    rm -rf "$TEMP_DIR"
}

deploy_layer() {
    print_status "Deploying Lambda layer to AWS..."

    # Delete existing layer versions if any (keep latest)
    print_status "Checking for existing layer versions..."

    # Create or update the layer
    LAYER_VERSION=$(aws lambda publish-layer-version \
        --layer-name "$LAYER_NAME" \
        --description "Dependencies for sentiment analysis (requests, boto3, pandas, numpy)" \
        --zip-file fileb://sentiment-layer.zip \
        --compatible-runtimes python3.11 python3.10 python3.9 \
        --region "$REGION" \
        --profile "$AWS_PROFILE" \
        --query 'Version' \
        --output text)

    if [[ $? -eq 0 ]]; then
        print_success "Layer version $LAYER_VERSION created successfully"

        # Get layer ARN
        LAYER_ARN=$(aws lambda get-layer-version \
            --layer-name "$LAYER_NAME" \
            --version-number "$LAYER_VERSION" \
            --region "$REGION" \
            --profile "$AWS_PROFILE" \
            --query 'LayerArn' \
            --output text)

        echo ""
        echo "==============================================="
        echo "🎉 LAMBDA LAYER DEPLOYMENT COMPLETE"
        echo "==============================================="
        echo "Layer Name: $LAYER_NAME"
        echo "Layer Version: $LAYER_VERSION"
        echo "Layer ARN: $LAYER_ARN"
        echo "Compatible Runtimes: python3.9, python3.10, python3.11"
        echo "==============================================="
        echo ""

        # Save layer ARN for later use
        echo "$LAYER_ARN" > layer-arn.txt
        print_success "Layer ARN saved to layer-arn.txt"

    else
        print_error "Failed to create Lambda layer"
        return 1
    fi
}

update_lambda_functions() {
    print_status "Updating Lambda functions to use the new layer..."

    LAYER_ARN=$(cat layer-arn.txt)

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
    print_status "Testing Lambda functions with dependencies..."

    # Test sentiment analyzer
    print_status "Testing news-sentiment-analyzer..."
    aws lambda invoke \
        --function-name news-sentiment-analyzer \
        --cli-binary-format raw-in-base64-out \
        --payload '{"symbol": "AAPL", "lookback_hours": 24}' \
        /tmp/sentiment-layer-test.json \
        --region "$REGION" \
        --profile "$AWS_PROFILE" > /dev/null

    if grep -q "errorMessage" /tmp/sentiment-layer-test.json; then
        print_warning "Sentiment analyzer test showed errors:"
        cat /tmp/sentiment-layer-test.json
    else
        print_success "Sentiment analyzer test completed successfully"
        RESPONSE_SIZE=$(cat /tmp/sentiment-layer-test.json | jq -r '.body' | wc -c)
        print_status "Response size: $RESPONSE_SIZE characters"
    fi

    # Clean up test file
    rm -f /tmp/sentiment-layer-test.json
}

# Main execution
main() {
    echo "================================================"
    echo "🔧 Lambda Layer Creation for Sentiment Analysis"
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

    # Verify AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        print_error "AWS credentials not configured for profile: $AWS_PROFILE"
        exit 1
    fi

    print_success "Prerequisites check passed"

    # Create and deploy layer
    create_layer
    deploy_layer
    update_lambda_functions
    test_functions

    # Clean up
    rm -f sentiment-layer.zip

    print_success "Lambda layer deployment completed successfully!"
    echo ""
    echo "🎯 Next Steps:"
    echo "  • Test sentiment analysis with real API keys"
    echo "  • Monitor CloudWatch logs for any remaining issues"
    echo "  • Consider adding NewsAPI and Finnhub API keys to SSM"
}

# Run main function
main "$@"