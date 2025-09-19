#!/bin/bash

# Deploy SigNoz Integration for Stock Analytics Engine
# This script implements the critical fixes from the validation report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AWS_PROFILE="stock-analytics-admin"
TERRAFORM_DIR="../infrastructure"
SIGNOZ_INGESTION_KEY="${TF_VAR_signoz_ingestion_key:-}"

echo -e "${BLUE}🚀 Starting SigNoz Integration Deployment${NC}"
echo "==========================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Validate prerequisites
validate_prerequisites() {
    echo -e "${BLUE}📋 Validating Prerequisites...${NC}"

    # Check AWS profile
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
        print_error "AWS profile '$AWS_PROFILE' not configured or invalid"
        exit 1
    fi
    print_status "AWS profile validated"

    # Check SigNoz ingestion key
    if [ -z "$SIGNOZ_INGESTION_KEY" ]; then
        print_warning "SigNoz ingestion key not set in TF_VAR_signoz_ingestion_key"
        echo "Please set it before deployment:"
        echo "export TF_VAR_signoz_ingestion_key=\"your-signoz-key\""
        exit 1
    fi
    print_status "SigNoz ingestion key configured"

    # Check Terraform
    if ! command -v terraform >/dev/null 2>&1; then
        print_error "Terraform not installed"
        exit 1
    fi
    print_status "Terraform validated"

    # Check directory structure
    if [ ! -d "$TERRAFORM_DIR" ]; then
        print_error "Terraform directory not found: $TERRAFORM_DIR"
        exit 1
    fi
    print_status "Directory structure validated"
}

# Phase 1: Deploy SigNoz-compatible OTEL layer
deploy_otel_layer() {
    echo -e "${BLUE}📦 Phase 1: Deploying OpenTelemetry Layer...${NC}"

    cd "$TERRAFORM_DIR"

    # Create the layer using updated requirements
    if [ -f "create-otel-layer-http.sh" ]; then
        print_status "Building SigNoz-compatible OTEL layer..."
        chmod +x create-otel-layer-http.sh
        ./create-otel-layer-http.sh

        # Capture the layer ARN
        LAYER_ARN=$(aws lambda list-layer-versions \
            --layer-name stock-analytics-otel-python-http \
            --profile "$AWS_PROFILE" \
            --query 'LayerVersions[0].LayerVersionArn' \
            --output text)

        if [ "$LAYER_ARN" != "None" ] && [ -n "$LAYER_ARN" ]; then
            print_status "OTEL layer deployed: $LAYER_ARN"
            export TF_VAR_custom_otel_layer_arn="$LAYER_ARN"
        else
            print_error "Failed to deploy OTEL layer"
            exit 1
        fi
    else
        print_error "OTEL layer creation script not found"
        exit 1
    fi
}

# Phase 2: Initialize and validate Terraform
prepare_terraform() {
    echo -e "${BLUE}🔧 Phase 2: Preparing Terraform...${NC}"

    cd "$TERRAFORM_DIR"

    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init

    # Validate configuration
    print_status "Validating Terraform configuration..."
    terraform validate

    # Plan deployment with SigNoz integration enabled
    print_status "Planning SigNoz integration deployment..."
    terraform plan \
        -var-file="terraform-tier1.tfvars" \
        -var="enable_signoz_integration=true" \
        -var="signoz_ingestion_key=$SIGNOZ_INGESTION_KEY" \
        -out="signoz-integration.tfplan"

    print_status "Terraform preparation complete"
}

# Phase 3: Deploy infrastructure with SigNoz integration
deploy_infrastructure() {
    echo -e "${BLUE}🏗️  Phase 3: Deploying Infrastructure...${NC}"

    cd "$TERRAFORM_DIR"

    # Apply the plan
    print_status "Applying SigNoz integration to infrastructure..."
    terraform apply "signoz-integration.tfplan"

    # Get outputs
    print_status "Retrieving deployment outputs..."
    terraform output otel_configuration

    print_status "Infrastructure deployment complete"
}

# Phase 4: Validate deployment
validate_deployment() {
    echo -e "${BLUE}🔍 Phase 4: Validating Deployment...${NC}"

    # Check Lambda functions have OTEL layers
    print_status "Validating Lambda function configurations..."

    FUNCTIONS=("stock-data-ingestion" "stock-recommendations-api" "ml-model-inference-lowcost" "dual-accuracy-tracker")

    for func in "${FUNCTIONS[@]}"; do
        echo "Checking $func..."

        # Check if function exists and has layers
        LAYERS=$(aws lambda get-function \
            --function-name "$func" \
            --profile "$AWS_PROFILE" \
            --query 'Configuration.Layers[].Arn' \
            --output text 2>/dev/null || echo "NOT_FOUND")

        if [[ "$LAYERS" == *"opentelemetry"* ]]; then
            print_status "$func: OTEL layer configured"
        else
            print_warning "$func: OTEL layer missing or function not found"
        fi

        # Check environment variables
        ENV_VARS=$(aws lambda get-function-configuration \
            --function-name "$func" \
            --profile "$AWS_PROFILE" \
            --query 'Environment.Variables.AWS_LAMBDA_EXEC_WRAPPER' \
            --output text 2>/dev/null || echo "NOT_FOUND")

        if [ "$ENV_VARS" = "/opt/otel-instrument" ]; then
            print_status "$func: OTEL environment configured"
        else
            print_warning "$func: OTEL environment missing"
        fi
    done
}

# Phase 5: Test trace generation
test_trace_generation() {
    echo -e "${BLUE}🧪 Phase 5: Testing Trace Generation...${NC}"

    # Test stock data ingestion function
    print_status "Testing trace generation with stock-data-ingestion..."

    aws lambda invoke \
        --function-name stock-data-ingestion \
        --payload '{"symbols": ["AAPL"], "test_mode": true}' \
        --profile "$AWS_PROFILE" \
        /tmp/signoz_test_response.json >/dev/null

    RESPONSE_STATUS=$(cat /tmp/signoz_test_response.json | grep -o '"statusCode": *[0-9]*' | grep -o '[0-9]*' || echo "500")

    if [ "$RESPONSE_STATUS" = "200" ]; then
        print_status "Lambda function executed successfully"
        print_status "Check SigNoz dashboard for traces within 1-2 minutes"
        print_status "Expected service name: stock-data-ingestion"
    else
        print_warning "Lambda function execution may have issues"
        echo "Response: $(cat /tmp/signoz_test_response.json)"
    fi

    # Clean up test file
    rm -f /tmp/signoz_test_response.json
}

# Phase 6: Generate validation report
generate_validation_report() {
    echo -e "${BLUE}📊 Phase 6: Generating Validation Report...${NC}"

    cat > /tmp/signoz_deployment_report.md << EOF
# SigNoz Integration Deployment Report

**Deployment Date**: $(date)
**AWS Profile**: $AWS_PROFILE
**SigNoz Endpoint**: ingest.us.signoz.cloud:443

## Infrastructure Status

### OpenTelemetry Layer
- **Layer Name**: stock-analytics-otel-python-http
- **Status**: Deployed
- **Dependencies**: SigNoz-compatible versions (distro==0.43b0, exporter==1.22.0)

### Lambda Functions Updated
EOF

    # Add function status to report
    FUNCTIONS=("stock-data-ingestion" "stock-recommendations-api" "ml-model-inference-lowcost" "dual-accuracy-tracker")

    for func in "${FUNCTIONS[@]}"; do
        LAYERS=$(aws lambda get-function \
            --function-name "$func" \
            --profile "$AWS_PROFILE" \
            --query 'Configuration.Layers[].Arn' \
            --output text 2>/dev/null || echo "NOT_FOUND")

        if [[ "$LAYERS" == *"opentelemetry"* ]]; then
            echo "- **$func**: ✅ OTEL layer configured" >> /tmp/signoz_deployment_report.md
        else
            echo "- **$func**: ❌ OTEL layer missing" >> /tmp/signoz_deployment_report.md
        fi
    done

    cat >> /tmp/signoz_deployment_report.md << EOF

## Environment Variables Configured
- AWS_LAMBDA_EXEC_WRAPPER=/opt/otel-instrument
- OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.us.signoz.cloud:443
- OTEL_PROPAGATORS=tracecontext,baggage,xray
- ENABLE_BUSINESS_TRACING=true
- PERFORMANCE_MONITORING=true

## Next Steps
1. Monitor SigNoz dashboard for incoming traces
2. Import Week 3 custom dashboards
3. Configure alerting rules
4. Fine-tune sampling rates based on actual data

## Validation Commands
\`\`\`bash
# Check traces in SigNoz
# Go to SigNoz Traces tab and filter by service.name = "stock-data-ingestion"

# Test trace generation
aws lambda invoke --function-name stock-data-ingestion \\
  --payload '{"symbols": ["AAPL"]}' \\
  --profile stock-analytics-admin response.json

# Monitor Lambda logs
aws logs tail /aws/lambda/stock-data-ingestion --follow \\
  --profile stock-analytics-admin
\`\`\`
EOF

    print_status "Validation report generated: /tmp/signoz_deployment_report.md"
    echo ""
    echo -e "${BLUE}📋 Deployment Summary:${NC}"
    cat /tmp/signoz_deployment_report.md
}

# Main execution
main() {
    validate_prerequisites
    deploy_otel_layer
    prepare_terraform
    deploy_infrastructure
    validate_deployment
    test_trace_generation
    generate_validation_report

    echo ""
    echo -e "${GREEN}🎉 SigNoz Integration Deployment Complete!${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Check SigNoz dashboard for traces within 1-2 minutes"
    echo "2. Import Week 3 custom dashboards from lambda_functions/shared/signoz_integration.py"
    echo "3. Monitor costs and adjust sampling rates as needed"
    echo "4. Set up alerting for critical performance thresholds"
    echo ""
    echo -e "${YELLOW}Expected Results:${NC}"
    echo "- Traces should appear in SigNoz with service names like 'stock-data-ingestion'"
    echo "- Week 1-3 business-aware tracing features preserved"
    echo "- Automatic instrumentation for HTTP/boto3 calls"
    echo "- Advanced performance monitoring data in traces"
}

# Execute main function
main "$@"