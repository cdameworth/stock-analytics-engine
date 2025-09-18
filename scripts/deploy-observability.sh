#!/bin/bash
# Deploy OpenTelemetry Observability Stack for Stock Analytics Engine
# Configured for cdameworth.grafana.net

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
AWS_PROFILE=${AWS_PROFILE:-stock-analytics-admin}
REGION=${AWS_REGION:-us-east-1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not found. Please install Terraform."
        exit 1
    fi

    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install jq for JSON processing."
        exit 1
    fi

    # Check AWS profile
    if ! aws sts get-caller-identity --profile $AWS_PROFILE &> /dev/null; then
        log_error "AWS profile '$AWS_PROFILE' not configured or invalid."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Validate existing infrastructure
validate_infrastructure() {
    log_info "Validating existing infrastructure..."

    # Check if core Lambda functions exist
    REQUIRED_FUNCTIONS=(
        "stock-data-ingestion"
        "stock-recommendations-api"
    )

    for func in "${REQUIRED_FUNCTIONS[@]}"; do
        if ! aws lambda get-function --function-name "$func" --profile $AWS_PROFILE &> /dev/null; then
            log_error "Required Lambda function '$func' not found. Deploy core infrastructure first."
            exit 1
        fi
    done

    # Check if DynamoDB table exists
    if ! aws dynamodb describe-table --table-name "stock-recommendations" --profile $AWS_PROFILE &> /dev/null; then
        log_error "Required DynamoDB table 'stock-recommendations' not found."
        exit 1
    fi

    log_success "Core infrastructure validation passed"
}

# Build OpenTelemetry Lambda layer
build_otel_layer() {
    log_info "Building OpenTelemetry Lambda layer..."

    cd "$PROJECT_ROOT/infrastructure"

    # Create temporary directory for layer build
    mkdir -p /tmp/otel-layer-build
    cd /tmp/otel-layer-build

    # Copy requirements and build script
    cp "$PROJECT_ROOT/infrastructure/otel-layer-requirements.txt" requirements.txt
    cp "$PROJECT_ROOT/infrastructure/otel-python-installer.sh" install.sh

    # Make installer executable
    chmod +x install.sh

    # Run build (this would normally happen in a Docker container for Lambda compatibility)
    log_info "Note: For production, build this layer in Amazon Linux environment"
    log_info "Using AWS managed OpenTelemetry layer instead"

    cd - > /dev/null
    log_success "OpenTelemetry layer configuration ready"
}

# Deploy observability infrastructure
deploy_infrastructure() {
    log_info "Deploying observability infrastructure..."

    cd "$PROJECT_ROOT/infrastructure"

    # Initialize Terraform (required for new providers)
    log_info "Initializing Terraform..."
    terraform init

    # Validate Terraform configuration
    log_info "Validating Terraform configuration..."
    if ! terraform validate; then
        log_error "Terraform validation failed"
        exit 1
    fi

    # Plan deployment
    log_info "Planning observability deployment..."
    terraform plan \
        -var-file="terraform-tier1.tfvars" \
        -var-file="terraform-cdameworth-observability.tfvars" \
        -out=observability.tfplan

    # Confirm deployment
    echo
    log_warning "This will deploy comprehensive observability infrastructure."
    log_warning "Estimated cost: ~$20-40/month additional AWS costs"
    echo "Do you want to proceed? (y/N)"
    read -r response

    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Applying observability infrastructure..."
        terraform apply observability.tfplan

        if [ $? -eq 0 ]; then
            log_success "Observability infrastructure deployed successfully!"
        else
            log_error "Deployment failed. Check Terraform output for details."
            exit 1
        fi
    else
        log_info "Deployment cancelled by user"
        exit 0
    fi

    cd - > /dev/null
}

# Verify deployment
verify_deployment() {
    log_info "Verifying observability deployment..."

    # Check if enhanced Lambda functions were created
    OTEL_FUNCTIONS=(
        "stock-data-ingestion-otel"
        "stock-recommendations-api-otel"
        "ml-model-inference-lowcost-otel"
    )

    for func in "${OTEL_FUNCTIONS[@]}"; do
        if aws lambda get-function --function-name "$func" --profile $AWS_PROFILE &> /dev/null; then
            log_success "Enhanced function '$func' deployed"
        else
            log_warning "Enhanced function '$func' not found (may use existing function)"
        fi
    done

    # Check CloudWatch dashboards
    if aws cloudwatch list-dashboards --profile $AWS_PROFILE --query "DashboardEntries[?contains(DashboardName, 'StockAnalytics')]" --output text | grep -q "StockAnalytics"; then
        log_success "CloudWatch dashboards created"
    else
        log_warning "CloudWatch dashboards not found"
    fi

    # Check Secrets Manager
    if aws secretsmanager describe-secret --secret-id "stock-analytics-grafana-credentials" --profile $AWS_PROFILE &> /dev/null; then
        log_success "Grafana credentials stored in Secrets Manager"
    else
        log_warning "Grafana credentials secret not found"
    fi

    log_success "Deployment verification completed"
}

# Configure Grafana Cloud
configure_grafana() {
    log_info "Grafana Cloud configuration complete via Terraform"
    log_info "Access your dashboards at: https://cdameworth.grafana.net"

    echo
    echo "Grafana Cloud Setup:"
    echo "1. Login to https://cdameworth.grafana.net"
    echo "2. Navigate to Dashboards → Browse"
    echo "3. Look for 'Stock Analytics Engine - Overview' dashboard"
    echo "4. Check Alerting → Alert Rules for monitoring rules"
    echo
    echo "If dashboards are not visible immediately:"
    echo "- Data sources may need a few minutes to sync"
    echo "- Generate some test data using the test script"
    echo "- Check AWS CloudWatch for metrics first"
}

# Generate test data
generate_test_data() {
    log_info "Generating test telemetry data..."

    # Trigger the stock recommendations API to generate traces
    log_info "Testing API endpoints..."

    # Get API Gateway URL from Terraform output
    cd "$PROJECT_ROOT/infrastructure"
    API_URL=$(terraform output -raw api_gateway_url 2>/dev/null || echo "")
    API_KEY=$(terraform output -raw api_key_value 2>/dev/null || echo "lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq")
    cd - > /dev/null

    if [ -n "$API_URL" ]; then
        log_info "Testing recommendations API at $API_URL"

        # Test recommendations endpoint
        curl -s -H "x-api-key: $API_KEY" \
             -H "Content-Type: application/json" \
             "$API_URL/recommendations?limit=5" \
             > /tmp/api_test.json

        if [ $? -eq 0 ]; then
            log_success "API test completed - check CloudWatch and Grafana for telemetry data"
        else
            log_warning "API test failed - manual testing may be needed"
        fi

        rm -f /tmp/api_test.json
    else
        log_warning "Could not determine API Gateway URL - test manually"
    fi

    # Trigger data ingestion function
    if aws lambda get-function --function-name "stock-data-ingestion" --profile $AWS_PROFILE &> /dev/null; then
        log_info "Triggering data ingestion for telemetry generation..."

        aws lambda invoke \
            --function-name "stock-data-ingestion" \
            --payload '{"source": "observability-test", "symbols": ["AAPL", "GOOGL", "MSFT"]}' \
            --profile $AWS_PROFILE \
            /tmp/ingestion_test.json > /dev/null

        if [ $? -eq 0 ]; then
            log_success "Data ingestion test completed"
        fi

        rm -f /tmp/ingestion_test.json
    fi
}

# Display next steps
show_next_steps() {
    echo
    echo "=========================================="
    log_success "Observability Deployment Complete!"
    echo "=========================================="
    echo
    echo "🎯 Next Steps:"
    echo "1. Access Grafana Cloud: https://cdameworth.grafana.net"
    echo "2. Check CloudWatch Dashboards in AWS Console"
    echo "3. Verify X-Ray traces in AWS X-Ray Console"
    echo "4. Run comprehensive tests:"
    echo "   ./scripts/test-observability.sh"
    echo
    echo "📊 Key Resources Created:"
    echo "• Enhanced Lambda functions with OpenTelemetry"
    echo "• CloudWatch dashboards and alarms"
    echo "• Grafana Cloud data sources and dashboards"
    echo "• X-Ray sampling rules and service maps"
    echo "• Custom business metrics collection"
    echo
    echo "💰 Cost Monitoring:"
    echo "• Budget set to \$75/month for observability"
    echo "• Cost alarms configured for early warning"
    echo "• Optimized sampling (10%) to control costs"
    echo
    echo "🔧 Troubleshooting:"
    echo "• Check logs: aws logs describe-log-groups --profile $AWS_PROFILE"
    echo "• View traces: AWS X-Ray Console"
    echo "• Documentation: docs/observability-deployment-guide.md"
    echo
    echo "📧 Alerts configured for: cdameworth@gmail.com"
    echo
}

# Main deployment flow
main() {
    echo "=========================================="
    echo "Stock Analytics Observability Deployment"
    echo "Target: cdameworth.grafana.net"
    echo "=========================================="
    echo

    check_prerequisites
    echo

    validate_infrastructure
    echo

    build_otel_layer
    echo

    deploy_infrastructure
    echo

    verify_deployment
    echo

    configure_grafana
    echo

    generate_test_data
    echo

    show_next_steps
}

# Run deployment
main "$@"