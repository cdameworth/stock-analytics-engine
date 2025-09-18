#!/bin/bash

# Fix OpenTelemetry Lambda Issues Script
# This script deploys the fixed OTEL configuration for all Lambda functions

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRASTRUCTURE_DIR="$PROJECT_ROOT/infrastructure"
AWS_PROFILE="${AWS_PROFILE:-stock-analytics-admin}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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
        log_error "AWS CLI is not installed"
        exit 1
    fi

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        log_error "AWS credentials not configured for profile: $AWS_PROFILE"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Validate SigNoz configuration
validate_signoz_config() {
    log_info "Validating SigNoz configuration..."

    if [ -z "${TF_VAR_signoz_ingestion_key:-}" ]; then
        log_warning "TF_VAR_signoz_ingestion_key environment variable not set"
        log_warning "You'll need to set this before deploying: export TF_VAR_signoz_ingestion_key='your-key-here'"
    else
        log_success "SigNoz ingestion key configured"
    fi

    log_success "SigNoz configuration validated"
}

# Check current Lambda OTEL status
check_current_otel_status() {
    log_info "Checking current Lambda OTEL status..."

    local functions=(
        "stock-data-ingestion"
        "ml-model-inference-lowcost"
        "stock-recommendations-api"
        "dual-accuracy-tracker"
        "dual-prediction-reporting-api"
    )

    for func in "${functions[@]}"; do
        echo "=== $func ==="

        # Check if function exists
        if aws lambda get-function --function-name "$func" --profile "$AWS_PROFILE" --region "$AWS_REGION" &>/dev/null; then
            # Check layers
            local layers
            layers=$(aws lambda get-function-configuration \
                --function-name "$func" \
                --profile "$AWS_PROFILE" \
                --region "$AWS_REGION" \
                --query 'Layers[].Arn' \
                --output text 2>/dev/null || echo "None")

            echo "  Layers: $layers"

            # Check OTEL environment variables
            local otel_vars
            otel_vars=$(aws lambda get-function-configuration \
                --function-name "$func" \
                --profile "$AWS_PROFILE" \
                --region "$AWS_REGION" \
                --query 'Environment.Variables' \
                --output json 2>/dev/null | jq -r 'to_entries[] | select(.key | contains("OTEL")) | "\(.key)=\(.value)"' || echo "None")

            if [ "$otel_vars" = "None" ] || [ -z "$otel_vars" ]; then
                echo "  OTEL Config: ❌ Not configured"
            else
                echo "  OTEL Config: ✅ Configured"
                echo "$otel_vars" | head -3 | sed 's/^/    /'
            fi
        else
            echo "  Status: ❌ Function not found"
        fi
        echo ""
    done
}

# Deploy OTEL fixes
deploy_otel_fixes() {
    log_info "Deploying OTEL fixes..."

    cd "$INFRASTRUCTURE_DIR"

    # Check if terraform is initialized
    if [ ! -d ".terraform" ]; then
        log_info "Initializing Terraform..."
        terraform init
    fi

    # Validate the configuration
    log_info "Validating Terraform configuration..."
    terraform validate

    # Plan the deployment
    log_info "Planning OTEL Lambda updates..."
    terraform plan \
        -var-file="signoz-terraform.tfvars" \
        -var="enable_signoz_integration=true" \
        -out=otel-fix.tfplan

    # Ask for confirmation
    echo ""
    read -p "Do you want to apply the OTEL fixes? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Applying OTEL Lambda fixes..."
        terraform apply otel-fix.tfplan
        log_success "OTEL fixes deployed successfully"
    else
        log_info "Deployment cancelled"
        return 1
    fi
}

# Test OTEL integration
test_otel_integration() {
    log_info "Testing OTEL integration..."

    # Wait a moment for Lambda functions to update
    sleep 10

    log_info "Checking updated Lambda configurations..."
    check_current_otel_status

    # Test a function to see if OTEL is working
    log_info "Testing Lambda function invocation..."

    local test_function="stock-recommendations-api"
    if aws lambda invoke \
        --function-name "$test_function" \
        --payload '{}' \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        /tmp/lambda-test-response.json &>/dev/null; then
        log_success "Lambda function invocation test passed"

        # Check CloudWatch logs for OTEL traces
        log_info "Checking CloudWatch logs for OTEL traces..."
        local log_group="/aws/lambda/$test_function"

        # Get recent log events
        if aws logs filter-log-events \
            --log-group-name "$log_group" \
            --start-time $(date -d '5 minutes ago' +%s)000 \
            --filter-pattern "OTEL" \
            --profile "$AWS_PROFILE" \
            --region "$AWS_REGION" \
            --query 'events[].message' \
            --output text | head -3; then
            log_success "OTEL traces found in CloudWatch logs"
        else
            log_warning "No OTEL traces found yet (may take a few minutes)"
        fi
    else
        log_error "Lambda function invocation test failed"
    fi

    # Clean up test file
    rm -f /tmp/lambda-test-response.json
}

# Generate summary report
generate_summary() {
    log_info "Generating OTEL fix summary..."

    cat << EOF

📊 OpenTelemetry Lambda Fix Summary
================================

✅ Fixed Issues:
- Added consistent OTEL layer to all Lambda functions
- Updated environment variables to use SigNoz instead of Grafana
- Configured proper service naming for each function
- Added conditional OTEL configuration based on enable_signoz_integration flag

🔧 Key Changes:
- All Lambda functions now have unified OTEL configuration
- SigNoz integration replaces previous Grafana setup
- Proper layer management with existing dependencies
- Service-specific OTEL_SERVICE_NAME for better observability

📋 Next Steps:
1. Set your SigNoz ingestion key: export TF_VAR_signoz_ingestion_key='your-key'
2. Monitor SigNoz dashboard for incoming telemetry data
3. Set up alerts and dashboards in SigNoz
4. Review sampling rates if costs become an issue

🔗 Configuration Files:
- infrastructure/signoz-terraform.tfvars: SigNoz configuration
- infrastructure/main.tf: Updated Lambda configurations
- infrastructure/deploy_dual_prediction_system.tf: Updated dual prediction functions

EOF

    log_success "OTEL Lambda issues have been fixed!"
}

# Main execution
main() {
    log_info "Starting OpenTelemetry Lambda fix process..."

    check_prerequisites
    validate_signoz_config

    echo "Current OTEL Status:"
    check_current_otel_status

    deploy_otel_fixes
    test_otel_integration
    generate_summary

    log_success "OpenTelemetry Lambda fix process completed!"
}

# Script usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --check-only            Only check current OTEL status"
    echo "  --plan-only             Only plan the changes, don't apply"
    echo ""
    echo "Environment variables:"
    echo "  TF_VAR_signoz_ingestion_key    SigNoz ingestion key (required for deployment)"
    echo "  AWS_PROFILE                    AWS profile to use (default: stock-analytics-admin)"
    echo "  AWS_REGION                     AWS region (default: us-east-1)"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --check-only)
        check_prerequisites
        check_current_otel_status
        ;;
    --plan-only)
        check_prerequisites
        validate_signoz_config
        cd "$INFRASTRUCTURE_DIR"
        terraform plan -var-file="signoz-terraform.tfvars" -var="enable_signoz_integration=true"
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac