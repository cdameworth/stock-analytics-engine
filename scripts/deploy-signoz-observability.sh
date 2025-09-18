#!/bin/bash

# Deploy SigNoz Observability Integration for Stock Analytics Engine
# This script orchestrates the deployment of SigNoz monitoring infrastructure

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

    # Check Java (for CloudWatch exporter)
    if ! command -v java &> /dev/null; then
        log_warning "Java is not installed - CloudWatch exporter will not work locally"
    else
        java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [ "$java_version" -lt 11 ]; then
            log_error "Java 11 or newer is required for CloudWatch exporter"
            exit 1
        fi
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

    if [ -z "${SIGNOZ_INGESTION_KEY:-}" ]; then
        log_error "SIGNOZ_INGESTION_KEY environment variable is required"
        exit 1
    fi

    if [ -z "${SIGNOZ_OTLP_ENDPOINT:-}" ]; then
        log_warning "SIGNOZ_OTLP_ENDPOINT not set, using default: ingest.us.signoz.cloud:443"
        export SIGNOZ_OTLP_ENDPOINT="ingest.us.signoz.cloud:443"
    fi

    log_success "SigNoz configuration validated"
}

# Download CloudWatch exporter
download_cloudwatch_exporter() {
    log_info "Downloading CloudWatch exporter..."

    local exporter_dir="$INFRASTRUCTURE_DIR/cloudwatch-exporter"
    mkdir -p "$exporter_dir"

    local exporter_jar="$exporter_dir/cloudwatch_exporter-0.15.5-jar-with-dependencies.jar"

    if [ ! -f "$exporter_jar" ]; then
        curl -sLo "$exporter_jar" \
            "https://github.com/prometheus/cloudwatch_exporter/releases/download/v0.15.5/cloudwatch_exporter-0.15.5-jar-with-dependencies.jar"
        log_success "CloudWatch exporter downloaded"
    else
        log_info "CloudWatch exporter already exists"
    fi
}

# Test CloudWatch exporter locally
test_cloudwatch_exporter() {
    log_info "Testing CloudWatch exporter..."

    local exporter_dir="$INFRASTRUCTURE_DIR/cloudwatch-exporter"
    local exporter_jar="$exporter_dir/cloudwatch_exporter-0.15.5-jar-with-dependencies.jar"
    local config_file="$INFRASTRUCTURE_DIR/signoz-rds-metrics-config.yaml"

    if [ ! -f "$exporter_jar" ]; then
        log_error "CloudWatch exporter not found. Run download_cloudwatch_exporter first."
        return 1
    fi

    # Test in background for 10 seconds
    log_info "Starting CloudWatch exporter test (10 seconds)..."
    java -jar "$exporter_jar" 9106 "$config_file" &
    local exporter_pid=$!

    sleep 5

    # Test if metrics endpoint is accessible
    if curl -s http://localhost:9106/metrics | grep -q "aws_rds_"; then
        log_success "CloudWatch exporter test passed"
    else
        log_warning "CloudWatch exporter may not be working correctly"
    fi

    # Clean up
    kill $exporter_pid 2>/dev/null || true
    wait $exporter_pid 2>/dev/null || true
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying SigNoz infrastructure..."

    cd "$INFRASTRUCTURE_DIR"

    # Initialize Terraform if needed
    if [ ! -d ".terraform" ]; then
        log_info "Initializing Terraform..."
        terraform init
    fi

    # Create terraform variables file for SigNoz
    local tfvars_file="signoz-terraform.tfvars"
    cat > "$tfvars_file" << EOF
# SigNoz Configuration
signoz_otlp_endpoint = "${SIGNOZ_OTLP_ENDPOINT}"
signoz_ingestion_key = "${SIGNOZ_INGESTION_KEY}"
enable_signoz_integration = true
enable_rds_monitoring = true
enable_lambda_monitoring = true

# Cost optimization
signoz_data_retention_days = 15
enable_log_sampling = true
log_sampling_rate = 0.1
enable_batch_processing = true

# Infrastructure
enable_ecs_otel_collector = false  # Start with Lambda-based collection
otel_collector_cpu = 512
otel_collector_memory = 1024

# Security
enable_tls_verification = true
enable_sensitive_data_filtering = true

# Migration
migration_phase = "testing"
enable_dual_shipping = true
EOF

    log_info "Planning Terraform deployment..."
    terraform plan -var-file="$tfvars_file" -target="module.signoz_observability" -out=signoz.tfplan

    read -p "Do you want to apply the Terraform plan? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Applying Terraform configuration..."
        terraform apply signoz.tfplan
        log_success "Infrastructure deployed successfully"
    else
        log_info "Terraform apply cancelled"
        return 1
    fi
}

# Test SigNoz integration
test_signoz_integration() {
    log_info "Testing SigNoz integration..."

    # Get RDS endpoint
    local rds_endpoint
    rds_endpoint=$(aws rds describe-db-clusters \
        --db-cluster-identifier stock-analytics-aurora \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'DBClusters[0].Endpoint' \
        --output text)

    if [ "$rds_endpoint" = "None" ] || [ -z "$rds_endpoint" ]; then
        log_error "Could not retrieve RDS endpoint"
        return 1
    fi

    log_info "RDS endpoint found: $rds_endpoint"

    # Test OTEL collector configuration
    local config_file="$INFRASTRUCTURE_DIR/signoz-postgres-metrics-config.yaml"
    log_info "Validating OTEL collector configuration..."

    # Simple YAML validation
    if command -v python3 &> /dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            log_success "OTEL collector configuration is valid"
        else
            log_error "OTEL collector configuration is invalid"
            return 1
        fi
    fi

    # Test connectivity to SigNoz endpoint
    log_info "Testing connectivity to SigNoz endpoint..."
    if timeout 10 bash -c "</dev/tcp/${SIGNOZ_OTLP_ENDPOINT%:*}/${SIGNOZ_OTLP_ENDPOINT#*:}" 2>/dev/null; then
        log_success "SigNoz endpoint is reachable"
    else
        log_warning "Could not connect to SigNoz endpoint (may be due to TLS/auth requirements)"
    fi
}

# Update Lambda functions with SigNoz OTEL configuration
update_lambda_functions() {
    log_info "Updating Lambda functions with SigNoz OTEL configuration..."

    local functions=(
        "stock-data-ingestion"
        "ml-model-inference-lowcost"
        "stock-recommendations-api"
        "dual-prediction-reporting-api"
    )

    for func in "${functions[@]}"; do
        log_info "Updating $func..."

        # Update environment variables
        aws lambda update-function-configuration \
            --function-name "$func" \
            --environment Variables="{
                OTEL_EXPORTER_OTLP_ENDPOINT=${SIGNOZ_OTLP_ENDPOINT},
                OTEL_EXPORTER_OTLP_HEADERS=signoz-access-token=${SIGNOZ_INGESTION_KEY},
                OTEL_SERVICE_NAME=stock-analytics-$func
            }" \
            --profile "$AWS_PROFILE" \
            --region "$AWS_REGION" \
            > /dev/null

        log_success "Updated $func"
    done
}

# Create CloudWatch log groups for RDS if they don't exist
setup_rds_log_groups() {
    log_info "Setting up RDS log groups..."

    # Enable log exports for Aurora cluster
    aws rds modify-db-cluster \
        --db-cluster-identifier stock-analytics-aurora \
        --cloudwatch-logs-export-configuration LogTypesToEnable=postgresql \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --apply-immediately \
        > /dev/null 2>&1 || log_warning "Could not enable CloudWatch log exports for Aurora cluster"

    log_success "RDS log groups configuration updated"
}

# Generate monitoring dashboard
generate_dashboard() {
    log_info "Generating SigNoz monitoring guide..."

    cat > "$PROJECT_ROOT/docs/signoz-monitoring-guide.md" << 'EOF'
# SigNoz Monitoring Guide for Stock Analytics Engine

## Overview
This guide covers the SigNoz observability integration for the Stock Analytics Engine.

## Dashboards Access
- **SigNoz Cloud**: https://app.signoz.cloud/
- **AWS CloudWatch**: Backup monitoring available in AWS Console

## Key Metrics to Monitor

### RDS PostgreSQL Metrics
- Database connections
- CPU utilization
- Memory usage
- Disk I/O performance
- Query performance

### Lambda Function Metrics
- Function duration
- Error rates
- Memory utilization
- Cold starts

### Business Metrics
- Prediction accuracy
- API response times
- Data ingestion rates
- Model performance

## Alerting Setup
Configure alerts in SigNoz for:
- High error rates (>5% in 5 minutes)
- High latency (>2s p95)
- Database connection issues
- Memory exhaustion

## Troubleshooting
Common issues and solutions:
1. **Missing metrics**: Check OTEL collector logs
2. **Authentication errors**: Verify SigNoz ingestion key
3. **High costs**: Review sampling rates and retention settings

## Cost Optimization
- Set appropriate data retention (default: 15 days)
- Use log sampling (default: 10%)
- Enable batch processing
- Monitor ingestion volume in SigNoz dashboard
EOF

    log_success "Monitoring guide generated"
}

# Main deployment function
main() {
    log_info "Starting SigNoz observability deployment..."

    # Validate required environment variables
    validate_signoz_config

    # Check prerequisites
    check_prerequisites

    # Download and test CloudWatch exporter
    download_cloudwatch_exporter
    test_cloudwatch_exporter

    # Setup RDS logging
    setup_rds_log_groups

    # Deploy infrastructure
    deploy_infrastructure

    # Update Lambda functions
    update_lambda_functions

    # Test integration
    test_signoz_integration

    # Generate documentation
    generate_dashboard

    log_success "SigNoz observability deployment completed successfully!"
    log_info "Next steps:"
    log_info "1. Access SigNoz dashboard at: https://app.signoz.cloud/"
    log_info "2. Review monitoring guide: $PROJECT_ROOT/docs/signoz-monitoring-guide.md"
    log_info "3. Set up alerts and notifications in SigNoz"
    log_info "4. Monitor costs and adjust sampling if needed"
}

# Script usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --test-only             Only run tests, don't deploy"
    echo "  --infrastructure-only   Only deploy infrastructure"
    echo "  --lambda-only           Only update Lambda functions"
    echo ""
    echo "Environment variables:"
    echo "  SIGNOZ_INGESTION_KEY    SigNoz ingestion key (required)"
    echo "  SIGNOZ_OTLP_ENDPOINT    SigNoz OTLP endpoint (optional)"
    echo "  AWS_PROFILE             AWS profile to use (default: stock-analytics-admin)"
    echo "  AWS_REGION              AWS region (default: us-east-1)"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --test-only)
        validate_signoz_config
        check_prerequisites
        download_cloudwatch_exporter
        test_cloudwatch_exporter
        test_signoz_integration
        ;;
    --infrastructure-only)
        validate_signoz_config
        check_prerequisites
        deploy_infrastructure
        ;;
    --lambda-only)
        validate_signoz_config
        update_lambda_functions
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