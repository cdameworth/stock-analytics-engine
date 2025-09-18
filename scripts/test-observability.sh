#!/bin/bash
# Test OpenTelemetry and Observability Integration
# Validates telemetry data flow and Grafana Cloud connectivity

set -e

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

# Check AWS CLI and profile
check_aws_setup() {
    log_info "Checking AWS setup..."

    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi

    if ! aws sts get-caller-identity --profile $AWS_PROFILE &> /dev/null; then
        log_error "AWS profile '$AWS_PROFILE' not configured or invalid."
        exit 1
    fi

    log_success "AWS setup verified"
}

# Test Lambda functions exist
test_lambda_functions() {
    log_info "Testing Lambda functions..."

    FUNCTIONS=(
        "stock-data-ingestion-otel"
        "ml-model-inference-lowcost-otel"
        "stock-recommendations-api-otel"
    )

    for func in "${FUNCTIONS[@]}"; do
        if aws lambda get-function --function-name "$func" --profile $AWS_PROFILE &> /dev/null; then
            log_success "Lambda function '$func' exists"

            # Check if OpenTelemetry layer is attached
            layers=$(aws lambda get-function --function-name "$func" --profile $AWS_PROFILE --query 'Configuration.Layers[].Arn' --output text)
            if echo "$layers" | grep -q "aws-otel-python"; then
                log_success "OpenTelemetry layer attached to '$func'"
            else
                log_warning "OpenTelemetry layer NOT found on '$func'"
            fi
        else
            log_warning "Lambda function '$func' not found (might use original function)"
        fi
    done
}

# Test CloudWatch dashboards
test_cloudwatch_dashboards() {
    log_info "Testing CloudWatch dashboards..."

    DASHBOARDS=(
        "StockAnalytics-OpenTelemetry-Overview"
        "StockAnalytics-Comprehensive-Monitoring"
    )

    for dashboard in "${DASHBOARDS[@]}"; do
        if aws cloudwatch list-dashboards --profile $AWS_PROFILE --query "DashboardEntries[?DashboardName=='$dashboard']" --output text | grep -q "$dashboard"; then
            log_success "Dashboard '$dashboard' exists"
        else
            log_warning "Dashboard '$dashboard' not found"
        fi
    done
}

# Test X-Ray sampling rules
test_xray_sampling() {
    log_info "Testing X-Ray sampling rules..."

    sampling_rules=$(aws xray get-sampling-rules --profile $AWS_PROFILE --query 'SamplingRuleRecords[?RuleName==`StockAnalyticsSampling` || RuleName==`StockAnalyticsDetailed`]' --output json)

    if echo "$sampling_rules" | jq -e '. | length > 0' &> /dev/null; then
        log_success "X-Ray sampling rules configured"
        echo "$sampling_rules" | jq '.[] | {RuleName: .SamplingRule.RuleName, FixedRate: .SamplingRule.FixedRate, ReservoirSize: .SamplingRule.ReservoirSize}'
    else
        log_warning "No Stock Analytics X-Ray sampling rules found"
    fi
}

# Test Secrets Manager for Grafana credentials
test_grafana_secrets() {
    log_info "Testing Grafana Cloud credentials..."

    if aws secretsmanager describe-secret --secret-id "stock-analytics-grafana-credentials" --profile $AWS_PROFILE &> /dev/null; then
        log_success "Grafana credentials secret exists"

        # Test secret access (without revealing content)
        if aws secretsmanager get-secret-value --secret-id "stock-analytics-grafana-credentials" --profile $AWS_PROFILE --query 'SecretString' --output text &> /dev/null; then
            log_success "Grafana credentials accessible"
        else
            log_error "Cannot access Grafana credentials"
        fi
    else
        log_warning "Grafana credentials secret not found"
    fi
}

# Trigger Lambda function to generate telemetry
test_lambda_execution() {
    log_info "Testing Lambda function execution..."

    # Test stock recommendations API
    FUNCTION_NAME="stock-recommendations-api"
    if aws lambda get-function --function-name "$FUNCTION_NAME" --profile $AWS_PROFILE &> /dev/null; then
        log_info "Triggering '$FUNCTION_NAME' for telemetry generation..."

        # Create test payload
        PAYLOAD='{
            "httpMethod": "GET",
            "path": "/recommendations",
            "headers": {"x-api-key": "test-key"},
            "queryStringParameters": {"limit": "5"},
            "requestContext": {"requestId": "test-observability-'$(date +%s)'"}
        }'

        # Invoke function
        RESULT=$(aws lambda invoke \
            --function-name "$FUNCTION_NAME" \
            --payload "$PAYLOAD" \
            --profile $AWS_PROFILE \
            --output json \
            /tmp/lambda-response.json)

        if echo "$RESULT" | jq -e '.StatusCode == 200' &> /dev/null; then
            log_success "Lambda function executed successfully"

            # Check for any errors in the response
            if grep -q "errorMessage" /tmp/lambda-response.json; then
                log_warning "Lambda execution had errors - check CloudWatch logs"
                cat /tmp/lambda-response.json | jq '.errorMessage' 2>/dev/null || true
            else
                log_success "Lambda execution completed without errors"
            fi
        else
            log_error "Lambda function execution failed"
            echo "$RESULT" | jq '.'
        fi

        rm -f /tmp/lambda-response.json
    else
        log_warning "Stock recommendations API function not found"
    fi
}

# Check CloudWatch logs for telemetry data
test_cloudwatch_logs() {
    log_info "Checking CloudWatch logs for telemetry data..."

    LOG_GROUPS=(
        "/aws/lambda/stock-data-ingestion"
        "/aws/lambda/stock-recommendations-api"
        "/aws/otel/stock-analytics"
    )

    for log_group in "${LOG_GROUPS[@]}"; do
        if aws logs describe-log-groups --log-group-name-prefix "$log_group" --profile $AWS_PROFILE --query 'logGroups[0].logGroupName' --output text | grep -q "$log_group"; then
            log_success "Log group '$log_group' exists"

            # Check for recent log entries
            recent_logs=$(aws logs describe-log-streams \
                --log-group-name "$log_group" \
                --order-by LastEventTime \
                --descending \
                --max-items 1 \
                --profile $AWS_PROFILE \
                --query 'logStreams[0].lastEventTime' \
                --output text 2>/dev/null || echo "0")

            if [ "$recent_logs" != "0" ] && [ "$recent_logs" != "None" ]; then
                # Convert timestamp to human readable
                recent_date=$(date -d "@$((recent_logs/1000))" 2>/dev/null || echo "unknown")
                log_success "Recent log activity: $recent_date"
            else
                log_warning "No recent log activity in '$log_group'"
            fi
        else
            log_warning "Log group '$log_group' not found"
        fi
    done
}

# Test CloudWatch custom metrics
test_custom_metrics() {
    log_info "Testing custom metrics..."

    NAMESPACES=(
        "StockAnalytics/DataIngestion"
        "StockAnalytics/MLInference"
        "StockAnalytics/Business"
        "StockAnalytics/OpenTelemetry"
    )

    for namespace in "${NAMESPACES[@]}"; do
        metrics=$(aws cloudwatch list-metrics --namespace "$namespace" --profile $AWS_PROFILE --query 'Metrics[].MetricName' --output text 2>/dev/null || echo "")

        if [ -n "$metrics" ]; then
            log_success "Custom metrics found in namespace '$namespace'"
            echo "  Metrics: $metrics"
        else
            log_warning "No metrics found in namespace '$namespace'"
        fi
    done
}

# Test CloudWatch alarms
test_cloudwatch_alarms() {
    log_info "Testing CloudWatch alarms..."

    alarms=$(aws cloudwatch describe-alarms \
        --alarm-name-prefix "StockAnalytics" \
        --profile $AWS_PROFILE \
        --query 'MetricAlarms[].{Name:AlarmName,State:StateValue}' \
        --output table 2>/dev/null || echo "")

    if [ -n "$alarms" ]; then
        log_success "Stock Analytics alarms found:"
        echo "$alarms"
    else
        log_warning "No Stock Analytics alarms found"
    fi
}

# Generate test data for observability
generate_test_data() {
    log_info "Generating test telemetry data..."

    # Trigger data ingestion function if it exists
    if aws lambda get-function --function-name "stock-data-ingestion" --profile $AWS_PROFILE &> /dev/null; then
        log_info "Triggering data ingestion for test data..."

        PAYLOAD='{
            "source": "manual-test",
            "processing_mode": "test",
            "symbols": ["AAPL", "GOOGL", "MSFT"]
        }'

        aws lambda invoke \
            --function-name "stock-data-ingestion" \
            --payload "$PAYLOAD" \
            --profile $AWS_PROFILE \
            --output json \
            /tmp/ingestion-response.json > /dev/null

        if [ $? -eq 0 ]; then
            log_success "Test data generation triggered"
        else
            log_warning "Failed to trigger test data generation"
        fi

        rm -f /tmp/ingestion-response.json
    fi
}

# Validate Grafana Cloud connectivity (basic check)
test_grafana_connectivity() {
    log_info "Testing Grafana Cloud connectivity..."

    # Check if OTLP endpoint is reachable
    OTLP_ENDPOINT="otlp-gateway-prod-us-central-0.grafana.net"

    if ping -c 1 "$OTLP_ENDPOINT" &> /dev/null; then
        log_success "Grafana Cloud OTLP endpoint is reachable"
    else
        log_warning "Cannot reach Grafana Cloud OTLP endpoint (this may be normal if ICMP is blocked)"
    fi

    # Test HTTPS connectivity
    if curl -s --connect-timeout 5 "https://$OTLP_ENDPOINT" &> /dev/null; then
        log_success "HTTPS connectivity to Grafana Cloud confirmed"
    else
        log_warning "HTTPS connectivity test failed (may need authentication)"
    fi
}

# Main test execution
main() {
    echo "=========================================="
    echo "OpenTelemetry Observability Test Suite"
    echo "=========================================="
    echo

    check_aws_setup
    echo

    test_lambda_functions
    echo

    test_cloudwatch_dashboards
    echo

    test_xray_sampling
    echo

    test_grafana_secrets
    echo

    test_cloudwatch_logs
    echo

    test_custom_metrics
    echo

    test_cloudwatch_alarms
    echo

    test_grafana_connectivity
    echo

    log_info "Generating test telemetry data..."
    generate_test_data
    test_lambda_execution
    echo

    log_info "Final validation..."
    sleep 5  # Wait for logs to propagate
    test_cloudwatch_logs

    echo
    echo "=========================================="
    log_success "Observability test suite completed!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Check Grafana Cloud dashboards at https://your-instance.grafana.net"
    echo "2. Verify traces in AWS X-Ray console"
    echo "3. Monitor CloudWatch metrics and alarms"
    echo "4. Review logs for any error patterns"
    echo
    echo "For troubleshooting, see: docs/observability-deployment-guide.md"
}

# Run tests
main "$@"