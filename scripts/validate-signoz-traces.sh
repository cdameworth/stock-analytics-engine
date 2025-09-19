#!/bin/bash

# SigNoz Trace Validation Script
# Validates that OpenTelemetry traces are properly configured and reaching SigNoz

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

AWS_PROFILE="stock-analytics-admin"

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo -e "${BLUE}🔍 SigNoz OpenTelemetry Validation${NC}"
echo "===================================="

# Test 1: Check Lambda Layer Configuration
echo -e "${BLUE}Test 1: Lambda Layer Configuration${NC}"

FUNCTIONS=("stock-data-ingestion" "stock-recommendations-api" "ml-model-inference-lowcost" "dual-accuracy-tracker" "dual-prediction-reporting-api")

for func in "${FUNCTIONS[@]}"; do
    echo "Checking $func..."

    # Get function configuration
    FUNC_CONFIG=$(aws lambda get-function-configuration \
        --function-name "$func" \
        --profile "$AWS_PROFILE" 2>/dev/null || echo "NOT_FOUND")

    if [ "$FUNC_CONFIG" = "NOT_FOUND" ]; then
        print_warning "$func: Function not found"
        continue
    fi

    # Check for OTEL layers
    OTEL_LAYERS=$(echo "$FUNC_CONFIG" | jq -r '.Layers[]?.Arn' 2>/dev/null | grep -i opentelemetry || echo "")

    if [ -n "$OTEL_LAYERS" ]; then
        print_status "$func: OTEL layer configured"
        echo "  Layer: $(echo "$OTEL_LAYERS" | head -n1)"
    else
        print_error "$func: No OTEL layer found"
    fi

    # Check required environment variables
    ENV_WRAPPER=$(echo "$FUNC_CONFIG" | jq -r '.Environment.Variables.AWS_LAMBDA_EXEC_WRAPPER // "NOT_SET"')
    ENV_ENDPOINT=$(echo "$FUNC_CONFIG" | jq -r '.Environment.Variables.OTEL_EXPORTER_OTLP_ENDPOINT // "NOT_SET"')
    ENV_HEADERS=$(echo "$FUNC_CONFIG" | jq -r '.Environment.Variables.OTEL_EXPORTER_OTLP_HEADERS // "NOT_SET"')

    if [ "$ENV_WRAPPER" = "/opt/otel-instrument" ]; then
        print_status "$func: AWS_LAMBDA_EXEC_WRAPPER configured"
    else
        print_error "$func: AWS_LAMBDA_EXEC_WRAPPER missing or incorrect"
    fi

    if [[ "$ENV_ENDPOINT" == *"signoz.cloud"* ]]; then
        print_status "$func: SigNoz endpoint configured"
    else
        print_error "$func: SigNoz endpoint missing or incorrect"
    fi

    if [[ "$ENV_HEADERS" == *"signoz-ingestion-key"* ]]; then
        print_status "$func: SigNoz ingestion key configured"
    else
        print_error "$func: SigNoz ingestion key missing"
    fi

    echo ""
done

# Test 2: Test Trace Generation
echo -e "${BLUE}Test 2: Trace Generation${NC}"

print_info "Testing trace generation with stock-data-ingestion function..."

# Create test payload
TEST_PAYLOAD='{"symbols": ["AAPL"], "test_mode": true, "source": "signoz_validation"}'

# Invoke function
aws lambda invoke \
    --function-name stock-data-ingestion \
    --payload "$TEST_PAYLOAD" \
    --profile "$AWS_PROFILE" \
    /tmp/trace_test_response.json >/dev/null 2>&1

# Check response
if [ -f "/tmp/trace_test_response.json" ]; then
    RESPONSE=$(cat /tmp/trace_test_response.json)

    if echo "$RESPONSE" | jq -e '.statusCode == 200' >/dev/null 2>&1; then
        print_status "Function executed successfully"
    else
        print_warning "Function execution returned non-200 status"
        echo "Response: $RESPONSE"
    fi

    # Check for OTEL-related content in response
    if echo "$RESPONSE" | grep -q -i "otel\|trace\|span"; then
        print_status "Response contains OTEL-related content"
    fi
else
    print_error "Failed to invoke function"
fi

# Clean up
rm -f /tmp/trace_test_response.json

# Test 3: Check CloudWatch Logs for OTEL Activity
echo -e "${BLUE}Test 3: CloudWatch Logs Analysis${NC}"

print_info "Checking recent logs for OTEL activity..."

# Get recent log events from stock-data-ingestion
LOG_EVENTS=$(aws logs filter-log-events \
    --log-group-name "/aws/lambda/stock-data-ingestion" \
    --start-time $(date -d '5 minutes ago' +%s)000 \
    --profile "$AWS_PROFILE" \
    --query 'events[].message' \
    --output text 2>/dev/null || echo "NO_LOGS")

if [ "$LOG_EVENTS" != "NO_LOGS" ]; then
    # Check for OTEL-related log entries
    if echo "$LOG_EVENTS" | grep -q -i "otel\|opentelemetry\|trace\|span"; then
        print_status "OTEL activity detected in logs"
    else
        print_warning "No OTEL activity in recent logs"
    fi

    # Check for errors
    if echo "$LOG_EVENTS" | grep -q -i "error\|exception\|failed"; then
        print_warning "Errors detected in recent logs"
        echo "Recent errors:"
        echo "$LOG_EVENTS" | grep -i "error\|exception\|failed" | head -3
    else
        print_status "No errors in recent logs"
    fi
else
    print_warning "No recent logs found (function may not have been invoked)"
fi

# Test 4: Network Connectivity to SigNoz
echo -e "${BLUE}Test 4: SigNoz Connectivity${NC}"

print_info "Testing connectivity to SigNoz endpoint..."

# Test HTTPS connectivity to SigNoz
if curl -s --max-time 10 "https://ingest.us.signoz.cloud" >/dev/null; then
    print_status "SigNoz endpoint is reachable"
else
    print_error "Cannot reach SigNoz endpoint"
fi

# Test 5: Environment Variable Summary
echo -e "${BLUE}Test 5: Environment Variable Summary${NC}"

print_info "Summary of OTEL configuration across all functions:"

echo ""
echo "| Function | EXEC_WRAPPER | OTLP_ENDPOINT | HEADERS | SERVICE_NAME |"
echo "|----------|--------------|---------------|---------|--------------|"

for func in "${FUNCTIONS[@]}"; do
    CONFIG=$(aws lambda get-function-configuration \
        --function-name "$func" \
        --profile "$AWS_PROFILE" 2>/dev/null || echo "NOT_FOUND")

    if [ "$CONFIG" = "NOT_FOUND" ]; then
        echo "| $func | ❌ | ❌ | ❌ | ❌ |"
        continue
    fi

    WRAPPER=$(echo "$CONFIG" | jq -r '.Environment.Variables.AWS_LAMBDA_EXEC_WRAPPER // "❌"')
    ENDPOINT=$(echo "$CONFIG" | jq -r '.Environment.Variables.OTEL_EXPORTER_OTLP_ENDPOINT // "❌"')
    HEADERS=$(echo "$CONFIG" | jq -r '.Environment.Variables.OTEL_EXPORTER_OTLP_HEADERS // "❌"')
    SERVICE=$(echo "$CONFIG" | jq -r '.Environment.Variables.OTEL_SERVICE_NAME // "❌"')

    # Simplify display
    [ "$WRAPPER" = "/opt/otel-instrument" ] && WRAPPER="✅" || WRAPPER="❌"
    [[ "$ENDPOINT" == *"signoz.cloud"* ]] && ENDPOINT="✅" || ENDPOINT="❌"
    [[ "$HEADERS" == *"signoz-ingestion-key"* ]] && HEADERS="✅" || HEADERS="❌"
    [ "$SERVICE" != "❌" ] && SERVICE="✅" || SERVICE="❌"

    echo "| $func | $WRAPPER | $ENDPOINT | $HEADERS | $SERVICE |"
done

echo ""

# Test 6: Recommendations
echo -e "${BLUE}Recommendations${NC}"

echo "1. 📊 Check SigNoz Dashboard:"
echo "   - Go to SigNoz Traces tab"
echo "   - Filter by service.name = 'stock-data-ingestion'"
echo "   - Look for traces from the last 5 minutes"
echo ""

echo "2. 🔍 Manual Trace Generation:"
echo "   aws lambda invoke --function-name stock-data-ingestion \\"
echo "     --payload '{\"symbols\": [\"AAPL\"]}' \\"
echo "     --profile stock-analytics-admin response.json"
echo ""

echo "3. 📋 Monitor Logs:"
echo "   aws logs tail /aws/lambda/stock-data-ingestion --follow \\"
echo "     --profile stock-analytics-admin"
echo ""

echo "4. 🎯 Expected in SigNoz:"
echo "   - Service names: stock-data-ingestion, ml-model-inference, etc."
echo "   - Automatic HTTP/boto3 spans from OTEL auto-instrumentation"
echo "   - Custom business spans with financial attributes"
echo "   - Week 3 performance monitoring data"
echo ""

# Final status
echo -e "${GREEN}🎯 Validation Complete${NC}"
echo ""
echo "If traces are not appearing in SigNoz within 2-3 minutes:"
echo "1. Check the SigNoz ingestion key is correct"
echo "2. Verify network connectivity from Lambda to SigNoz"
echo "3. Check CloudWatch logs for OTEL-related errors"
echo "4. Ensure the custom OTEL layer was deployed successfully"