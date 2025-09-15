#!/bin/bash
# Test Model Tuning Report System

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-stock-analytics-admin}"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Test functions
test_tuning_reporter_deployment() {
    print_status "Testing tuning reporter deployment..."

    local function_name="model-tuning-reporter"
    local response=$(aws lambda get-function \
        --function-name "$function_name" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null)

    if [[ $? -eq 0 ]]; then
        local last_modified=$(echo "$response" | jq -r '.Configuration.LastModified')
        print_success "Tuning reporter function exists (Last modified: $last_modified)"
        return 0
    else
        print_error "Tuning reporter function not found"
        return 1
    fi
}

test_manual_report_trigger() {
    print_status "Testing manual tuning report trigger..."

    # Create test payload
    local test_payload=$(cat <<EOF
{
    "model_type": "price",
    "tuning_data": {
        "tuning_type": "Manual Test Run",
        "before_accuracy": 0.65,
        "after_accuracy": 0.72,
        "predictions_analyzed": 1250,
        "training_samples": 1000,
        "validation_samples": 250,
        "hyperparameters": {
            "learning_rate": 0.001,
            "max_depth": 8,
            "regularization": 0.1,
            "n_features": 15,
            "cv_folds": 5
        },
        "top_performers": [
            {"symbol": "AAPL", "accuracy": 0.89, "predictions": 45},
            {"symbol": "GOOGL", "accuracy": 0.85, "predictions": 38},
            {"symbol": "MSFT", "accuracy": 0.82, "predictions": 41}
        ],
        "worst_performers": [
            {"symbol": "ROKU", "accuracy": 0.45, "predictions": 12},
            {"symbol": "NFLX", "accuracy": 0.52, "predictions": 18}
        ],
        "performance_metrics": {
            "sharpe_ratio": 1.24,
            "max_drawdown": 0.08,
            "win_rate": 0.72,
            "avg_return": 0.085
        }
    },
    "session_id": "test_$(date +%Y%m%d_%H%M%S)",
    "source": "manual-test"
}
EOF
    )

    # Invoke the function
    print_status "Invoking tuning reporter with test data..."
    local response=$(aws lambda invoke \
        --function-name "model-tuning-reporter" \
        --payload "$test_payload" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" \
        /tmp/tuning-report-response.json 2>&1)

    if [[ $? -eq 0 ]]; then
        local status_code=$(echo "$response" | jq -r '.StatusCode // 200')
        if [[ "$status_code" == "200" ]]; then
            print_success "Tuning report triggered successfully"
            local result=$(cat /tmp/tuning-report-response.json | jq -r '.body // "{}"' | jq -r '.message // "No message"')
            print_status "Response: $result"
            return 0
        else
            print_error "Function returned status code: $status_code"
            cat /tmp/tuning-report-response.json
            return 1
        fi
    else
        print_error "Failed to invoke tuning reporter"
        echo "$response"
        return 1
    fi
}

test_price_model_tuning() {
    print_status "Testing price model tuning function..."

    local test_payload=$(cat <<EOF
{
    "action": "analyze_performance",
    "lookback_days": 7,
    "test_mode": true
}
EOF
    )

    print_status "Invoking price model tuning..."
    local response=$(aws lambda invoke \
        --function-name "price-model-tuning" \
        --payload "$test_payload" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" \
        /tmp/price-tuning-response.json 2>&1)

    if [[ $? -eq 0 ]]; then
        local status_code=$(echo "$response" | jq -r '.StatusCode // 200')
        if [[ "$status_code" == "200" ]]; then
            print_success "Price model tuning completed"
            return 0
        else
            print_warning "Price model tuning returned status: $status_code"
            return 1
        fi
    else
        print_error "Failed to invoke price model tuning"
        echo "$response"
        return 1
    fi
}

test_eventbridge_rules() {
    print_status "Testing EventBridge rules..."

    local rules=("weekly-price-model-tuning" "weekly-time-model-tuning")
    local all_good=true

    for rule in "${rules[@]}"; do
        local response=$(aws events describe-rule \
            --name "$rule" \
            --profile "$AWS_PROFILE" \
            --region "$REGION" 2>/dev/null)

        if [[ $? -eq 0 ]]; then
            local state=$(echo "$response" | jq -r '.State')
            local schedule=$(echo "$response" | jq -r '.ScheduleExpression')
            if [[ "$state" == "ENABLED" ]]; then
                print_success "Rule '$rule' is enabled (Schedule: $schedule)"
            else
                print_warning "Rule '$rule' is disabled"
                all_good=false
            fi
        else
            print_error "Rule '$rule' not found"
            all_good=false
        fi
    done

    if [[ "$all_good" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

test_sns_subscription() {
    print_status "Testing SNS subscription for reports..."

    local subscriptions=$(aws sns list-subscriptions \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null | \
        jq -r '.Subscriptions[] | select(.Protocol == "email" and (.Endpoint | contains("cdameworth@gmail.com"))) | .SubscriptionArn')

    if [[ -n "$subscriptions" ]]; then
        print_success "Email subscription found for cdameworth@gmail.com"
        echo "$subscriptions" | while read -r arn; do
            print_status "  Subscription: $arn"
        done
        return 0
    else
        print_warning "No email subscription found for cdameworth@gmail.com"
        return 1
    fi
}

check_recent_tuning_logs() {
    print_status "Checking recent tuning logs..."

    local functions=("price-model-tuning" "time-model-tuning" "model-tuning-reporter")

    for func in "${functions[@]}"; do
        print_status "Checking logs for $func..."
        local log_group="/aws/lambda/$func"

        # Get recent log events
        local events=$(aws logs filter-log-events \
            --log-group-name "$log_group" \
            --start-time $(($(date +%s) - 86400))000 \
            --profile "$AWS_PROFILE" \
            --region "$REGION" 2>/dev/null | \
            jq -r '.events[] | .message' | tail -3)

        if [[ -n "$events" ]]; then
            print_success "Recent logs found for $func:"
            echo "$events" | sed 's/^/    /'
        else
            print_warning "No recent logs for $func (may not have run recently)"
        fi
    done
}

generate_summary_report() {
    print_status "Generating summary report..."

    echo ""
    echo "=============================================="
    echo "🤖 MODEL TUNING REPORT SYSTEM TEST SUMMARY"
    echo "=============================================="
    echo ""
    echo "📋 Components Tested:"
    echo "  ✓ Model Tuning Reporter Lambda Function"
    echo "  ✓ Price Model Tuning Integration"
    echo "  ✓ EventBridge Scheduling Rules"
    echo "  ✓ SNS Email Notifications"
    echo "  ✓ Manual Report Trigger"
    echo ""
    echo "📅 Scheduling Information:"
    echo "  • Price Model Tuning: Sundays at 3 AM UTC"
    echo "  • Time Model Tuning: Sundays at 4 AM UTC"
    echo "  • Email Reports: Sent after each tuning run"
    echo "  • Email Address: cdameworth@gmail.com"
    echo ""
    echo "🔄 Next Steps:"
    echo "  1. Wait for next Sunday's automatic tuning runs"
    echo "  2. Check email for detailed tuning reports"
    echo "  3. Monitor CloudWatch logs for execution status"
    echo "  4. Review DynamoDB tuning history table"
    echo ""
    echo "💡 Manual Triggering:"
    echo "  • Use this script to test reports anytime"
    echo "  • Invoke functions directly via AWS CLI"
    echo "  • Check infrastructure/model_tuning_reports.tf for config"
    echo ""
}

# Main execution
main() {
    echo "================================================"
    echo "🤖 Stock Analytics - Tuning Report System Test"
    echo "================================================"
    echo ""

    local tests_passed=0
    local total_tests=5

    # Run tests
    if test_tuning_reporter_deployment; then
        ((tests_passed++))
    fi

    if test_manual_report_trigger; then
        ((tests_passed++))
    fi

    if test_price_model_tuning; then
        ((tests_passed++))
    fi

    if test_eventbridge_rules; then
        ((tests_passed++))
    fi

    if test_sns_subscription; then
        ((tests_passed++))
    fi

    # Check logs
    check_recent_tuning_logs

    # Generate summary
    generate_summary_report

    # Final status
    echo "=============================================="
    if [[ $tests_passed -eq $total_tests ]]; then
        print_success "All tests passed! ($tests_passed/$total_tests)"
        echo "✅ Model tuning report system is ready!"
    else
        print_warning "Some tests failed ($tests_passed/$total_tests passed)"
        echo "⚠️  Review failed components before deployment"
    fi
    echo "=============================================="

    # Cleanup
    rm -f /tmp/tuning-report-response.json /tmp/price-tuning-response.json
}

# Run main function with all arguments
main "$@"