# Week 3 Advanced Observability Deployment Guide

## Overview

Week 3 introduces advanced observability intelligence features that build upon the Week 1-2 implementation:

- **Performance monitoring** with ML accuracy tracking and business intelligence
- **Trading intelligence** for market opportunity analysis
- **Dynamic sampling optimization** based on performance metrics
- **SigNoz dashboard integration** with custom financial analytics dashboards

## Architecture Changes

### New Components Added

```
📁 lambda_functions/shared/
├── observability_intelligence.py    # Performance monitoring & alerting
├── signoz_integration.py           # SigNoz dashboards & queries
└── week3_observability_demo.py     # Demo and validation script

📁 Enhanced Components:
├── dual_accuracy_tracker.py        # Week 3 performance intelligence
├── business_tracing.py             # (Week 2 - unchanged)
└── market_utils.py                 # (Week 2 - unchanged)
```

### Key Features

#### 1. Performance Monitoring (`observability_intelligence.py`)
- **PerformanceMonitor**: Real-time ML accuracy tracking with business context
- **DynamicSamplingOptimizer**: Cost-aware sampling rate optimization
- **TradingIntelligence**: Market opportunity analysis from trace data
- **Automated alerting** for performance degradation

#### 2. SigNoz Integration (`signoz_integration.py`)
- **ML Performance Dashboard**: Model accuracy, confidence calibration, symbol rankings
- **Trading Signals Dashboard**: Signal quality, profitability analysis
- **Market Overview Dashboard**: Real-time market session analytics
- **Cost Optimization Dashboard**: Trace volume and sampling efficiency

#### 3. Enhanced Accuracy Tracking (`dual_accuracy_tracker.py`)
- **Advanced ML accuracy tracking** with confidence correlation
- **Trading signal quality monitoring**
- **Market opportunity identification** from validation results
- **Performance-driven optimization feedback loops**

## Deployment Steps

### 1. Lambda Layer Updates

Update the OpenTelemetry layer with Week 3 dependencies:

```bash
cd infrastructure/

# Add Week 3 Python dependencies to OTEL layer
cat >> otel-layer-requirements.txt << EOF
# Week 3 Advanced Observability
statistics>=1.0.3.5
dataclasses-json>=0.5.7
numpy>=1.21.0
EOF

# Rebuild OTEL layer
./create-otel-layer.sh
```

### 2. Environment Variables

Add Week 3 configuration to Lambda functions:

```bash
# Update Terraform with Week 3 environment variables
export TF_VAR_week3_features_enabled="true"
export TF_VAR_performance_monitoring_enabled="true"
export TF_VAR_signoz_integration_enabled="true"

# Apply infrastructure changes
terraform apply -var-file="terraform-tier1.tfvars" -auto-approve
```

### 3. SigNoz Dashboard Import

```bash
# Generate dashboard configurations
python3 lambda_functions/week3_observability_demo.py

# Import to SigNoz (manual step in SigNoz UI)
# 1. Go to SigNoz Dashboards > Import
# 2. Upload /tmp/signoz_dashboards.json
# 3. Configure data source and refresh intervals
```

### 4. Validation Testing

```bash
# Test Week 3 features
aws lambda invoke \
    --function-name week3-observability-demo \
    --payload '{"demo_type": "full_demo"}' \
    --profile stock-analytics-admin \
    response.json

# Test enhanced accuracy tracking
aws lambda invoke \
    --function-name dual-accuracy-tracker \
    --payload '{"action": "validate_all", "lookback_days": 7}' \
    --profile stock-analytics-admin \
    accuracy_response.json

# Check trace data in SigNoz
# Query: service.name = "stock_analytics" AND operation_name LIKE "%performance%"
```

## Expected Outcomes

### 1. Enhanced Observability
- **ML accuracy tracking** with 95%+ confidence calibration monitoring
- **Real-time performance alerts** for model degradation (< 60% accuracy)
- **Business-aware sampling** reducing trace volume by 30-50% without losing insights

### 2. Trading Intelligence
- **Market opportunity scoring** with 70+ confidence threshold identification
- **Symbol performance rankings** updated every 15 minutes during market hours
- **Market session correlation** analysis for optimal trading timing

### 3. Cost Optimization
- **Dynamic sampling rates** adjusted based on model performance
- **Intelligent trace reduction** prioritizing high-value financial events
- **Cost monitoring** with automated alerts above $60/day tracing costs

### 4. SigNoz Dashboards
- **4 custom dashboards** with 15+ financial-specific widgets
- **Real-time market analytics** with 30-second refresh rates
- **Historical trend analysis** with 90-day lookback capabilities

## Monitoring & Validation

### Performance Metrics to Track

```sql
-- ML Model Accuracy Trend (SigNoz Query)
SELECT
    toStartOfInterval(timestamp, INTERVAL 1 hour) as hour,
    avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as accuracy
FROM signoz_traces.distributed_signoz_index_v2
WHERE attribute_ml_accuracy_price_error_pct != ''
AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY hour ORDER BY hour;

-- Trading Opportunity Detection
SELECT
    attribute_finance_symbol as symbol,
    max(toFloat64(attribute_opportunity_score)) as max_score
FROM signoz_traces.distributed_signoz_index_v2
WHERE spanName = 'trading.opportunity_analysis'
AND timestamp >= now() - INTERVAL 1 HOUR
GROUP BY symbol HAVING max_score > 70
ORDER BY max_score DESC;
```

### Key Success Indicators

1. **Model Performance**: Accuracy tracking shows 75%+ hit rate
2. **Cost Efficiency**: Trace volume reduced by 40% while maintaining quality
3. **Business Intelligence**: 5+ high-confidence opportunities identified daily
4. **System Health**: Performance alerts < 2 per day, resolution time < 15 minutes

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check OTEL layer deployment
aws lambda get-layer-version \
    --layer-name otel-layer \
    --version-number $(aws lambda list-layer-versions --layer-name otel-layer --query 'LayerVersions[0].Version' --output text) \
    --profile stock-analytics-admin

# Verify dependencies
aws lambda invoke \
    --function-name test-otel-layer \
    --payload '{"test": "import_check"}' \
    --profile stock-analytics-admin \
    import_test.json
```

#### 2. SigNoz Connection Issues
```bash
# Test SigNoz ingestion
export SIGNOZ_INGESTION_KEY="your-key"
python3 -c "
import requests
response = requests.post('https://ingest.signoz.io/v1/traces',
    headers={'Authorization': f'Bearer {SIGNOZ_INGESTION_KEY}'})
print(f'SigNoz connection: {response.status_code}')
"
```

#### 3. Performance Monitoring Not Working
```bash
# Check Lambda logs for Week 3 features
aws logs tail /aws/lambda/dual-accuracy-tracker --follow --profile stock-analytics-admin | grep -i "week3\|performance\|intelligence"

# Verify environment variables
aws lambda get-function-configuration \
    --function-name dual-accuracy-tracker \
    --profile stock-analytics-admin \
    --query 'Environment.Variables'
```

## Production Considerations

### 1. Cost Management
- **Trace sampling optimization**: Reduce sampling for non-critical periods
- **Dashboard refresh rates**: Balance real-time needs vs. query costs
- **Storage retention**: Configure 30-day trace retention for cost control

### 2. Performance Impact
- **Async processing**: Performance monitoring runs asynchronously
- **Batch operations**: Group multiple accuracy validations per invocation
- **Circuit breakers**: Disable advanced features if basic monitoring fails

### 3. Security & Compliance
- **No PII in traces**: Financial data only (prices, symbols, confidence scores)
- **API key rotation**: SigNoz ingestion keys rotated every 90 days
- **Access control**: Dashboard access limited to trading team members

## Next Steps

After successful Week 3 deployment:

1. **Monitor dashboards** for 7 days to establish baselines
2. **Tune sampling rates** based on actual cost and quality metrics
3. **Add custom alerts** for business-specific thresholds
4. **Train team** on new observability features and dashboard usage
5. **Document** any custom queries or dashboard modifications

## Support & Resources

- **Week 3 Demo Script**: `lambda_functions/week3_observability_demo.py`
- **SigNoz Documentation**: https://signoz.io/docs/
- **OpenTelemetry Python**: https://opentelemetry.io/docs/instrumentation/python/
- **AWS Lambda Observability**: https://docs.aws.amazon.com/lambda/latest/dg/monitoring-insights.html

---

**Week 3 Implementation Complete** ✅

Advanced observability intelligence is now active with:
- Performance monitoring and alerting
- Trading intelligence and market opportunity analysis
- Dynamic sampling optimization
- Custom SigNoz dashboards for financial analytics