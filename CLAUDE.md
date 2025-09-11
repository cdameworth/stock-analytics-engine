# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Stock Analytics Engine is a serverless AWS platform for stock analysis with advanced ML model tuning, designed to achieve market-beating performance through systematic optimization and evidence-based validation.

### Core Infrastructure
- **Lambda Functions**: Data ingestion, ML inference, dual prediction models, analytics reporting
- **API Gateway**: RESTful endpoints with API key authentication 
- **DynamoDB**: Stock recommendations and analytics storage with TTL cleanup
- **S3**: Data lake for historical data and ML model artifacts
- **Aurora PostgreSQL**: High-performance database cluster
- **ElastiCache Valkey**: Redis-compatible caching layer
- **EventBridge**: Automated scheduling for data collection and model optimization

### Lambda Function Architecture
The system uses multiple specialized Lambda functions:
- `stock_data_ingestion.py`: Market data collection from Alpha Vantage API
- `stock_recommendations_api.py`: Main API handler for recommendations
- `ml_model_inference.py`: ML model predictions and inference
- `dual_accuracy_tracker.py`: Performance monitoring and analytics
- `dual_prediction_reporting_api.py`: Analytics dashboard API
- `price_prediction_model.py` / `time_to_hit_predictor_slim.py`: Dual prediction models
- Various tuning services for model optimization

## Development Commands

### Infrastructure Management
```bash
# Use the stock-analytics-admin AWS CLI profile
export AWS_PROFILE=stock-analytics-admin

# Navigate to infrastructure directory
cd infrastructure

# Initialize Terraform
terraform init

# Deploy infrastructure (production tier)
terraform apply -var-file="terraform-tier1.tfvars" -auto-approve

# Deploy specific modules
terraform apply -target="module.deploy_advanced_tuning" -auto-approve

# Get deployment outputs
terraform output api_endpoints
terraform output api_key_value
```

### Testing
```bash
# Run comprehensive test suite
python3 run_tests.py

# Test individual components
python3 tests/test_framework.py
```

### API Testing
```bash
# Test main recommendations API
curl -H "x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq" \
     "https://2cqomr4nb2.execute-api.us-east-1.amazonaws.com/prod/recommendations"

# Test analytics dashboard
curl -H "x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq" \
     "https://2cqomr4nb2.execute-api.us-east-1.amazonaws.com/prod/analytics/dashboard"
```

### Lambda Function Management
```bash
# Trigger model tuning manually
aws lambda invoke --function-name advanced-model-tuning-service \
  --payload '{"action":"comprehensive_tuning","lookback_days":90}' \
  --profile stock-analytics-admin response.json

# Monitor Lambda logs
aws logs tail /aws/lambda/stock-data-ingestion --follow --profile stock-analytics-admin
aws logs tail /aws/lambda/ml-model-inference-tier --follow --profile stock-analytics-admin
```

## Key Configuration Files

### Terraform Configuration (in `infrastructure/` directory)
- `main.tf`: Core infrastructure definitions
- `variables.tf`: Configurable parameters and defaults
- `outputs.tf`: API endpoints and resource information
- `api_key_auth.tf`: API Gateway authentication setup
- `deploy_dual_prediction_system.tf`: Dual prediction system infrastructure
- `reporting_scheduler.tf`: Automated reporting system
- `terraform-tier1.tfvars`: Production configuration (~$245/month)
- `terraform-current.tfvars`: Current deployment settings

### Lambda Dependencies (in `lambda_requirements/` directory)
- `requirements_pandas.txt`: Python dependencies for data analysis
- `requirements_simple.txt`: Minimal requirements for basic functions
- `requirements_dual.txt`: Dependencies for dual prediction system

### Project Organization
- `📁 infrastructure/`: All Terraform configuration files
- `📁 lambda_functions/`: Python Lambda function source code
- `📁 lambda_requirements/`: Lambda dependency specifications
- `📁 scripts/`: Shell scripts and automation utilities
- `📁 docs/`: Project documentation and guides
- `📁 tests/`: Test framework and validation scripts

## Authentication & API Access

### API Key Authentication
All API endpoints require the `x-api-key` header:
```
x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq
```

### API Endpoints
- `GET /recommendations` - All stock recommendations
- `GET /recommendations/{symbol}` - Specific symbol recommendation
- `GET /dual-predictions/analytics` - Dual prediction analytics
- `GET /analytics/dashboard` - Performance dashboard

## Data Flow & Scheduling

### Automated Data Collection
- **Market Hours**: Every 5 minutes (9 AM - 4 PM EST) via EventBridge
- **Evening Processing**: Every 10 minutes (5-11 PM EST) for extended coverage
- **Model Optimization**: Weekly comprehensive tuning (Sundays 2 AM)
- **Daily Analytics**: Performance validation (6 AM EST daily)

### ML Model Pipeline
1. Data ingestion → S3 storage → DynamoDB caching
2. ML inference triggered by EventBridge patterns
3. Dual prediction models (price + time-to-hit)
4. Performance tracking and analytics reporting
5. Automated model retraining based on performance thresholds

## Performance Targets & Monitoring

### AI Performance Goals
- **Hit Rate**: 65% accuracy threshold for deployment
- **Market Outperformance**: 3% above S&P 500 benchmark
- **Risk Management**: Sharpe ratio >1.0, max drawdown <15%

### Monitoring Infrastructure  
- **CloudWatch Dashboards**: AI performance metrics, API usage, infrastructure health
- **Cost Monitoring**: Alerts when monthly costs exceed $300
- **Performance Tracking**: Hit rate trends, confidence calibration, symbol-level performance

## Git Workflow Requirements

**CRITICAL**: Claude Code must follow these git workflow patterns for proper ops health:

### Commit Integration Pattern
- **After completing any task/todo**: Check for changes → Validate → Commit → Push
- **After successful tests/validation**: Always commit with test results
- **Before risky operations**: Create restore point commits  
- **Every 30-45 minutes**: Commit incremental progress during active development
- **Before switching contexts**: Commit current state

### Validation Sequence (Required Before Each Commit)
```bash
git status              # Review pending changes
python3 run_tests.py   # Run tests (if available)  
git diff               # Review actual changes
git add -A             # Stage all changes
git commit -m "..."    # Commit with standard message format
git push origin main   # Push to remote
```

### Commit Message Format (Required)
```
<type>(<scope>): <description>

[optional body with bullet points]

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**: feat, fix, refactor, test, docs, config, deploy
**Scopes**: lambda, infrastructure, api, ml, tests, docs

### Git Triggers (When to Commit)
- 🔴 **Always**: After tests pass, before risky ops, after validation gates
- 🟡 **Usually**: After task completion, bug fixes, feature additions  
- 🟢 **Often**: After cleanup, documentation updates, logical breakpoints

## Important Development Notes

### AWS Profile Requirement
Always use the `stock-analytics-admin` AWS CLI profile when working with infrastructure:
```bash
export AWS_PROFILE=stock-analytics-admin
# or
aws --profile stock-analytics-admin [command]
```

### Cost Optimization
Current deployment (Tier 1) costs ~$245/month with:
- Aurora PostgreSQL: db.r5.large cluster
- ElastiCache Valkey: 3x cache.r6g.large nodes  
- Lambda: 2048MB memory, 300s timeout
- Premium Alpha Vantage API: 75 calls/minute

### Database Architecture
- **DynamoDB**: `stock-recommendations` table with GSI for symbol-based queries
- **Aurora PostgreSQL**: High-performance analytics and historical data
- **ElastiCache Valkey**: Redis-compatible caching for API responses

### Security Considerations
- API keys stored in AWS Secrets Manager
- All Lambda functions run in private subnets with VPC configuration
- Database connections secured through security groups
- S3 buckets with lifecycle policies for cost management

## Troubleshooting Common Issues

### Lambda Function Debugging
1. Check CloudWatch logs for specific function errors
2. Verify environment variables are properly set
3. Ensure IAM permissions for cross-service access
4. Monitor memory and timeout limits

### API Gateway Issues
1. Verify API key is correctly configured in requests
2. Check CORS headers for browser-based requests
3. Monitor CloudWatch logs for API Gateway access patterns
4. Validate request/response transformations

### Infrastructure Deployment
1. Ensure AWS credentials and profile are properly configured
2. Check Terraform state consistency
3. Verify all required variables are set in tfvars files
4. Monitor AWS service limits and quotas