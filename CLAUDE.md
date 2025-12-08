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
- `ml_model_inference.py` / `ml_inference_native.py`: ML model predictions and inference
- `dual_accuracy_tracker.py`: Performance monitoring and analytics
- `dual_prediction_reporting_api.py`: Analytics dashboard API
- `price_prediction_model.py` / `time_to_hit_predictor_slim.py`: Dual prediction models
- `price_model_tuning.py` / `time_model_tuning.py`: Model optimization services
- `model_tuning_reporter.py`: Tuning performance reporting

### Shared Utilities Architecture
Centralized utilities in `lambda_functions/shared/` provide reusable functionality:
- `config.py`: Centralized configuration management with environment-based settings
- `lambda_utils.py`: Lambda response formatting, input validation, environment configuration
- `types.py`: Type definitions for predictions, analytics, and API responses
- `error_handling.py`: Structured exception handling and logging
- `security.py`: Input sanitization, API key validation, secure token generation
- `ml_utils.py`: Machine learning utilities and model management
- `performance_optimization.py`: In-memory caching, connection pooling, batch processing
- `observability.py`: CloudWatch metrics, structured logging, health checks
- `business_tracing.py`: Business-level tracing for model performance and data flow
- `signoz_integration.py`: OpenTelemetry integration for distributed tracing
- `market_utils.py`: Market data utilities and trading day calculations

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
# Run comprehensive test suite with all validations
python3 run_tests.py

# Test individual components (unittest or pytest)
python3 tests/test_framework.py
python3 -m pytest tests/ -v

# Quick syntax check only
python3 -m py_compile lambda_functions/**/*.py
```

### OpenTelemetry & Observability
```bash
# Check current OTEL status for all Lambda functions
./scripts/fix-otel-lambda-issues.sh --check-only

# Fix OTEL Lambda issues and deploy SigNoz integration
export TF_VAR_signoz_ingestion_key="your-signoz-ingestion-key"
./scripts/fix-otel-lambda-issues.sh

# Plan OTEL changes without applying
./scripts/fix-otel-lambda-issues.sh --plan-only
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
# Trigger model tuning manually (price model)
aws lambda invoke --function-name price-model-tuning \
  --payload '{"lookback_days":90}' \
  --profile stock-analytics-admin response.json

# Trigger time model tuning
aws lambda invoke --function-name time-model-tuning \
  --payload '{"lookback_days":90}' \
  --profile stock-analytics-admin response.json

# Monitor Lambda logs
aws logs tail /aws/lambda/stock-data-ingestion --follow --profile stock-analytics-admin
aws logs tail /aws/lambda/ml-model-inference-lowcost --follow --profile stock-analytics-admin
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

### Observability Infrastructure
- `aws-native-observability.tf`: CloudWatch dashboards, X-Ray tracing, native AWS monitoring
- `distributed-tracing-configuration.tf`: X-Ray distributed tracing across Lambda functions
- `signoz-minimal-observability.tf`: SigNoz integration for advanced observability
- `lambda-otel-configuration.tf`: OpenTelemetry configuration for Lambda functions

### Project Organization
- `📁 infrastructure/`: All Terraform configuration files
- `📁 lambda_functions/`: Python Lambda function source code
  - `📁 shared/`: Reusable utilities and common functionality
- `📁 lambda_requirements/`: Lambda dependency specifications
- `📁 scripts/`: Shell scripts and automation utilities (observability, deployment, testing)
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
- **Market Hours**: Every 5 minutes (9 AM - 4 PM EST) via EventBridge - Tier 1 only
- **Evening Processing**: Every 10 minutes (5-11 PM EST) for extended coverage
- **Model Optimization**: Weekly comprehensive tuning (Sundays 2 AM)
- **Daily Analytics**: Performance validation (6 AM EST daily via `ai_validation_schedule`)

### EventBridge Schedule Expressions
Schedules are defined in Terraform and use cron expressions:
```
# Market hours: Every 5 minutes, 9 AM - 4 PM EST (14:00-21:00 UTC), Mon-Fri
cron(*/5 14-21 ? * MON-FRI *)

# Daily validation: 6 AM EST (11:00 UTC)
cron(0 11 * * ? *)

# Weekly tuning: Sundays 2 AM EST (7:00 UTC)
cron(0 7 ? * SUN *)
```

### ML Model Pipeline
1. **Data Ingestion**: `stock_data_ingestion.py` fetches from Alpha Vantage → stores in S3 + DynamoDB
2. **ML Inference**: EventBridge triggers → `ml_model_inference.py` / `ml_inference_native.py`
3. **Dual Predictions**:
   - `price_prediction_model.py`: Predicts target price
   - `time_to_hit_predictor_slim.py`: Predicts time to reach target
4. **Performance Tracking**: `dual_accuracy_tracker.py` validates predictions against actual prices
5. **Analytics Reporting**: `dual_prediction_reporting_api.py` exposes metrics via API
6. **Model Tuning**: `price_model_tuning.py` / `time_model_tuning.py` optimize hyperparameters
7. **Deployment Gate**: Only deploy models achieving 65% hit rate + 3% market outperformance

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

## Deployment Options

### AWS Lambda (Current Production)
Serverless architecture using AWS Lambda, API Gateway, DynamoDB, and EventBridge.
- **Cost**: ~$245/month
- **Deployment**: Terraform (`infrastructure/`)
- **Guide**: See Infrastructure Management section below

### Railway (Alternative Platform)
Containerized microservices deployment on Railway platform.
- **Cost**: ~$51-85/month (65-79% savings)
- **Deployment**: Docker containers (`railway/`)
- **Guide**: See [railway/RAILWAY_DEPLOYMENT.md](railway/RAILWAY_DEPLOYMENT.md)
- **Architecture**: 3 services (API, Data Ingestion, Model Tuning)

## Important Development Notes

### AWS Profile Requirement
Always use the `stock-analytics-admin` AWS CLI profile when working with infrastructure:
```bash
export AWS_PROFILE=stock-analytics-admin
# or
aws --profile stock-analytics-admin [command]
```

### Code Architecture Patterns

#### Shared Utilities Usage
When adding new Lambda functions, always leverage the shared utilities:
```python
# Standard imports for Lambda functions
from lambda_functions.shared.config import get_config, FeatureFlags
from lambda_functions.shared.lambda_utils import LambdaResponse, InputValidator
from lambda_functions.shared.error_handling import StructuredLogger, ValidationError
from lambda_functions.shared.security import InputSanitizer
from lambda_functions.shared.observability import MetricsCollector

# Example Lambda handler pattern
def lambda_handler(event, context):
    logger = StructuredLogger(__name__)
    config = get_config()
    metrics = MetricsCollector("function_name")

    try:
        # Validate input
        validator = InputValidator(event)
        symbol = validator.get_required('symbol')

        # Business logic here

        return LambdaResponse.success(data)
    except ValidationError as e:
        logger.log_error("Validation failed", error=e)
        return LambdaResponse.error(str(e), 400)
```

#### Configuration Management
Use centralized configuration instead of environment variables directly:
```python
# ❌ Don't do this
table_name = os.environ.get('RECOMMENDATIONS_TABLE')

# ✅ Do this instead
from lambda_functions.shared.config import get_config
config = get_config()
table_name = config.database.recommendations_table
```

#### Feature Flags
Check feature flags before using optional functionality:
```python
from lambda_functions.shared.config import FeatureFlags

if FeatureFlags.is_caching_enabled():
    # Use caching
    pass

if FeatureFlags.is_dual_prediction_enabled():
    # Use dual prediction system
    pass
```

### Cost Optimization
Current deployment (Tier 1) costs ~$245/month with:
- Aurora PostgreSQL: db.r5.large cluster
- ElastiCache Valkey: 3x cache.r6g.large nodes
- Lambda: 2048MB memory, 300s timeout
- Premium Alpha Vantage API: 75 calls/minute

#### Deployment Tiers
Available Terraform configurations:
- `terraform-tier1.tfvars`: Production (~$245/month) - db.r5.large, 3x cache.r6g.large, premium API
- `terraform-tier2.tfvars`: Mid-tier configuration
- `terraform-tier3.tfvars`: High-performance configuration
- `terraform-free-tier.tfvars`: Free tier development environment
- `terraform-current.tfvars`: Current active deployment

### Database Architecture
- **DynamoDB Tables**:
  - `stock-recommendations`: Main recommendations with GSI for symbol queries
  - `ai-performance-analytics`: Model performance tracking
  - `price-predictions`: Price prediction history
  - `time-to-hit-predictions`: Time-to-target predictions
- **Aurora PostgreSQL**: High-performance analytics and historical data (optional, Tier 1+)
- **ElastiCache Valkey**: Redis-compatible caching for API responses (optional, Tier 1+)

### Security Considerations
- API keys stored in AWS Secrets Manager (both legacy and premium keys)
- All Lambda functions can run in VPC configuration for database access
- Database connections secured through security groups
- S3 buckets with lifecycle policies for cost management
- Input sanitization through `security.py` shared module

## Troubleshooting Common Issues

### Lambda Function Debugging
1. Check CloudWatch logs for specific function errors:
   ```bash
   aws logs tail /aws/lambda/FUNCTION_NAME --follow --profile stock-analytics-admin
   ```
2. Verify environment variables are properly set in Terraform Lambda resources
3. Ensure IAM permissions for cross-service access (DynamoDB, S3, Secrets Manager)
4. Monitor memory and timeout limits (default: 2048MB memory, 300s timeout)
5. Check shared utilities imports if seeing ModuleNotFoundError
6. Validate configuration using `get_config().validate()` in shared utilities

### API Gateway Issues
1. Verify API key is correctly configured in requests (`x-api-key` header)
2. Check CORS headers for browser-based requests
3. Monitor CloudWatch logs for API Gateway access patterns
4. Validate request/response transformations
5. Confirm API Gateway deployment stage is active (prod)

### Infrastructure Deployment
1. Ensure AWS credentials and profile are properly configured:
   ```bash
   aws sts get-caller-identity --profile stock-analytics-admin
   ```
2. Check Terraform state consistency:
   ```bash
   cd infrastructure
   terraform state list
   terraform plan -var-file="terraform-current.tfvars"
   ```
3. Verify all required variables are set in tfvars files (especially `alpha_vantage_api_key`)
4. Monitor AWS service limits and quotas
5. For Lambda deployment issues, check that Lambda layer dependencies are properly built

### Observability Issues
1. **X-Ray Tracing Not Working**:
   - Verify X-Ray daemon is enabled in Lambda configuration
   - Check IAM permissions include `xray:PutTraceSegments`
   - Use distributed tracing utilities from `business_tracing.py`

2. **SigNoz Integration Issues**:
   - Verify SigNoz ingestion key is set: `TF_VAR_signoz_ingestion_key`
   - Run connectivity test: `./scripts/fix-otel-lambda-issues.sh --check-only`
   - Check OpenTelemetry layer configuration in `lambda-otel-configuration.tf`

3. **CloudWatch Metrics Missing**:
   - Verify MetricsCollector is being used in Lambda functions
   - Check CloudWatch IAM permissions
   - Confirm custom metrics are properly namespaced

### Testing Issues
1. **Import Errors in Tests**:
   - Ensure `lambda_functions/` is in Python path
   - Install required dependencies: `pip3 install boto3 pytest`
   - Set required environment variables for testing (see `run_tests.py`)

2. **Test Framework Failures**:
   - Run with verbose output: `python3 run_tests.py`
   - Check individual test categories (syntax, imports, security, etc.)
   - Verify test environment variables are set correctly