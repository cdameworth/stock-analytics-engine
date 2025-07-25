# Stock Analytics Engine - Overwatch Demo Application

A comprehensive AWS-based stock analysis and ML recommendation engine that provides real-time market intelligence through a fully observable infrastructure.

## 🏗️ Architecture Overview

This application demonstrates a production-ready microservices architecture for AI-powered stock analysis using AWS services with comprehensive observability for the Overwatch infrastructure monitoring platform.

### Core Components

- **Lambda Functions**: Serverless data processing and ML inference
- **SageMaker**: ML model hosting and inference endpoints
- **API Gateway**: RESTful API for stock recommendations
- **RDS Aurora**: PostgreSQL cluster for structured data
- **DynamoDB**: NoSQL storage for recommendations and caching  
- **ElastiCache Redis**: High-performance caching layer
- **S3**: Data lake for raw stock data and ML models
- **EventBridge**: Scheduled data ingestion (every 15 minutes)
- **CloudWatch**: Comprehensive logging and monitoring
- **X-Ray**: Distributed tracing
- **SNS/SQS**: Asynchronous messaging and dead letter queues

## 📊 Data Flow

1. **Data Ingestion**: EventBridge triggers Lambda function every 15 minutes
2. **Alpha Vantage API**: Fetches real-time stock data for major indexes and popular stocks
3. **Data Processing**: Lambda processes and stores data in S3 data lake
4. **ML Inference**: SageMaker endpoint analyzes market conditions and generates predictions
5. **Recommendations**: Results stored in DynamoDB with Redis caching
6. **API Serving**: API Gateway provides REST endpoints for consuming applications

## 🔧 Configuration

### Environment Variables

```bash
# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=YFT4NTLIWG9Z05LA

# AWS Configuration
AWS_REGION=us-east-1

# Lambda Environment Variables
S3_BUCKET=stock-analytics-data-lake-${random_suffix}
REDIS_ENDPOINT=${redis_cluster_address}
DYNAMODB_TABLE=stock-recommendations
SAGEMAKER_ENDPOINT=stock-prediction-endpoint
RDS_ENDPOINT=${aurora_cluster_endpoint}
```

### Terraform Deployment

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="aws_region=us-east-1" -var="environment=production"

# Deploy infrastructure
terraform apply -auto-approve

# Get API Gateway URL
terraform output api_gateway_url
```

## 📈 API Endpoints

### Get Stock Recommendations
```http
GET /recommendations?type=BUY&risk=LOW&limit=10&min_confidence=0.7
```

### Get Recommendation by Symbol
```http
GET /recommendations/AAPL?include_history=true
```

### Health Check
```http
GET /health
```

## 🎯 Key Features

### Stock Data Processing
- **Real-time Data**: Fetches data from Alpha Vantage API every 15 minutes
- **Market Coverage**: Major indexes (SPY, QQQ, IWM, DIA, VTI) and popular stocks
- **Technical Analysis**: Moving averages, volatility calculations, volume analysis
- **Data Lake**: All raw data stored in S3 for historical analysis

### ML-Powered Recommendations
- **SageMaker Integration**: Production ML model hosted on managed endpoints  
- **Feature Engineering**: Technical indicators, market sentiment, correlation analysis
- **Prediction Confidence**: Each recommendation includes confidence scoring
- **Fallback Logic**: Rule-based recommendations when ML service is unavailable

### High Availability & Performance
- **Multi-AZ Deployment**: Aurora and ElastiCache deployed across availability zones
- **Auto Scaling**: Lambda functions scale automatically with demand
- **Caching Strategy**: Redis for API responses, DynamoDB for structured data
- **Circuit Breakers**: Graceful degradation when external services fail

### Comprehensive Observability
- **CloudWatch Dashboards**: Real-time metrics for all services
- **Custom Metrics**: Business metrics (recommendation count, accuracy, API latency)
- **Distributed Tracing**: X-Ray integration across all Lambda functions
- **Alerting**: SNS notifications for critical issues and performance degradation
- **Log Aggregation**: Centralized logging with structured JSON format

## 📊 Monitoring & Observability

### CloudWatch Metrics
- Lambda duration, errors, and invocations
- API Gateway request count, latency, and error rates  
- DynamoDB consumed capacity and throttling
- SageMaker endpoint invocations and model latency
- Custom business metrics (recommendations generated, ML accuracy)

### Alarms & Notifications
- Lambda error rate > 5% triggers SNS alert
- API Gateway 4XX errors > 10/5min triggers investigation
- SageMaker endpoint failures trigger fallback mode
- RDS performance insights for database optimization

### Performance Optimization
- Lambda memory and timeout tuning based on CloudWatch insights
- DynamoDB performance indexes for efficient queries
- S3 lifecycle policies for cost optimization
- Redis cluster monitoring for cache hit rates

## 🚀 Deployment Guide

### Prerequisites
- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- Alpha Vantage API key

### Infrastructure Deployment
```bash
# Clone repository
git clone <repository-url>
cd stock-analytics-engine

# Set variables
export TF_VAR_alpha_vantage_api_key="YFT4NTLIWG9Z05LA"
export TF_VAR_aws_region="us-east-1"
export TF_VAR_environment="production"

# Deploy infrastructure
terraform init
terraform apply

# Package and deploy Lambda functions
zip -r stock_data_ingestion.zip lambda_functions/stock_data_ingestion.py
zip -r ml_model_inference.zip lambda_functions/ml_model_inference.py
zip -r stock_recommendations_api.zip lambda_functions/stock_recommendations_api.py

# Update Lambda function code
aws lambda update-function-code --function-name stock-data-ingestion --zip-file fileb://stock_data_ingestion.zip
aws lambda update-function-code --function-name ml-model-inference --zip-file fileb://ml_model_inference.zip
aws lambda update-function-code --function-name stock-recommendations-api --zip-file fileb://stock_recommendations_api.zip
```

### Testing
```bash
# Test API endpoint
curl "$(terraform output -raw api_gateway_url)/recommendations?limit=5"

# Test specific stock
curl "$(terraform output -raw api_gateway_url)/recommendations/AAPL"

# Monitor logs
aws logs tail /aws/lambda/stock-data-ingestion --follow
```

## 🔍 Troubleshooting

### Common Issues
1. **API Rate Limits**: Alpha Vantage free tier has 5 calls/minute limit
2. **Lambda Timeouts**: Increase timeout for ML inference function if needed
3. **SageMaker Cold Start**: First model invocation may take longer
4. **Redis Connection**: Ensure Lambda functions are in correct VPC subnets

### Debug Commands
```bash
# Check Lambda function logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/

# Monitor DynamoDB metrics
aws cloudwatch get-metric-statistics --namespace AWS/DynamoDB --metric-name ConsumedReadCapacityUnits

# Test SageMaker endpoint
aws sagemaker-runtime invoke-endpoint --endpoint-name stock-prediction-endpoint --body '{"instances":[[0.95,1.02,0.03,0.8,0.7,0.02,0.6]]}' output.json
```

## 📋 Cost Optimization

### Estimated Monthly Costs (Production)
- **Lambda**: ~$50 (15-minute intervals, 2880 invocations/month)
- **SageMaker**: ~$150 (ml.m5.large endpoint)
- **RDS Aurora**: ~$200 (2x db.r6g.large instances)
- **ElastiCache**: ~$180 (2x cache.r7g.large nodes)
- **API Gateway**: ~$10 (100K requests/month)
- **S3 + DynamoDB**: ~$20
- **Total**: ~$610/month

### Cost Optimization Strategies
- Use SageMaker Serverless Inference for lower traffic
- Implement DynamoDB on-demand billing
- Use Lambda Provisioned Concurrency only if needed
- Archive old S3 data to cheaper storage classes

## 🎯 Integration with Overwatch

This application serves as a comprehensive example for the Overwatch infrastructure monitoring platform, demonstrating:

- **Multi-service Architecture**: Complex AWS service interactions
- **Real-time Telemetry**: CloudWatch custom metrics and logs
- **Performance Monitoring**: Application and infrastructure metrics
- **Dependency Mapping**: Service-to-service communication patterns
- **Alert Management**: SNS integration for operational notifications
- **Cost Tracking**: Resource usage and optimization opportunities

The Stock Analytics Engine provides realistic ADM (Application Dependency Mapping) and telemetry data for testing Overwatch's visualization and monitoring capabilities across a production-grade microservices architecture.