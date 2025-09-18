# Stock Analytics Engine

AWS serverless stock analysis platform with advanced ML model tuning designed for market-beating performance.

## 🏗️ Architecture Overview

### Core Infrastructure
- **Lambda Functions**: Data ingestion, ML inference, and recommendations API
- **API Gateway**: Secured RESTful endpoints with API key authentication
- **DynamoDB**: Stock recommendations and AI performance analytics storage
- **S3**: Data lake and ML model artifacts
- **EventBridge**: Automated scheduling for data collection and model optimization
- **CloudWatch**: Monitoring with custom performance dashboards
- **Secrets Manager**: Secure API key management

### Advanced Model Tuning System
5-component Lambda architecture targeting **65% hit rate** and **3% market outperformance**:

- **Advanced Tuning Orchestrator**: Master coordinator for comprehensive model optimization
- **Backtesting Engine**: Walk-forward validation with S&P 500 benchmark comparison
- **Fundamental Data Enrichment**: P/E ratios, earnings growth, and sector rotation analysis
- **Ensemble Model Engine**: XGBoost, LightGBM, and Neural Network combination
- **Market Validation Engine**: Real-time benchmark compliance and competitive scoring

## 📊 Advanced Tuning Workflow

### Automated Optimization Schedule
- **Daily Assessment**: 8 AM EventBridge trigger for performance monitoring
- **Weekly Comprehensive Tuning**: Sunday 2 AM full optimization cycle
- **Manual Triggers**: On-demand optimization via Lambda invoke

### End-to-End Process
1. **Historical Validation**: Backtesting engine validates model performance using walk-forward analysis
2. **Feature Enhancement**: Fundamental data enrichment adds P/E ratios, earnings trends, sector rotation
3. **Ensemble Training**: Multi-algorithm training with XGBoost, LightGBM, and Neural Networks
4. **Market Benchmarking**: Validation against S&P 500, QQQ, and IWM performance
5. **Deployment Gate**: Only deploy models achieving 65% hit rate and 3% market outperformance

## 🔐 Security & Authentication

### API Key Authentication
- **Protected Endpoints**: All API endpoints require valid API key in `x-api-key` header
- **Usage Plans**: Rate limiting (100 req/sec, 200 burst, 10K/month quota)
- **API Key**: `lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq`

### Example Authenticated Request
```bash
curl -H "x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq" \
     "https://2cqomr4nb2.execute-api.us-east-1.amazonaws.com/prod/recommendations"
```

## 📈 API Endpoints

### Stock Recommendations
```http
GET /recommendations
GET /recommendations/{symbol}
Headers: x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq
```

### AI Performance Analytics
```http
GET /analytics/dashboard       # Overall performance metrics
GET /analytics/detailed        # Detailed analytics with trends  
GET /analytics/history         # Historical performance data
Headers: x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq
```

## 🚀 Deployment Configurations

### Tier 1: Production Stability (~$245/month)
- **Database**: db.r5.large with 7-day backup retention
- **Cache**: 3x cache.r6g.large Valkey clusters
- **Lambda**: 2048MB memory, 300s timeout
- **API**: Premium Alpha Vantage key (75 calls/min)
- **Features**: AI Analytics, automated model tuning, enhanced monitoring

```bash
terraform apply -var-file="terraform-tier1.tfvars" -auto-approve
```

### Key Tier 1 Features
- **Production Stability**: Disabled spot instances, enhanced monitoring
- **High Availability**: Multi-AZ deployment with 3 cache clusters
- **AI Analytics**: Full performance monitoring and model optimization
- **Enhanced Data Ingestion**: Every 2 hours during market + end-of-day
- **Premium API Access**: 150+ stocks across all major sectors

## 🧠 AI Performance Analytics Features

### Real-time Performance Monitoring
- **Hit Rate Analysis**: Tracks prediction accuracy against actual market movements
- **Time-to-Hit Calculations**: Measures how quickly predictions reach target prices
- **Confidence Calibration**: Analyzes correlation between prediction confidence and accuracy
- **Symbol Performance**: Per-stock performance tracking and optimization

### Automated Model Improvement
- **Hyperparameter Tuning**: GridSearchCV optimization using scikit-learn
- **A/B Testing**: Compare different model configurations
- **Performance Trending**: Track model improvement over time
- **Automated Retraining**: Triggered based on performance thresholds

### Analytics Infrastructure
- **DynamoDB Analytics Table**: Scalable storage with GSI for symbol-based queries
- **CloudWatch Integration**: Custom metrics and automated dashboards
- **Scheduled Automation**: Daily validation (6 AM UTC), weekly tuning (Sundays 2 AM UTC)
- **Frontend API**: Dedicated endpoints for dashboard integration

## 📊 Monitoring & Observability

### CloudWatch Dashboards
- **Stock Analytics AI Performance**: Comprehensive AI metrics dashboard
- **Hit Rate Trends**: Real-time accuracy monitoring
- **Model Performance**: Confidence vs accuracy analysis
- **API Usage**: Rate limiting and authentication metrics

### Key Metrics Tracked
- **ML Model Performance**: Hit rates, time-to-prediction, confidence accuracy
- **API Performance**: Request latency, authentication success rates
- **Infrastructure Health**: Lambda duration, DynamoDB performance, cache hit rates
- **Business Metrics**: Recommendations generated, symbol coverage, prediction accuracy

### Automated Alerting
- **Cost Management**: Alerts when monthly costs exceed $300
- **Performance Degradation**: ML model accuracy dropping below thresholds  
- **API Issues**: Authentication failures, rate limit violations
- **Infrastructure Problems**: Lambda errors, database connection issues

## 🔧 Infrastructure Management

### Terraform Configuration
- **main.tf**: Core AWS services and networking
- **deploy_advanced_tuning.tf**: Advanced model tuning Lambda system
- **api_key_auth.tf**: API authentication and usage plans  
- **outputs.tf**: API endpoints and configuration details

### Environment Configuration
- **terraform-tier1.tfvars**: Production configuration with AI analytics
- **variables.tf**: Configurable parameters and defaults
- **API Key Management**: Secure key rotation and access control

### Lambda Functions
- **stock_data_ingestion.py**: Market data collection and processing
- **stock_recommendations_api.py**: Main API request handling
- **lambda_functions/price_model_tuning.py**: Price prediction model optimization and tuning
- **lambda_functions/time_model_tuning.py**: Time-to-hit prediction model optimization
- **lambda_functions/dual_accuracy_tracker.py**: Performance monitoring and analytics tracking
- **lambda_functions/model_tuning_reporter.py**: Model optimization reporting and insights

## 💰 Cost Optimization & Tiers

### Current Deployment: Tier 1 Production
- **Estimated Cost**: $245/month
- **Key Benefits**: Production stability, AI analytics, premium API access
- **Scalability**: Handles 150+ stocks with enhanced monitoring

### Cost Breakdown
- **Database**: db.r5.large cluster (~$140/month)
- **Cache**: 3x cache.r6g.large nodes (~$65/month) 
- **Lambda**: Enhanced memory allocation (~$25/month)
- **API Gateway + CloudWatch**: (~$15/month)

### Optimization Strategies
- **Lambda-Only ML**: Serverless ensemble models for cost efficiency (no SageMaker)
- **Evidence-Based Validation**: Walk-forward backtesting prevents unrealistic expectations
- **Market Benchmarking**: S&P 500 compliance ensures competitive performance
- **Automated Scheduling**: EventBridge reduces manual intervention costs

## 🔧 Development & Deployment

### Prerequisites
- AWS CLI configured with `stock-analytics-admin` profile
- Terraform >= 1.0
- Alpha Vantage Premium API key

### Quick Deployment
```bash
# Deploy core infrastructure
terraform apply -var-file="terraform-current.tfvars" -auto-approve

# Deploy advanced tuning system
terraform apply -target="module.deploy_advanced_tuning" -auto-approve

# Get API endpoints and key
terraform output api_endpoints
terraform output api_key_value
```

### Testing the System
```bash
# Test main recommendations API
curl -H "x-api-key: lp1ISnIC60ambkegNj3yn1SHaX8EVuQ41pECFNfq" \
     "https://2cqomr4nb2.execute-api.us-east-1.amazonaws.com/prod/recommendations"

# Trigger advanced model tuning
aws lambda invoke --function-name advanced-model-tuning-service \
  --payload '{"action":"comprehensive_tuning","lookback_days":90}' \
  --profile stock-analytics-admin response.json

# Monitor tuning progress
aws logs tail /aws/lambda/advanced-model-tuning-service --follow --profile stock-analytics-admin
```

## 📈 Performance Features

### Market-Beating Design
- **Evidence-Based Validation**: Walk-forward backtesting with real market data
- **Fundamental Analysis**: P/E ratios, earnings growth, sector rotation signals  
- **Ensemble ML**: XGBoost, LightGBM, and Neural Network combination
- **Benchmark Compliance**: S&P 500 outperformance validation

### Operational Excellence
- **Automated Optimization**: EventBridge-scheduled weekly tuning cycles
- **Risk Management**: Sharpe ratio >1.0, max drawdown <15% requirements
- **Performance Gates**: 65% hit rate threshold before deployment
- **Real-time Monitoring**: CloudWatch dashboards with custom metrics

## 🚀 Getting Started

```bash
# Clone and deploy
git clone <repository>
cd stock-analytics-engine
terraform init
terraform apply -var-file="terraform-current.tfvars"

# Test the system
aws lambda invoke --function-name advanced-model-tuning-service \
  --payload '{"action":"comprehensive_tuning"}' response.json
```

This platform implements advanced ML model tuning designed to achieve consistent market-beating performance through evidence-based validation and systematic optimization.