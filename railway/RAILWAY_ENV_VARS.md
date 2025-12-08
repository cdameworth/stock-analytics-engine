# Railway Environment Variables Configuration

Complete guide for configuring environment variables for Stock Analytics Engine on Railway.

## 🔧 Railway Service Setup

Deploy the Stock Analytics Engine as three separate Railway services:

1. **API Service** (`railway/api-service/`)
2. **Data Ingestion Worker** (`railway/data-ingestion/`)
3. **Model Tuning Worker** (`railway/model-tuning/`)

Each service requires its own set of environment variables.

---

## 🌐 API Service Environment Variables

### Required Variables

```bash
# AWS Configuration (for DynamoDB and S3 access)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1

# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
ALPHA_VANTAGE_API_KEY_SECRET_ARN=arn:aws:secretsmanager:us-east-1:account:secret:name

# Database Configuration
RECOMMENDATIONS_TABLE=stock-recommendations
ANALYTICS_TABLE=ai-performance-analytics
PRICE_PREDICTIONS_TABLE=price-predictions
TIME_PREDICTIONS_TABLE=time-to-hit-predictions

# S3 Buckets
S3_DATA_BUCKET=stock-analytics-data-lake
S3_MODEL_BUCKET=stock-analytics-ml-models
S3_PERFORMANCE_BUCKET=stock-analytics-model-performance
```

### Optional Variables

```bash
# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
FLASK_DEBUG=false

# Feature Flags
ENABLE_DUAL_PREDICTIONS=true
ENABLE_ADVANCED_TUNING=true
ENABLE_CACHING=false  # Set to true if using Redis
ENABLE_CUSTOM_METRICS=true
DEBUG_MODE=false

# Performance Targets
TARGET_HIT_RATE=0.65
TARGET_SHARPE_RATIO=1.0
TARGET_MARKET_OUTPERFORMANCE=0.05

# API Configuration
USE_PREMIUM_API_KEY=true
PREMIUM_API_CALLS_PER_MINUTE=75
CONNECT_TEST_TIMEOUT=2
PER_CALL_TIMEOUT=6

# Caching (if using Railway Redis addon)
REDIS_URL=${{Redis.REDIS_URL}}
VALKEY_ENDPOINT=${{Redis.REDIS_URL}}
CACHE_DEFAULT_TTL=300

# Rate Limiting
MAX_SYMBOLS_PER_RUN=12
ABORT_AFTER_SEC=40
```

---

## 📊 Data Ingestion Worker Environment Variables

### Required Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1

# Alpha Vantage API
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
ALPHA_VANTAGE_API_KEY_SECRET_ARN=arn:aws:secretsmanager:us-east-1:account:secret:name

# Database Configuration
RECOMMENDATIONS_TABLE=stock-recommendations
S3_DATA_BUCKET=stock-analytics-data-lake

# Worker Schedule Configuration
MARKET_INTERVAL_MINUTES=5      # Data collection frequency during market hours
EVENING_INTERVAL_MINUTES=10    # Data collection frequency during evening
```

### Optional Variables

```bash
# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Premium API Settings
USE_PREMIUM_API_KEY=true
PREMIUM_API_CALLS_PER_MINUTE=75

# Trading Configuration
MAX_SYMBOLS_PER_RUN=12
ABORT_AFTER_SEC=40
```

---

## 🧠 Model Tuning Worker Environment Variables

### Required Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1

# Database Configuration
ANALYTICS_TABLE=ai-performance-analytics
PRICE_PREDICTIONS_TABLE=price-predictions
TIME_PREDICTIONS_TABLE=time-to-hit-predictions

# S3 Buckets
S3_MODEL_BUCKET=stock-analytics-ml-models
S3_PERFORMANCE_BUCKET=stock-analytics-model-performance
```

### Optional Variables

```bash
# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Feature Flags
ENABLE_ADVANCED_TUNING=true
ENABLE_CONTINUOUS_MONITORING=false  # Enable 6-hour accuracy checks

# Performance Targets
TARGET_HIT_RATE=0.65
TARGET_SHARPE_RATIO=1.0
TARGET_MARKET_OUTPERFORMANCE=0.05
```

---

## 🗄️ Railway Database Add-ons

### PostgreSQL (Optional - for advanced analytics)

If using Railway's PostgreSQL add-on:

```bash
# Automatically provided by Railway
DATABASE_URL=${{Postgres.DATABASE_URL}}
PGHOST=${{Postgres.PGHOST}}
PGPORT=${{Postgres.PGPORT}}
PGUSER=${{Postgres.PGUSER}}
PGPASSWORD=${{Postgres.PGPASSWORD}}
PGDATABASE=${{Postgres.PGDATABASE}}
```

### Redis (Optional - for caching)

If using Railway's Redis add-on:

```bash
# Automatically provided by Railway
REDIS_URL=${{Redis.REDIS_URL}}
VALKEY_ENDPOINT=${{Redis.REDIS_URL}}
ENABLE_CACHING=true
```

---

## 🔒 AWS DynamoDB Alternative

Since Railway doesn't provide DynamoDB, you have two options:

### Option 1: Use AWS DynamoDB (Recommended)
Keep using AWS DynamoDB tables with proper IAM credentials (as shown above).

### Option 2: Migrate to Railway PostgreSQL

If you want to eliminate AWS dependency:

1. **Create PostgreSQL tables:**
   ```sql
   -- Run these in Railway PostgreSQL console
   CREATE TABLE stock_recommendations (
       symbol VARCHAR(10) PRIMARY KEY,
       recommendation JSONB,
       timestamp TIMESTAMP,
       ttl BIGINT
   );

   CREATE TABLE ai_performance_analytics (
       id SERIAL PRIMARY KEY,
       symbol VARCHAR(10),
       metrics JSONB,
       timestamp TIMESTAMP
   );

   CREATE TABLE price_predictions (
       id SERIAL PRIMARY KEY,
       symbol VARCHAR(10),
       prediction JSONB,
       timestamp TIMESTAMP
   );

   CREATE TABLE time_predictions (
       id SERIAL PRIMARY KEY,
       symbol VARCHAR(10),
       prediction JSONB,
       timestamp TIMESTAMP
   );
   ```

2. **Update shared utilities** to use PostgreSQL instead of DynamoDB
3. **Set environment variables:**
   ```bash
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   USE_DYNAMODB=false  # Add this flag
   ```

---

## 📦 AWS S3 Alternative

Railway doesn't provide S3-compatible storage by default. Options:

### Option 1: Continue using AWS S3 (Simplest)
Keep AWS credentials and S3 bucket names as configured.

### Option 2: Use Railway Volumes
For smaller datasets, use Railway's persistent volumes:

```bash
# In your Dockerfile, add volume mount
VOLUME /data

# Update S3 references to use local filesystem
S3_DATA_BUCKET=/data/stock-data
S3_MODEL_BUCKET=/data/models
S3_PERFORMANCE_BUCKET=/data/performance
```

---

## 🚀 Quick Setup Guide

### Step 1: Create Railway Project
```bash
railway login
railway init
railway link
```

### Step 2: Add Services

**API Service:**
```bash
railway service create api-service
railway up -s api-service -d railway/api-service/Dockerfile
```

**Data Ingestion Worker:**
```bash
railway service create data-ingestion
railway up -s data-ingestion -d railway/data-ingestion/Dockerfile
```

**Model Tuning Worker:**
```bash
railway service create model-tuning
railway up -s model-tuning -d railway/model-tuning/Dockerfile
```

### Step 3: Configure Environment Variables

For each service, set variables via Railway dashboard or CLI:

```bash
# Example: Set variables for api-service
railway variables set \
  -s api-service \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  ALPHA_VANTAGE_API_KEY=your_api_key \
  RECOMMENDATIONS_TABLE=stock-recommendations \
  ENVIRONMENT=production
```

### Step 4: Enable Public Networking (API Service Only)

```bash
railway domain -s api-service
```

This generates a public URL like: `https://api-service-production.up.railway.app`

---

## 🔗 Service References

Railway allows services to reference each other using template variables:

```bash
# If you deploy a separate PostgreSQL service
DATABASE_URL=${{postgres-service.DATABASE_URL}}

# If you deploy a Redis service
REDIS_URL=${{redis-service.REDIS_URL}}
```

---

## ⚙️ Cost Optimization

Railway pricing is based on resource usage:

- **Starter Plan**: $5/month + usage
- **Developer Plan**: $20/month + usage

### Recommendations:
1. **Start with API Service only** - Test basic functionality
2. **Add Data Ingestion Worker** - Once API is working
3. **Add Model Tuning Worker** - For production ML optimization
4. **Use Railway Redis** - Instead of AWS ElastiCache ($0.30/GB-month)
5. **Consider Railway PostgreSQL** - Instead of AWS Aurora ($245/month savings)

---

## 🛠️ Troubleshooting

### Issue: AWS Credentials Not Working
```bash
# Verify credentials in Railway dashboard
railway variables -s api-service

# Test AWS access
railway run -s api-service -- aws sts get-caller-identity
```

### Issue: Services Can't Connect to DynamoDB
```bash
# Check IAM permissions include:
# - dynamodb:GetItem
# - dynamodb:PutItem
# - dynamodb:Query
# - dynamodb:Scan
# - s3:GetObject
# - s3:PutObject
```

### Issue: Worker Services Not Running
```bash
# Check logs
railway logs -s data-ingestion --tail 100

# Verify health checks are passing
railway service -s data-ingestion
```

---

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway CLI Guide](https://docs.railway.app/develop/cli)
- [Railway Environment Variables](https://docs.railway.app/develop/variables)
- [Railway Volumes](https://docs.railway.app/develop/volumes)
