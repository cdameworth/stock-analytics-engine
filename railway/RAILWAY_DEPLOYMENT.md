# Railway Deployment Guide for Stock Analytics Engine

Complete guide for deploying the Stock Analytics Engine to Railway platform.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Changes](#architecture-changes)
4. [Deployment Steps](#deployment-steps)
5. [Service Configuration](#service-configuration)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Cost Comparison](#cost-comparison)
9. [Migration from AWS](#migration-from-aws)

---

## 🎯 Overview

Railway deployment transforms the AWS Lambda-based serverless architecture into containerized microservices:

- **3 Docker Services** replace 15+ Lambda functions
- **Persistent workers** replace EventBridge schedules
- **Railway PostgreSQL/Redis** can replace Aurora/ElastiCache
- **Public API endpoint** replaces API Gateway

### Architecture Transformation

| AWS Component | Railway Equivalent |
|---------------|-------------------|
| Lambda Functions | Docker Containers |
| API Gateway | Railway Public Networking |
| EventBridge Schedules | Python `schedule` library |
| CloudWatch Logs | Railway Logging |
| Secrets Manager | Railway Environment Variables |
| Aurora PostgreSQL | Railway PostgreSQL Add-on |
| ElastiCache Valkey | Railway Redis Add-on |

---

## ✅ Prerequisites

### 1. Railway Account
```bash
# Sign up at https://railway.app
# Install Railway CLI
npm install -g @railway/cli

# Or using Homebrew
brew install railway
```

### 2. Login to Railway
```bash
railway login
```

### 3. AWS Resources (if keeping AWS integration)
- DynamoDB tables (stock-recommendations, ai-performance-analytics, etc.)
- S3 buckets (data-lake, ml-models, performance)
- IAM credentials with DynamoDB + S3 access
- Alpha Vantage API key

### 4. Local Development Setup
```bash
# Clone repository
git clone <your-repo-url>
cd stock-analytics-engine

# Verify Docker is installed
docker --version

# Test local builds (optional)
cd railway/api-service
docker build -t stock-api:test .
```

---

## 🏗️ Architecture Changes

### Lambda to Container Transformation

**Before (AWS Lambda):**
```
Individual functions invoked by:
- API Gateway (HTTP requests)
- EventBridge (schedules)
- Direct invocation
```

**After (Railway):**
```
Persistent services:
- API Service: Flask web server (HTTP)
- Data Ingestion: Scheduled worker
- Model Tuning: Scheduled worker
```

### Key Differences

1. **Stateless → Stateful**: Workers run continuously instead of on-demand
2. **Event-driven → Schedule-driven**: Uses Python `schedule` library
3. **Auto-scaling → Fixed resources**: Configure replicas manually
4. **Pay-per-request → Pay-per-second**: Different pricing model

---

## 🚀 Deployment Steps

### Step 1: Initialize Railway Project

```bash
# Create new Railway project
railway init

# Link to existing project (if applicable)
railway link <project-id>
```

### Step 2: Create Services

**Option A: Using Railway CLI**

```bash
# API Service
railway service create api-service
railway up -s api-service -d railway/api-service

# Data Ingestion Worker
railway service create data-ingestion
railway up -s data-ingestion -d railway/data-ingestion

# Model Tuning Worker
railway service create model-tuning
railway up -s model-tuning -d railway/model-tuning
```

**Option B: Using Railway Dashboard**

1. Go to https://railway.app/dashboard
2. Create new project: "Stock Analytics Engine"
3. Add three services:
   - Click "New" → "Empty Service"
   - Name: `api-service`
   - Settings → Source → Connect to GitHub repo
   - Settings → Build → Set Dockerfile path: `railway/api-service/Dockerfile`
4. Repeat for `data-ingestion` and `model-tuning`

### Step 3: Configure Environment Variables

**API Service Variables:**

```bash
railway variables set -s api-service \
  AWS_ACCESS_KEY_ID="your_key" \
  AWS_SECRET_ACCESS_KEY="your_secret" \
  AWS_REGION="us-east-1" \
  ALPHA_VANTAGE_API_KEY="your_api_key" \
  RECOMMENDATIONS_TABLE="stock-recommendations" \
  ANALYTICS_TABLE="ai-performance-analytics" \
  S3_DATA_BUCKET="stock-analytics-data-lake" \
  S3_MODEL_BUCKET="stock-analytics-ml-models" \
  ENVIRONMENT="production" \
  LOG_LEVEL="INFO" \
  ENABLE_DUAL_PREDICTIONS="true"
```

**Data Ingestion Worker Variables:**

```bash
railway variables set -s data-ingestion \
  AWS_ACCESS_KEY_ID="your_key" \
  AWS_SECRET_ACCESS_KEY="your_secret" \
  AWS_REGION="us-east-1" \
  ALPHA_VANTAGE_API_KEY="your_api_key" \
  RECOMMENDATIONS_TABLE="stock-recommendations" \
  S3_DATA_BUCKET="stock-analytics-data-lake" \
  MARKET_INTERVAL_MINUTES="5" \
  EVENING_INTERVAL_MINUTES="10" \
  ENVIRONMENT="production"
```

**Model Tuning Worker Variables:**

```bash
railway variables set -s model-tuning \
  AWS_ACCESS_KEY_ID="your_key" \
  AWS_SECRET_ACCESS_KEY="your_secret" \
  AWS_REGION="us-east-1" \
  ANALYTICS_TABLE="ai-performance-analytics" \
  S3_MODEL_BUCKET="stock-analytics-ml-models" \
  S3_PERFORMANCE_BUCKET="stock-analytics-model-performance" \
  TARGET_HIT_RATE="0.65" \
  ENVIRONMENT="production"
```

See [RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md) for complete variable reference.

### Step 4: Add Database Add-ons (Optional)

**PostgreSQL (Replaces Aurora):**

```bash
# Add PostgreSQL through Railway dashboard
railway add postgresql -s api-service

# Auto-generated variables:
# DATABASE_URL=${{Postgres.DATABASE_URL}}
# PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
```

**Redis (Replaces ElastiCache):**

```bash
# Add Redis through Railway dashboard
railway add redis -s api-service

# Auto-generated variables:
# REDIS_URL=${{Redis.REDIS_URL}}

# Enable caching
railway variables set -s api-service \
  ENABLE_CACHING="true" \
  VALKEY_ENDPOINT='${{Redis.REDIS_URL}}'
```

### Step 5: Enable Public Access (API Service)

```bash
# Generate public domain for API service
railway domain -s api-service

# Returns: https://api-service-production.up.railway.app
```

Or via dashboard:
1. Select `api-service`
2. Settings → Networking → Generate Domain
3. Optionally add custom domain

### Step 6: Deploy Services

```bash
# Deploy all services
railway up

# Or deploy individually
railway up -s api-service
railway up -s data-ingestion
railway up -s model-tuning
```

### Step 7: Verify Deployment

```bash
# Check service status
railway status

# View logs
railway logs -s api-service
railway logs -s data-ingestion --tail 100
railway logs -s model-tuning

# Test API endpoint
curl https://api-service-production.up.railway.app/health

# Test recommendations endpoint
curl https://api-service-production.up.railway.app/recommendations
```

---

## 🔧 Service Configuration

### API Service

**Dockerfile Location:** `railway/api-service/Dockerfile`

**Key Features:**
- Flask + Gunicorn web server
- Multi-worker configuration (2 workers, 4 threads)
- Health check endpoint: `/health`
- Converts Lambda handlers to REST endpoints

**Endpoints:**
```
GET  /                           - API information
GET  /health                     - Health check
GET  /recommendations            - All recommendations
GET  /recommendations/{symbol}   - Single symbol
GET  /analytics/dashboard        - Analytics dashboard
GET  /analytics/detailed         - Detailed analytics
POST /custom-request             - Custom analysis
```

**Resource Recommendations:**
- Memory: 2GB
- CPU: 2 vCPU
- Replicas: 1-3 (based on load)

### Data Ingestion Worker

**Dockerfile Location:** `railway/data-ingestion/Dockerfile`

**Key Features:**
- Scheduled data collection using Python `schedule`
- Market hours detection (9 AM - 4 PM EST)
- Evening processing (5 PM - 11 PM EST)
- End-of-day comprehensive run

**Schedule Configuration:**
```python
MARKET_INTERVAL_MINUTES=5      # Every 5 minutes during market hours
EVENING_INTERVAL_MINUTES=10    # Every 10 minutes in evening
```

**Resource Recommendations:**
- Memory: 1GB
- CPU: 1 vCPU
- Replicas: 1 (single instance sufficient)

### Model Tuning Worker

**Dockerfile Location:** `railway/model-tuning/Dockerfile`

**Key Features:**
- Daily accuracy validation (6 AM EST)
- Weekly comprehensive tuning (Sunday 2 AM EST)
- Optional continuous monitoring (every 6 hours)

**Schedule Configuration:**
```python
# Daily validation
schedule.every().day.at("11:00").do(run_daily_validation)

# Weekly tuning (Sunday)
schedule.every().sunday.at("07:00").do(run_weekly_comprehensive_tuning)

# Optional continuous monitoring
ENABLE_CONTINUOUS_MONITORING=true  # Every 6 hours
```

**Resource Recommendations:**
- Memory: 2GB (ML libraries need more RAM)
- CPU: 2 vCPU
- Replicas: 1

---

## 🧪 Testing

### Local Testing with Docker

```bash
# Build API service
cd railway/api-service
docker build -t stock-api:test .

# Run with environment variables
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e ALPHA_VANTAGE_API_KEY=your_api_key \
  -e RECOMMENDATIONS_TABLE=stock-recommendations \
  -e PORT=8080 \
  stock-api:test

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/recommendations
```

### Railway Preview Environments

Create feature branches for isolated testing:

```bash
# Create feature branch
git checkout -b feature/railway-migration

# Push changes
git push origin feature/railway-migration

# Railway automatically creates preview environment
# Access at: https://api-service-feature-railway-migration.up.railway.app
```

### Integration Testing

```bash
# Test complete workflow
# 1. API returns recommendations
curl https://your-api-url/recommendations

# 2. Check data ingestion logs
railway logs -s data-ingestion --tail 50

# 3. Verify model tuning runs
railway logs -s model-tuning --tail 50

# 4. Validate analytics
curl https://your-api-url/analytics/dashboard
```

---

## 📊 Monitoring

### Railway Dashboard

Access monitoring at: `https://railway.app/project/<project-id>`

**Metrics Available:**
- CPU usage (%)
- Memory usage (MB)
- Network (requests/sec)
- Deployment status
- Build logs
- Runtime logs

### Log Monitoring

```bash
# Real-time logs
railway logs -s api-service --follow

# Search logs
railway logs -s api-service | grep "ERROR"

# Export logs
railway logs -s api-service --since 1h > logs.txt
```

### Health Checks

**API Service:**
```bash
# Health endpoint
curl https://your-api-url/health

# Expected response
{
  "status": "healthy",
  "service": "stock-analytics-api",
  "timestamp": "2025-12-08T10:30:00Z",
  "version": "1.0.0"
}
```

**Worker Services:**
Check logs for successful execution:
```bash
railway logs -s data-ingestion | grep "completed successfully"
railway logs -s model-tuning | grep "tuning completed"
```

### Alerts

Configure alerts in Railway dashboard:
1. Project Settings → Notifications
2. Add webhook for Slack/Discord/Email
3. Set alert conditions:
   - Deployment failures
   - High CPU/memory usage
   - Service crashes

---

## 💰 Cost Comparison

### AWS Current Costs (~$245/month)

```
Aurora PostgreSQL (db.r5.large):     $140/month
ElastiCache (3x cache.r6g.large):    $65/month
Lambda executions:                   $25/month
API Gateway + CloudWatch:            $15/month
─────────────────────────────────────────────
Total:                               $245/month
```

### Railway Estimated Costs

**Starter Plan ($5/month + usage):**
```
Base subscription:                   $5/month
API Service (2GB RAM, 2 vCPU):       ~$15/month
Data Ingestion (1GB RAM, 1 vCPU):    ~$8/month
Model Tuning (2GB RAM, 2 vCPU):      ~$15/month
PostgreSQL (1GB storage):            $5/month
Redis (256MB):                       $3/month
─────────────────────────────────────────────
Total:                               ~$51/month
```

**Developer Plan ($20/month + usage):**
```
Base subscription:                   $20/month
API Service (higher limits):         ~$20/month
Data Ingestion:                      ~$10/month
Model Tuning:                        ~$20/month
PostgreSQL (higher performance):     $10/month
Redis (512MB):                       $5/month
─────────────────────────────────────────────
Total:                               ~$85/month
```

### Cost Savings: **~$160-190/month** (65-77% reduction)

---

## 🔄 Migration from AWS

### Phase 1: Dual Operation

1. **Deploy Railway services** alongside existing AWS infrastructure
2. **Test Railway deployment** with production data
3. **Compare performance** and accuracy
4. **Keep AWS as fallback**

### Phase 2: Traffic Migration

1. **Update DNS/routing** to point to Railway API
2. **Monitor error rates** and performance
3. **Keep AWS Lambda functions** running as backup
4. **Gradual traffic shift** (10% → 50% → 100%)

### Phase 3: Data Migration (Optional)

**DynamoDB → PostgreSQL:**

```python
# Export DynamoDB data
aws dynamodb scan --table-name stock-recommendations > export.json

# Import to PostgreSQL
python scripts/migrate_dynamo_to_postgres.py export.json
```

**S3 → Railway Volumes:**

```bash
# Download S3 data
aws s3 sync s3://stock-analytics-data-lake ./local-data

# Upload to Railway volume
railway volume create data-volume -s api-service
railway run -s api-service -- rsync -av ./local-data /data/
```

### Phase 4: AWS Decommission

1. **Stop EventBridge schedules**
2. **Disable Lambda functions**
3. **Delete API Gateway**
4. **Optionally keep DynamoDB/S3** or migrate fully

---

## 🛠️ Troubleshooting

### Issue: Service Won't Start

```bash
# Check build logs
railway logs -s api-service --deployment <deployment-id>

# Common issues:
# - Missing environment variables
# - Dockerfile path incorrect
# - Port binding issues
```

### Issue: Cannot Connect to AWS Resources

```bash
# Verify AWS credentials
railway run -s api-service -- env | grep AWS

# Test DynamoDB access
railway run -s api-service -- python -c "
import boto3
dynamo = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamo.Table('stock-recommendations')
print(table.table_status)
"
```

### Issue: Worker Not Running Scheduled Tasks

```bash
# Check worker logs
railway logs -s data-ingestion --tail 200 | grep "schedule"

# Verify timezone configuration
railway run -s data-ingestion -- python -c "
from datetime import datetime
import pytz
print(datetime.now(pytz.timezone('US/Eastern')))
"
```

### Issue: High Memory Usage

```bash
# Check resource usage
railway status

# Scale up resources in dashboard:
# Settings → Resources → Memory Limit → Increase
```

---

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway CLI Reference](https://docs.railway.app/reference/cli-api)
- [Environment Variables Guide](./RAILWAY_ENV_VARS.md)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Flask Production Deployment](https://flask.palletsprojects.com/en/3.0.x/deploying/)

---

## 🎉 Next Steps

1. ✅ Deploy Railway services
2. ✅ Configure environment variables
3. ✅ Test API endpoints
4. ✅ Monitor worker logs
5. ✅ Set up alerts
6. 🔄 Migrate traffic from AWS
7. 🔄 Decommission AWS resources (optional)
8. 🚀 Scale as needed

**Questions or issues?** Open an issue in the repository or check Railway's [community forum](https://help.railway.app/).
