# Railway Deployment - Summary

## ✅ What Was Created

Railway configuration files to deploy the Stock Analytics Engine as containerized microservices, replacing the AWS Lambda serverless architecture.

### Files Created

1. **`railway.json`** (project root)
   - Project-level Railway configuration
   - Specifies Dockerfile builder
   - Restart policy configuration

2. **`railway/api-service/`**
   - `Dockerfile` - API service container definition
   - `app.py` - Flask wrapper converting Lambda handlers to REST endpoints

3. **`railway/data-ingestion/`**
   - `Dockerfile` - Data ingestion worker container
   - `worker.py` - Scheduled data collection from Alpha Vantage

4. **`railway/model-tuning/`**
   - `Dockerfile` - Model tuning worker container
   - `worker.py` - ML model optimization and accuracy tracking

5. **`railway/RAILWAY_DEPLOYMENT.md`**
   - Complete deployment guide (7,500+ words)
   - Step-by-step instructions
   - Testing procedures
   - Troubleshooting guide

6. **`railway/RAILWAY_ENV_VARS.md`**
   - Comprehensive environment variable reference
   - Service-specific configurations
   - Database add-on setup
   - AWS alternatives

7. **`railway/README.md`**
   - Quick start guide
   - Architecture overview
   - Cost comparison
   - Migration checklist

8. **Updated `CLAUDE.md`**
   - Added Railway deployment option section
   - Cross-reference to Railway documentation

---

## 🏗️ Architecture Transformation

### Before (AWS Lambda)

```
┌─────────────────────────────────────────────┐
│          API Gateway (REST API)             │
├─────────────────────────────────────────────┤
│  Lambda Functions (15+ individual)         │
│  - stock_recommendations_api                │
│  - dual_prediction_reporting_api            │
│  - custom_stock_request_api                 │
│  - stock_data_ingestion                     │
│  - price_model_tuning                       │
│  - time_model_tuning                        │
│  - dual_accuracy_tracker                    │
│  - And 8+ more...                           │
├─────────────────────────────────────────────┤
│  EventBridge (Scheduled triggers)           │
│  - Every 5 min (market hours)               │
│  - Daily (6 AM EST)                         │
│  - Weekly (Sunday 2 AM)                     │
├─────────────────────────────────────────────┤
│  Data Layer                                 │
│  - DynamoDB (4 tables)                      │
│  - S3 (3 buckets)                           │
│  - Aurora PostgreSQL (optional, $140/mo)    │
│  - ElastiCache Valkey (optional, $65/mo)    │
└─────────────────────────────────────────────┘

Cost: ~$245/month
```

### After (Railway)

```
┌─────────────────────────────────────────────┐
│     Railway Public Networking (HTTPS)       │
├─────────────────────────────────────────────┤
│  api-service (Flask + Gunicorn)             │
│  - All API endpoints in one container       │
│  - 2 workers, 4 threads                     │
│  - Health checks                            │
│  - 2GB RAM, 2 vCPU                          │
├─────────────────────────────────────────────┤
│  data-ingestion (Python scheduler)          │
│  - Market hours: Every 5 min                │
│  - Evening: Every 10 min                    │
│  - 1GB RAM, 1 vCPU                          │
├─────────────────────────────────────────────┤
│  model-tuning (Python scheduler)            │
│  - Daily validation: 6 AM EST               │
│  - Weekly tuning: Sunday 2 AM               │
│  - 2GB RAM, 2 vCPU                          │
├─────────────────────────────────────────────┤
│  Data Layer (Hybrid Options)                │
│  Option A: Keep AWS                         │
│  - DynamoDB (same tables)                   │
│  - S3 (same buckets)                        │
│                                             │
│  Option B: Railway Add-ons                  │
│  - PostgreSQL ($5-10/month)                 │
│  - Redis ($3-5/month)                       │
│  - Volumes (persistent storage)             │
└─────────────────────────────────────────────┘

Cost: ~$51-85/month (65-79% savings)
```

---

## 📊 Service Details

### 1. API Service (`railway/api-service/`)

**Purpose**: REST API replacing API Gateway + Lambda functions

**Key Features**:
- Flask web framework with Gunicorn WSGI server
- Converts Lambda event/context to HTTP requests
- Multi-worker configuration (2 workers, 4 threads)
- Health check endpoint at `/health`
- Auto-restarts on failure

**Endpoints**:
```
GET  /                           → API info
GET  /health                     → Health check
GET  /recommendations            → All recommendations
GET  /recommendations/{symbol}   → Single symbol
GET  /analytics/dashboard        → Dashboard data
GET  /analytics/detailed         → Detailed analytics
POST /custom-request             → Custom analysis
```

**Technology Stack**:
- Python 3.11
- Flask 3.0.0
- Gunicorn 21.2.0
- Boto3 1.34.0 (AWS SDK)
- All existing Lambda dependencies

### 2. Data Ingestion Worker (`railway/data-ingestion/`)

**Purpose**: Scheduled data collection replacing EventBridge triggers

**Key Features**:
- Python `schedule` library for cron-like scheduling
- Market hours detection (9 AM - 4 PM EST, Mon-Fri)
- Evening processing (5 PM - 11 PM EST)
- End-of-day comprehensive run (4:30 PM EST)
- Timezone-aware scheduling

**Schedule**:
```python
# Market hours (9 AM - 4 PM EST, Mon-Fri)
Every 5 minutes → run_market_hours_job()

# Evening hours (5 PM - 11 PM EST, Mon-Fri)
Every 10 minutes → run_evening_job()

# End of day (4:30 PM EST, Mon-Fri)
Daily at 16:30 → run_data_ingestion()
```

**Configuration**:
```bash
MARKET_INTERVAL_MINUTES=5
EVENING_INTERVAL_MINUTES=10
```

### 3. Model Tuning Worker (`railway/model-tuning/`)

**Purpose**: ML model optimization replacing EventBridge scheduled tuning

**Key Features**:
- Daily accuracy validation
- Weekly comprehensive model tuning
- Optional continuous monitoring
- Price prediction model tuning
- Time-to-hit prediction model tuning
- Performance tracking and reporting

**Schedule**:
```python
# Daily validation (6 AM EST / 11:00 UTC)
Every day at 11:00 UTC → run_daily_validation()

# Weekly comprehensive tuning (Sunday 2 AM EST / 7:00 UTC)
Every Sunday at 07:00 UTC → run_weekly_comprehensive_tuning()

# Optional continuous monitoring
Every 6 hours → run_accuracy_tracking()  # if enabled
```

**Configuration**:
```bash
ENABLE_CONTINUOUS_MONITORING=false
TARGET_HIT_RATE=0.65
TARGET_SHARPE_RATIO=1.0
TARGET_MARKET_OUTPERFORMANCE=0.05
```

---

## 💰 Cost Breakdown

### AWS (Current)

| Component | Cost/Month | Notes |
|-----------|-----------|-------|
| Aurora PostgreSQL (db.r5.large) | $140 | Optional, high-performance DB |
| ElastiCache Valkey (3x cache.r6g.large) | $65 | Optional, Redis cache |
| Lambda executions | $25 | 15+ functions, scheduled |
| API Gateway | $10 | REST API + usage plans |
| CloudWatch | $5 | Logs + metrics |
| **Total** | **$245** | |

### Railway (Estimated)

#### Starter Plan ($5/month base)

| Component | Cost/Month | Notes |
|-----------|-----------|-------|
| Subscription | $5 | Base plan |
| api-service (2GB, 2 vCPU) | $15 | ~$10/GB RAM |
| data-ingestion (1GB, 1 vCPU) | $8 | ~$8/GB RAM |
| model-tuning (2GB, 2 vCPU) | $15 | ~$10/GB RAM |
| PostgreSQL (1GB) | $5 | Database add-on |
| Redis (256MB) | $3 | Cache add-on |
| **Total** | **$51** | **79% savings** |

#### Developer Plan ($20/month base)

| Component | Cost/Month | Notes |
|-----------|-----------|-------|
| Subscription | $20 | Higher limits |
| api-service (2GB, 2 vCPU) | $20 | Higher performance |
| data-ingestion (1GB, 1 vCPU) | $10 | |
| model-tuning (2GB, 2 vCPU) | $20 | |
| PostgreSQL (better perf) | $10 | More storage/IOPS |
| Redis (512MB) | $5 | More capacity |
| **Total** | **$85** | **65% savings** |

**Cost Savings**: $160-190/month

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login
```

### Deploy All Services

```bash
# 1. Initialize Railway project
railway init

# 2. Deploy API service
railway service create api-service
railway up -s api-service -d railway/api-service

# 3. Deploy data ingestion
railway service create data-ingestion
railway up -s data-ingestion -d railway/data-ingestion

# 4. Deploy model tuning
railway service create model-tuning
railway up -s model-tuning -d railway/model-tuning

# 5. Enable public API
railway domain -s api-service
```

### Configure Environment Variables

```bash
# Set AWS credentials (all services)
railway variables set \
  -s api-service \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  ALPHA_VANTAGE_API_KEY=your_api_key \
  RECOMMENDATIONS_TABLE=stock-recommendations \
  ENVIRONMENT=production

# Repeat for data-ingestion and model-tuning
# See railway/RAILWAY_ENV_VARS.md for complete list
```

---

## 🧪 Testing

### Local Docker Test

```bash
cd railway/api-service
docker build -t stock-api:test .
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=key \
  -e ALPHA_VANTAGE_API_KEY=key \
  -e PORT=8080 \
  stock-api:test

curl http://localhost:8080/health
```

### Railway Deployment Test

```bash
# Check status
railway status

# View logs
railway logs -s api-service --tail 100

# Test API
curl https://your-api-url.up.railway.app/health
curl https://your-api-url.up.railway.app/recommendations
```

---

## 📈 Migration Strategy

### Phase 1: Deploy Railway (Week 1)
- Deploy all three services
- Configure environment variables
- Test with production AWS resources
- Monitor logs and performance

### Phase 2: Parallel Operation (Week 2-3)
- Run Railway alongside AWS
- Compare API responses and accuracy
- Validate all endpoints
- Monitor costs and performance

### Phase 3: Traffic Migration (Week 4)
- Update DNS/routing to Railway API
- Gradual traffic shift (10% → 50% → 100%)
- Monitor error rates
- Keep AWS as fallback

### Phase 4: AWS Decommission (Week 5+)
- Stop EventBridge schedules
- Disable Lambda functions
- Delete API Gateway
- Optional: Migrate DynamoDB → PostgreSQL

---

## 🎯 Key Benefits

### Cost Savings
- **65-79% reduction** in monthly infrastructure costs
- No Aurora PostgreSQL ($140/month) required
- No ElastiCache Valkey ($65/month) required
- Predictable pricing model

### Simplicity
- **3 services** instead of 15+ Lambda functions
- Single deployment process
- Unified logging and monitoring
- Easier debugging (persistent containers)

### Performance
- **Persistent connections** (no Lambda cold starts)
- Local caching possible
- Shared memory between workers
- Faster request processing

### Flexibility
- Easy to add new endpoints
- Simple to modify schedules
- Can scale individual services
- Preview environments for testing

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [railway/README.md](./README.md) | Quick start and overview |
| [railway/RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md) | Complete deployment guide (7,500+ words) |
| [railway/RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md) | Environment variables reference |
| [railway/SUMMARY.md](./SUMMARY.md) | This file - architecture summary |

---

## ✅ What's Included

- ✅ Complete Dockerfiles for all services
- ✅ Flask API wrapper with Lambda compatibility
- ✅ Scheduled workers with timezone awareness
- ✅ Health check endpoints
- ✅ Logging and error handling
- ✅ Environment variable configuration
- ✅ Migration guides and checklists
- ✅ Cost comparison and optimization tips
- ✅ Troubleshooting documentation
- ✅ Testing procedures

---

## 🔄 Next Steps

1. **Read**: [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md) for detailed instructions
2. **Configure**: Review [RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md) for required variables
3. **Deploy**: Follow Quick Start guide above
4. **Test**: Verify all endpoints and worker schedules
5. **Monitor**: Set up alerts and check logs
6. **Migrate**: Gradually shift traffic from AWS

---

**Total Deployment Time**: ~2-3 hours
**Migration Timeline**: ~4-5 weeks (phased)
**Cost Savings**: ~$160-190/month (65-79% reduction)
**Complexity Reduction**: 15+ Lambda functions → 3 Docker services
