# Railway-Native Deployment Guide (AWS-Free)

Complete guide for deploying the Stock Analytics Engine to Railway without any AWS dependencies.

## Overview

This deployment uses **only Railway services** - no AWS accounts, credentials, or services required:

| Component | Railway Solution |
|-----------|------------------|
| Database | Railway PostgreSQL |
| Cache | Railway Redis |
| Storage | PostgreSQL (JSONB) |
| Compute | Railway Containers |
| Scheduling | Python `schedule` library |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Railway Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   API Service   │  │ Data Ingestion  │  │ Model Tuning │ │
│  │   (Flask)       │  │    Worker       │  │   Worker     │ │
│  │   Port: 8080    │  │   Scheduled     │  │   Scheduled  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
│           └──────────┬─────────┴───────────────────┘         │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │   PostgreSQL        │                           │
│           │   (Railway Add-on)  │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │   Redis (Optional)  │                           │
│           │   (Railway Add-on)  │                           │
│           └─────────────────────┘                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Railway Account**: Sign up at https://railway.app
2. **Railway CLI** (optional):
   ```bash
   npm install -g @railway/cli
   railway login
   ```
3. **Alpha Vantage API Key**: Get free key at https://www.alphavantage.co/support/#api-key

## Deployment Steps

### Step 1: Create Railway Project

```bash
# Via CLI
railway init
railway link

# Or via Dashboard: https://railway.app/new
```

### Step 2: Add PostgreSQL Database

1. In Railway Dashboard, click "New" → "Database" → "PostgreSQL"
2. Railway automatically provides `DATABASE_URL`

### Step 3: Add Redis (Optional, for caching)

1. Click "New" → "Database" → "Redis"
2. Railway automatically provides `REDIS_URL`

### Step 4: Deploy Services

**Option A: GitHub Integration (Recommended)**

1. Connect your GitHub repository to Railway
2. Create three services with these settings:

| Service | Dockerfile Path |
|---------|----------------|
| api-service | `railway/api-service/Dockerfile` |
| data-ingestion | `railway/data-ingestion/Dockerfile` |
| model-tuning | `railway/model-tuning/Dockerfile` |

**Option B: Railway CLI**

```bash
# Deploy API Service
railway service create api-service
cd railway/api-service
railway up

# Deploy Data Ingestion Worker
railway service create data-ingestion
cd ../data-ingestion
railway up

# Deploy Model Tuning Worker
railway service create model-tuning
cd ../model-tuning
railway up
```

### Step 5: Configure Environment Variables

**All Services:**
```bash
ENVIRONMENT=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}  # If using Redis
```

**Data Ingestion Service:**
```bash
ALPHA_VANTAGE_API_KEY=your_api_key_here
MAX_SYMBOLS_PER_RUN=3
MARKET_INTERVAL_MINUTES=5
EVENING_INTERVAL_MINUTES=10
```

**Model Tuning Service:**
```bash
TARGET_HIT_RATE=0.65
ENABLE_CONTINUOUS_MONITORING=false
```

### Step 6: Enable Public Access (API Service only)

1. Select `api-service` in Dashboard
2. Go to Settings → Networking → Generate Domain
3. Your API is now accessible at: `https://api-service-xxx.up.railway.app`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/recommendations` | GET | All stock recommendations |
| `/recommendations/{symbol}` | GET | Recommendation for symbol |
| `/prices` | GET | Latest prices for all symbols |
| `/prices/{symbol}` | GET | Latest price for symbol |
| `/quotes/{symbol}` | GET | Historical quotes |
| `/analytics/dashboard` | GET | Analytics dashboard |

## Database Schema

The services automatically create these PostgreSQL tables:

```sql
-- Stock quotes (historical data)
stock_quotes (symbol, date, open_price, high_price, low_price,
              close_price, volume, moving_avg_5, moving_avg_20, volatility)

-- Latest prices (real-time cache)
latest_prices (symbol, price, change_amount, change_percent, volume)

-- Recommendations
stock_recommendations (symbol, recommendation, confidence,
                       target_price, current_price, analysis_data)

-- Price predictions
price_predictions (symbol, prediction_date, predicted_price,
                   actual_price, confidence, model_version)

-- Model performance tracking
model_performance (model_type, evaluation_date, total_predictions,
                   correct_predictions, hit_rate, metrics)

-- Prediction validations
prediction_validations (symbol, prediction_date, was_correct,
                        actual_change_percent, confidence)

-- Ingestion logs
ingestion_logs (run_id, symbols_attempted, symbols_succeeded,
                duration_seconds, errors)
```

## Schedule Configuration

**Data Ingestion Worker:**
- Market hours (9 AM - 4 PM EST): Every 5 minutes
- Evening hours (5 PM - 11 PM EST): Every 10 minutes
- End of day: 4:30 PM EST

**Model Tuning Worker:**
- Daily validation: 6 AM EST
- Weekly comprehensive tuning: Sunday 2 AM EST
- Continuous monitoring (optional): Every 6 hours

## Cost Estimate

Railway uses usage-based pricing:

| Resource | Estimated Cost |
|----------|---------------|
| API Service (2GB RAM) | ~$15/month |
| Data Ingestion (1GB RAM) | ~$8/month |
| Model Tuning (1GB RAM) | ~$8/month |
| PostgreSQL | ~$5/month |
| Redis (optional) | ~$3/month |
| **Total** | **~$36-39/month** |

*Compared to AWS deployment (~$245/month): **85% cost savings***

## Monitoring

### Logs
```bash
# Via CLI
railway logs -s api-service --follow
railway logs -s data-ingestion --tail 100
railway logs -s model-tuning
```

### Health Checks
```bash
curl https://your-api-url/health
```

### Performance Dashboard
```bash
curl https://your-api-url/analytics/dashboard
```

## Troubleshooting

### Service Won't Start
```bash
# Check build logs
railway logs -s service-name --deployment latest

# Common issues:
# - Missing DATABASE_URL
# - Docker build failed
# - Port binding issues
```

### Database Connection Failed
```bash
# Verify DATABASE_URL is set
railway variables -s api-service | grep DATABASE

# Test connection manually
railway run -s api-service -- python -c "
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Connected!')
"
```

### No Data Being Collected
1. Verify Alpha Vantage API key is set
2. Check data-ingestion logs for rate limiting
3. Confirm market hours (data collection is time-based)

## Migration from AWS

If migrating from the AWS deployment:

1. **Export existing data** (optional):
   ```bash
   # Export DynamoDB data
   aws dynamodb scan --table-name stock-recommendations > data.json
   ```

2. **Deploy Railway services** (this guide)

3. **Import data** (optional):
   ```bash
   # Connect to Railway PostgreSQL and import
   railway run -s api-service -- python import_data.py
   ```

4. **Update DNS/routing** to point to Railway

5. **Decommission AWS resources** once Railway is verified

## Files Structure

```
railway/
├── shared/
│   ├── __init__.py
│   └── logger.py           # AWS-free structured logger
├── api-service/
│   ├── Dockerfile
│   ├── app.py              # Original (AWS-dependent)
│   └── app_native.py       # Railway-native (no AWS)
├── data-ingestion/
│   ├── Dockerfile
│   ├── worker.py           # Original (AWS-dependent)
│   └── worker_native.py    # Railway-native (no AWS)
├── model-tuning/
│   ├── Dockerfile
│   ├── worker.py           # Original (AWS-dependent)
│   └── worker_native.py    # Railway-native (no AWS)
├── RAILWAY_DEPLOYMENT.md   # Original docs (hybrid approach)
├── RAILWAY_ENV_VARS.md     # Environment variables reference
└── RAILWAY_NATIVE.md       # This file (AWS-free approach)
```

## Support

- Railway Documentation: https://docs.railway.app
- Alpha Vantage API Docs: https://www.alphavantage.co/documentation/
- Project Issues: Open an issue in the repository
