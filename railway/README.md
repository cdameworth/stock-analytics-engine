# Stock Analytics Engine - Railway Deployment

Transform the AWS Lambda serverless architecture into Railway containerized microservices.

## 🚀 Quick Start

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Deploy services
railway up -s api-service -d railway/api-service
railway up -s data-ingestion -d railway/data-ingestion
railway up -s model-tuning -d railway/model-tuning

# 5. Configure environment variables (see RAILWAY_ENV_VARS.md)
railway variables set -s api-service AWS_ACCESS_KEY_ID=your_key ...

# 6. Enable public API
railway domain -s api-service
```

## 📁 Directory Structure

```
railway/
├── README.md                      # This file
├── RAILWAY_DEPLOYMENT.md          # Complete deployment guide
├── RAILWAY_ENV_VARS.md            # Environment variables reference
├── api-service/                   # API Service (Flask web server)
│   ├── Dockerfile
│   └── app.py                     # Lambda → REST API wrapper
├── data-ingestion/                # Data Ingestion Worker
│   ├── Dockerfile
│   └── worker.py                  # Scheduled data collection
└── model-tuning/                  # Model Tuning Worker
    ├── Dockerfile
    └── worker.py                  # ML model optimization
```

## 🏗️ Architecture

### Services Overview

| Service | Purpose | Type | Resources |
|---------|---------|------|-----------|
| **api-service** | REST API endpoints | Web Server | 2GB RAM, 2 vCPU |
| **data-ingestion** | Market data collection | Worker | 1GB RAM, 1 vCPU |
| **model-tuning** | ML model optimization | Worker | 2GB RAM, 2 vCPU |

### Migration from AWS

```
AWS Lambda Functions      →  Railway Docker Containers
API Gateway              →  Railway Public Networking
EventBridge Schedules    →  Python schedule library
CloudWatch Logs          →  Railway Logging
Secrets Manager          →  Railway Environment Variables
Aurora PostgreSQL (opt)  →  Railway PostgreSQL Add-on
ElastiCache (opt)        →  Railway Redis Add-on
```

## 📊 Service Details

### API Service

**Endpoints:**
- `GET /health` - Health check
- `GET /recommendations` - All stock recommendations
- `GET /recommendations/{symbol}` - Single symbol recommendation
- `GET /analytics/dashboard` - Performance dashboard
- `GET /analytics/detailed` - Detailed analytics
- `POST /custom-request` - Custom stock analysis

**Technology:**
- Flask + Gunicorn
- 2 workers, 4 threads
- 120s timeout
- Health checks enabled

### Data Ingestion Worker

**Schedule:**
- Market hours: Every 5 minutes (9 AM - 4 PM EST)
- Evening: Every 10 minutes (5 PM - 11 PM EST)
- End-of-day: 4:30 PM EST comprehensive run

**Configuration:**
```bash
MARKET_INTERVAL_MINUTES=5
EVENING_INTERVAL_MINUTES=10
```

### Model Tuning Worker

**Schedule:**
- Daily validation: 6 AM EST
- Weekly comprehensive tuning: Sunday 2 AM EST
- Optional continuous monitoring: Every 6 hours

**Configuration:**
```bash
ENABLE_CONTINUOUS_MONITORING=false  # Set to true for 6-hour checks
TARGET_HIT_RATE=0.65
TARGET_SHARPE_RATIO=1.0
```

## 💰 Cost Comparison

| Platform | Monthly Cost | Savings |
|----------|-------------|---------|
| **AWS (Current)** | $245/month | - |
| **Railway Starter** | $51/month | 79% reduction |
| **Railway Developer** | $85/month | 65% reduction |

**AWS Breakdown:**
- Aurora PostgreSQL: $140/month
- ElastiCache Valkey: $65/month
- Lambda + API Gateway: $40/month

**Railway Breakdown:**
- Subscription: $5-20/month
- API Service: $15-20/month
- Data Ingestion: $8-10/month
- Model Tuning: $15-20/month
- PostgreSQL: $5-10/month
- Redis: $3-5/month

## 🔧 Environment Variables

### Required for All Services

```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
ALPHA_VANTAGE_API_KEY=your_api_key
```

### Service-Specific

See [RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md) for complete configuration.

## 🧪 Testing

### Local Docker Build

```bash
# Build and test API service
cd api-service
docker build -t stock-api:test .
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=key \
  -e ALPHA_VANTAGE_API_KEY=key \
  -e PORT=8080 \
  stock-api:test

# Test endpoint
curl http://localhost:8080/health
```

### Railway Deployment Test

```bash
# Check service status
railway status

# View logs
railway logs -s api-service --tail 100

# Test API
curl https://api-service-production.up.railway.app/health
```

## 📚 Documentation

- **[RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)** - Complete deployment guide
  - Prerequisites
  - Step-by-step deployment
  - Testing procedures
  - Monitoring setup
  - Migration strategy

- **[RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md)** - Environment variables
  - Required variables per service
  - Optional configurations
  - Database add-ons
  - AWS alternatives

## 🔄 Migration Path

### Phase 1: Deploy Railway (Week 1)
1. Deploy all three services
2. Configure environment variables
3. Test with production AWS resources
4. Monitor logs and performance

### Phase 2: Parallel Operation (Week 2-3)
1. Run Railway alongside AWS
2. Compare API responses
3. Validate accuracy metrics
4. Monitor costs

### Phase 3: Traffic Migration (Week 4)
1. Update DNS to Railway API
2. Gradual traffic shift (10% → 50% → 100%)
3. Monitor error rates
4. Keep AWS as fallback

### Phase 4: AWS Decommission (Week 5+)
1. Stop EventBridge schedules
2. Disable Lambda functions
3. Delete API Gateway
4. Optional: Migrate DynamoDB → PostgreSQL

## 🛠️ Common Issues

### Service Won't Start
```bash
# Check build logs
railway logs -s api-service --deployment <id>

# Verify environment variables
railway variables -s api-service
```

### AWS Connection Issues
```bash
# Test AWS credentials
railway run -s api-service -- aws sts get-caller-identity

# Check DynamoDB access
railway run -s api-service -- python -c "import boto3; ..."
```

### Worker Not Executing
```bash
# Check worker logs
railway logs -s data-ingestion --tail 200

# Verify schedule execution
railway logs -s data-ingestion | grep "completed successfully"
```

## 📞 Support

- **Railway Documentation**: https://docs.railway.app/
- **Railway CLI**: https://docs.railway.app/develop/cli
- **Community Forum**: https://help.railway.app/
- **GitHub Issues**: Open an issue in this repository

## ✅ Deployment Checklist

- [ ] Install Railway CLI
- [ ] Create Railway project
- [ ] Deploy API service
- [ ] Deploy data-ingestion worker
- [ ] Deploy model-tuning worker
- [ ] Configure environment variables (all services)
- [ ] Add PostgreSQL add-on (optional)
- [ ] Add Redis add-on (optional)
- [ ] Enable public domain for API
- [ ] Test health endpoints
- [ ] Verify data ingestion logs
- [ ] Verify model tuning logs
- [ ] Set up monitoring alerts
- [ ] Test complete workflow
- [ ] Document API endpoint URL
- [ ] Update application clients

## 🎯 Next Steps

1. Read [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md) for detailed instructions
2. Configure environment variables from [RAILWAY_ENV_VARS.md](./RAILWAY_ENV_VARS.md)
3. Deploy services using Railway CLI
4. Test API endpoints
5. Monitor worker logs
6. Scale resources as needed

---

**Cost Savings**: ~$160-190/month (65-77% reduction)
**Deployment Time**: ~2-3 hours
**Migration Time**: ~4-5 weeks (phased approach)
