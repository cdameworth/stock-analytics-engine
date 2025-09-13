# Stock Analytics Engine API Documentation

This document provides information about the Swagger/OpenAPI documentation for the Stock Analytics Engine.

## 📖 Overview

The Stock Analytics Engine provides a comprehensive REST API for accessing stock recommendations, ML predictions, and performance analytics. All endpoints are documented using OpenAPI 3.0 specification and served through an interactive Swagger UI.

## 🚀 Quick Deployment

Deploy the API documentation using the provided script:

```bash
# Deploy complete Swagger UI infrastructure
./scripts/deploy-swagger-docs.sh deploy

# Update only the documentation files
./scripts/deploy-swagger-docs.sh update

# Enable CloudFront for better performance
./scripts/deploy-swagger-docs.sh cloudfront

# Show deployment information
./scripts/deploy-swagger-docs.sh info
```

## 🔗 API Endpoints

The API includes the following endpoints:

### Stock Recommendations
- `GET /recommendations` - Get all stock recommendations
- `GET /recommendations/{symbol}` - Get recommendation for specific stock symbol

### Analytics & Performance
- `GET /analytics/dashboard` - Get comprehensive performance dashboard
- `GET /dual-predictions/analytics` - Get dual prediction model analytics

## 🔑 Authentication

All API endpoints require authentication via API key:

```bash
curl -H "x-api-key: YOUR_API_KEY" \
     "https://your-api-gateway-url/prod/recommendations"
```

### Rate Limits
- **100 requests per second**
- **200 burst limit**
- **10,000 requests per month**

## 📊 Performance Targets

The system is designed to achieve:
- **Hit Rate**: 65% accuracy threshold for deployment
- **Market Outperformance**: 3% above S&P 500 benchmark
- **Risk Management**: Sharpe ratio >1.0, max drawdown <15%

## 🏗️ Infrastructure

The Swagger UI is hosted on AWS using:

### Core Components
- **S3 Static Website**: Hosts Swagger UI and OpenAPI specification
- **API Gateway**: Serves the REST API with authentication
- **CloudFront** (optional): CDN for improved performance

### Files Structure
```
docs/
├── swagger.yaml              # OpenAPI 3.0 specification
infrastructure/
├── swagger_ui.tf            # Terraform for hosting infrastructure
├── swagger-ui-template.html # Custom Swagger UI template
scripts/
├── deploy-swagger-docs.sh   # Deployment automation script
```

## 🔧 Customization

### Updating API Documentation

1. **Edit the OpenAPI spec**: Modify `docs/swagger.yaml`
2. **Update infrastructure**: Run the deployment script
3. **Test changes**: Visit the Swagger UI URL

### Custom Styling

The Swagger UI uses a custom template (`infrastructure/swagger-ui-template.html`) with:
- Stock Analytics branding
- Quick start guide
- API key persistence
- Enhanced error handling

### CloudFront Integration

Enable CloudFront for better performance:

```bash
# Enable CloudFront for documentation
./scripts/deploy-swagger-docs.sh cloudfront
```

## 📱 Usage Examples

### Get All Recommendations
```bash
curl -H "x-api-key: YOUR_API_KEY" \
     "https://your-api-gateway-url/prod/recommendations?limit=10"
```

### Get Specific Stock
```bash
curl -H "x-api-key: YOUR_API_KEY" \
     "https://your-api-gateway-url/prod/recommendations/AAPL"
```

### Get Performance Dashboard
```bash
curl -H "x-api-key: YOUR_API_KEY" \
     "https://your-api-gateway-url/prod/analytics/dashboard"
```

## 🔍 Response Examples

### Stock Recommendation Response
```json
{
  "recommendation_id": "AAPL_20250913_143022",
  "symbol": "AAPL",
  "prediction": "BUY",
  "confidence": 0.87,
  "target_price": 185.50,
  "current_price": 175.23,
  "upside_potential": 0.0586,
  "timestamp": "2025-09-13T14:30:22Z",
  "model_version": "v2.1",
  "features_used": 42,
  "risk_score": 0.23
}
```

### Analytics Dashboard Response
```json
{
  "performance_summary": {
    "current_hit_rate": 0.672,
    "target_hit_rate": 0.65,
    "market_outperformance": 0.032,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.087
  },
  "model_metrics": {
    "price_model": {
      "accuracy": 0.672,
      "prediction_count": 1247,
      "avg_confidence": 0.78
    },
    "time_model": {
      "accuracy": 0.634,
      "prediction_count": 1189,
      "avg_confidence": 0.71
    }
  }
}
```

## 🛠️ Development

### Prerequisites
- Terraform >= 1.0
- AWS CLI configured with `stock-analytics-admin` profile
- Python 3 (for validation scripts)

### Local Testing

Test the OpenAPI specification locally:

```bash
# Validate YAML syntax (requires PyYAML)
python3 -c "import yaml; yaml.safe_load(open('docs/swagger.yaml'))"

# Start local Swagger UI (requires Docker)
docker run -p 8080:8080 -v $(pwd)/docs:/docs swaggerapi/swagger-ui \
    SWAGGER_JSON=/docs/swagger.yaml
```

### Terraform Integration

The Swagger UI infrastructure is integrated with the main Terraform configuration:

```hcl
# Deploy with documentation
terraform apply -target="module.swagger_ui"

# Update documentation only
terraform apply -target="aws_s3_object.swagger_spec"
```

## 🚨 Troubleshooting

### Common Issues

1. **S3 bucket access denied**: Check bucket policy and public access settings
2. **CloudFront not updating**: Wait 10-15 minutes for distribution deployment
3. **API Gateway CORS errors**: Verify CORS headers in Terraform configuration
4. **Swagger UI not loading**: Check S3 website configuration and file uploads

### Debug Commands

```bash
# Check S3 bucket website status
aws s3api get-bucket-website --bucket your-swagger-bucket

# Test API Gateway endpoints
curl -v -H "x-api-key: YOUR_KEY" "https://your-api-url/prod/recommendations"

# View CloudFront distribution status
aws cloudfront get-distribution --id YOUR_DISTRIBUTION_ID
```

## 📈 Monitoring

The documentation infrastructure includes:
- **S3 access logs** for usage tracking
- **CloudWatch metrics** for performance monitoring
- **Cost alerts** for budget management

## 🔄 Updates

To update the API documentation:

1. **Modify** `docs/swagger.yaml`
2. **Run** `./scripts/deploy-swagger-docs.sh update`
3. **Verify** changes at the Swagger UI URL

The deployment script automatically handles:
- File validation
- S3 uploads
- Cache invalidation (if CloudFront enabled)
- URL distribution

## 📞 Support

For issues with the API documentation:
1. Check the troubleshooting section above
2. Review CloudWatch logs for errors
3. Validate the OpenAPI specification syntax
4. Test API endpoints independently

The Swagger UI provides an interactive way to explore and test the Stock Analytics Engine API, making it easier for developers to integrate with the platform and understand the available functionality.