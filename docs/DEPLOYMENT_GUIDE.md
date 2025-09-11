# Stock Analytics Engine - Deployment Guide

This guide explains how to deploy different tiers of the Stock Analytics Engine infrastructure using Terraform configurations.

## Available Deployment Tiers

### Current Configuration (~$45/month)
- **File**: `terraform-current.tfvars`
- **Use Case**: Development, testing, proof of concept
- **Features**: Basic functionality with minimal resources
- **API Limits**: Free Alpha Vantage (5 calls/minute)

### Tier 1: Stability Fix (~$245/month) **[RECOMMENDED FOR PRODUCTION]**
- **File**: `terraform-tier1.tfvars`
- **Use Case**: Production deployment with stability and moderate scale
- **Features**: Premium API, enhanced caching, multi-instance SageMaker
- **API Limits**: Premium Alpha Vantage (75 calls/minute)

### Tier 2: High Scale (~$985/month)
- **File**: `terraform-tier2.tfvars`
- **Use Case**: High-traffic applications, enterprise features
- **Features**: Multi-AZ deployment, advanced monitoring, high concurrency
- **API Limits**: Premium Alpha Vantage (75 calls/minute)

### Tier 3: Enterprise Scale (~$3,200/month)
- **File**: `terraform-tier3.tfvars`
- **Use Case**: Global deployment, maximum performance
- **Features**: Multi-region, enterprise compliance, maximum resources
- **API Limits**: Premium Alpha Vantage (consider Professional upgrade)

## Deployment Instructions

### Prerequisites

1. **AWS CLI configured with stock-analytics-admin profile**:
   ```bash
   aws configure --profile stock-analytics-admin
   ```

2. **Terraform installed** (v1.0+)

3. **Premium Alpha Vantage API Key** (for Tier 1+):
   - Already created in AWS Secrets Manager: `stock-analytics-alpha-vantage-premium-api-key`
   - Value: `GAIDJHNQOZSBALES`

### Deploy Tier 1 (Recommended)

```bash
# 1. Initialize Terraform (if not done already)
terraform init

# 2. Plan deployment with Tier 1 configuration
terraform plan -var-file="terraform-tier1.tfvars"

# 3. Apply Tier 1 configuration
terraform apply -var-file="terraform-tier1.tfvars"

# 4. Verify deployment
terraform output -var-file="terraform-tier1.tfvars"
```

### Deploy Other Tiers

```bash
# For Current (Development) Configuration
terraform plan -var-file="terraform-current.tfvars"
terraform apply -var-file="terraform-current.tfvars"

# For Tier 2 (High Scale)
terraform plan -var-file="terraform-tier2.tfvars"
terraform apply -var-file="terraform-tier2.tfvars"

# For Tier 3 (Enterprise Scale)
terraform plan -var-file="terraform-tier3.tfvars"
terraform apply -var-file="terraform-tier3.tfvars"
```

### Upgrade Between Tiers

To upgrade from one tier to another:

```bash
# Example: Upgrade from Current to Tier 1
terraform plan -var-file="terraform-tier1.tfvars"
terraform apply -var-file="terraform-tier1.tfvars"

# Example: Downgrade from Tier 1 to Current
terraform plan -var-file="terraform-current.tfvars"
terraform apply -var-file="terraform-current.tfvars"
```

## Configuration Customization

### Modify tfvars Files

Edit the appropriate `.tfvars` file to customize settings:

```bash
# Edit Tier 1 configuration
nano terraform-tier1.tfvars

# Key settings to customize:
# - cost_alert_email: Add your email for cost alerts
# - domain_name: Set custom domain (Tier 2+ only)
# - additional_tags: Add organization-specific tags
```

### Override Specific Variables

You can override specific variables without modifying the tfvars files:

```bash
# Override cost alert email for Tier 1
terraform apply -var-file="terraform-tier1.tfvars" -var="cost_alert_email=admin@company.com"

# Override domain settings for Tier 2
terraform apply -var-file="terraform-tier2.tfvars" \
  -var="domain_name=api.company.com" \
  -var="certificate_arn=arn:aws:acm:us-east-1:123456789:certificate/abc123"
```

## Monitoring and Verification

### After Deployment

1. **Check API Endpoint**:
   ```bash
   # Get API Gateway URL
   terraform output api_gateway_url
   
   # Test API endpoint
   curl "$(terraform output -raw api_gateway_url)/recommendations"
   ```

2. **Verify Lambda Functions**:
   ```bash
   aws lambda list-functions --profile stock-analytics-admin \
     --query 'Functions[?contains(FunctionName, `stock`)].{Name:FunctionName,Runtime:Runtime,Memory:MemorySize}' \
     --output table
   ```

3. **Check SageMaker Endpoint**:
   ```bash
   aws sagemaker list-endpoints --profile stock-analytics-admin \
     --query 'Endpoints[?contains(EndpointName, `stock`)].{Name:EndpointName,Status:EndpointStatus}' \
     --output table
   ```

4. **Monitor Costs**:
   ```bash
   # Check current month costs
   aws ce get-cost-and-usage --profile stock-analytics-admin \
     --time-period Start=2025-08-01,End=2025-08-31 \
     --granularity MONTHLY \
     --metrics BlendedCost \
     --group-by Type=DIMENSION,Key=SERVICE
   ```

## Troubleshooting

### Common Issues

1. **Terraform State Lock**:
   ```bash
   # If deployment is stuck, check for locks
   terraform force-unlock <LOCK_ID>
   ```

2. **Resource Name Conflicts**:
   ```bash
   # If resources already exist, import them
   terraform import aws_secretsmanager_secret.alpha_vantage_premium_api_key stock-analytics-alpha-vantage-premium-api-key
   ```

3. **API Key Issues**:
   ```bash
   # Update the premium API key in Secrets Manager
   aws secretsmanager update-secret \
     --secret-id stock-analytics-alpha-vantage-premium-api-key \
     --secret-string "GAIDJHNQOZSBALES" \
     --profile stock-analytics-admin
   ```

### Rollback Strategy

If deployment fails or needs to be rolled back:

```bash
# 1. Rollback to previous tfvars configuration
terraform apply -var-file="terraform-current.tfvars"

# 2. Or destroy and recreate (data loss warning!)
terraform destroy -var-file="terraform-tier1.tfvars"
terraform apply -var-file="terraform-current.tfvars"
```

## Cost Management

### Set Up Billing Alerts

Add your email to the tfvars file:

```hcl
cost_alert_email = "admin@company.com"
```

### Regular Cost Monitoring

```bash
# Weekly cost check
aws ce get-cost-and-usage --profile stock-analytics-admin \
  --time-period Start=2025-08-16,End=2025-08-23 \
  --granularity DAILY \
  --metrics BlendedCost
```

### Cost Optimization Tips

1. **Use Tier 1 for most production workloads**
2. **Monitor Lambda memory usage and adjust accordingly**
3. **Consider spot instances for batch processing (Tier 2+)**
4. **Implement auto-scaling policies**
5. **Set up CloudWatch cost anomaly detection**

## Next Steps

After successful deployment:

1. **Update Lambda functions** with enhanced stock coverage
2. **Configure monitoring dashboards**
3. **Set up automated testing**
4. **Implement CI/CD pipeline**
5. **Configure backup and disaster recovery**

## Support

For issues or questions:
1. Check CloudWatch logs for Lambda functions
2. Review Terraform plan output before applying
3. Monitor CloudWatch metrics and alarms
4. Check AWS Cost Explorer for unexpected charges