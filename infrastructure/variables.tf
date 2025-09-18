variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API Key"
  type        = string
  sensitive   = true
}

variable "newsapi_key" {
  description = "NewsAPI.org API Key for news sentiment analysis"
  type        = string
  sensitive   = true
  default     = ""
}

variable "finnhub_key" {
  description = "Finnhub API Key for news sentiment analysis"
  type        = string
  sensitive   = true
  default     = ""
}

variable "lambda_function_names" {
  description = "List of Lambda function names that need sentiment cache access"
  type        = list(string)
  default     = [
    "enhanced-feature-extractor",
    "news-sentiment-analyzer",
    "ml-model-inference-tier",
    "stock-data-ingestion"
  ]
}

variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.r5.large" # Upgraded for Tier 1 capacity
}

variable "db_backup_retention_period" {
  description = "Number of days to retain database backups"
  type        = number
  default     = 3
}

variable "db_deletion_protection" {
  description = "Enable deletion protection for database"
  type        = bool
  default     = true
}

variable "valkey_node_type" {
  description = "Valkey node type"
  type        = string
  default     = "cache.r6g.large" # Upgraded for Tier 1 capacity
}

variable "valkey_num_cache_clusters" {
  description = "Number of Valkey cache clusters"
  type        = number
  default     = 1
}

variable "sagemaker_instance_type" {
  description = "SageMaker instance type"
  type        = string
  default     = "ml.m5.large" # Upgraded for Tier 1 capacity
}

variable "sagemaker_instance_count" {
  description = "Number of SageMaker instances"
  type        = number
  default     = 2 # Increased for high availability
}

variable "lambda_memory_size" {
  description = "Lambda function memory size"
  type        = number
  default     = 2048 # Upgraded for Tier 1 capacity
}

variable "lambda_timeout" {
  description = "Lambda function timeout"
  type        = number
  default     = 300 # Increased for complex ML processing
}

variable "lambda_log_retention" {
  description = "Lambda CloudWatch logs retention in days"
  type        = number
  default     = 3
}

variable "api_gateway_log_retention" {
  description = "API Gateway CloudWatch logs retention in days"
  type        = number
  default     = 3
}

variable "s3_transition_glacier_days" {
  description = "Days before transitioning S3 objects to Glacier"
  type        = number
  default     = 15
}

variable "s3_expiration_days" {
  description = "Days before S3 objects expire"
  type        = number
  default     = 180
}

variable "enable_spot_instances" {
  description = "Enable Spot instances where possible"
  type        = bool
  default     = true
}

variable "enable_graviton_instances" {
  description = "Use ARM-based Graviton instances where possible"
  type        = bool
  default     = true
}

# Swagger UI and Documentation variables
variable "enable_cloudfront_for_docs" {
  description = "Enable CloudFront distribution for Swagger UI documentation"
  type        = bool
  default     = false
}

variable "enable_cloudfront_for_api" {
  description = "Enable CloudFront distribution for API Gateway caching"
  type        = bool
  default     = false
}

variable "enable_sagemaker_serverless" {
  description = "Enable SageMaker serverless inference"
  type        = bool
  default     = false
}

variable "disable_sagemaker_entirely" {
  description = "Completely disable SageMaker infrastructure (Lambda ML only)"
  type        = bool
  default     = false
}

variable "sagemaker_serverless_memory_mb" {
  description = "Memory size in MB for SageMaker serverless endpoint (1024-6144)"
  type        = number
  default     = 1024
}

variable "sagemaker_serverless_max_concurrency" {
  description = "Maximum concurrency for SageMaker serverless endpoint"
  type        = number
  default     = 5
}

variable "enable_ai_analytics" {
  description = "Enable AI performance analytics and model tuning features"
  type        = bool
  default     = true
}

variable "ai_validation_schedule" {
  description = "Schedule expression for AI performance validation"
  type        = string
  default     = "cron(0 11 * * ? *)" # 6 AM EST daily (11:00 UTC)
}

variable "additional_tags" {
  description = "Additional resource tags"
  type        = map(string)
  default     = {}
}

# Note: SigNoz variables moved to signoz-variables.tf to avoid conflicts

variable "stock_data_ingestion_schedule" {
  description = "MAXIMIZED schedule for stock data ingestion - runs every 5 minutes during market hours"
  type        = string
  default     = "cron(*/5 14-21 ? * MON-FRI *)" # Every 5 minutes from 9:00 AM to 4:00 PM EST (14-21 UTC) - 96 runs per day
}

variable "recommendation_ttl_hours" {
  description = "TTL for active recommendations in hours (filters stale recommendations from API)"
  type        = number
  default     = 24 # 24 hours - recommendations expire after 1 trading day
}

variable "recommendation_max_age_hours" {
  description = "Maximum age for recommendations in hours (absolute safety limit)"
  type        = number
  default     = 48 # 48 hours - absolute maximum age, never show recommendations older than 2 days
}

variable "cost_alarm_threshold" {
  description = "Cost alarm threshold in USD"
  type        = number
  default     = 50 # Alert when monthly costs exceed $50
}

variable "cost_alert_email" {
  description = "Email address for cost alerts (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_nat_gateway" {
  description = "Create NAT Gateway for private subnet internet egress"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Domain name for the API (e.g., api.example.com)"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ARN of SSL certificate for the domain (must be in us-east-1 for API Gateway)"
  type        = string
  default     = ""
}

variable "hosted_zone_id" {
  description = "Route 53 hosted zone ID for the domain"
  type        = string
  default     = ""
}

variable "use_premium_api_key" {
  description = "Use premium Alpha Vantage API key for higher rate limits"
  type        = bool
  default     = true
}

variable "premium_api_calls_per_minute" {
  description = "Premium Alpha Vantage API calls per minute limit"
  type        = number
  default     = 75
}

variable "enable_lambda_provisioned_concurrency" {
  description = "Enable provisioned concurrency for Lambda functions"
  type        = bool
  default     = true
}

variable "lambda_provisioned_concurrency_count" {
  description = "Number of provisioned concurrent executions"
  type        = number
  default     = 5
}

variable "notification_email" {
  description = "Email address for daily analytics reports"
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address for CloudWatch alarm notifications"
  type        = string
  sensitive   = true
}

# Public subnet for NAT (1 AZ is enough)
resource "aws_subnet" "public_subnet_1" {
  count                   = var.enable_nat_gateway ? 1 : 0
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.10.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(
    {
      Name = "public-subnet-1"
    },
    var.additional_tags
  )
}
