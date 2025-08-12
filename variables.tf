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

variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.t4g.small"
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
  default     = "cache.t4g.micro"
}

variable "valkey_num_cache_clusters" {
  description = "Number of Valkey cache clusters"
  type        = number
  default     = 1
}

variable "sagemaker_instance_type" {
  description = "SageMaker instance type"
  type        = string
  default     = "ml.t2.small"
}

variable "sagemaker_instance_count" {
  description = "Number of SageMaker instances"
  type        = number
  default     = 1
}

variable "lambda_memory_size" {
  description = "Lambda function memory size"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Lambda function timeout"
  type        = number
  default     = 120
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

variable "enable_sagemaker_serverless" {
  description = "Enable SageMaker serverless inference"
  type        = bool
  default     = false
}

variable "additional_tags" {
  description = "Additional resource tags"
  type        = map(string)
  default     = {}
}

variable "stock_data_ingestion_schedule" {
  description = "Schedule expression for stock data ingestion (EventBridge)"
  type        = string
  default     = "cron(0 14,18,22 ? * MON-FRI *)"  # Market hours: 9 AM, 1 PM, 5 PM EST
}

variable "cost_alarm_threshold" {
  description = "Cost alarm threshold in USD"
  type        = number
  default     = 50  # Alert when monthly costs exceed $50
}

variable "cost_alert_email" {
  description = "Email address for cost alerts (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

