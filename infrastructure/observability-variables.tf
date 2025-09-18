# OpenTelemetry and Observability Variables for Stock Analytics Engine

# Grafana Cloud Configuration
variable "grafana_instance_id" {
  description = "Grafana Cloud instance ID (username for OTLP)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "grafana_aws_access_key" {
  description = "AWS Access Key for Grafana CloudWatch data source"
  type        = string
  default     = ""
  sensitive   = true
}

variable "grafana_aws_secret_key" {
  description = "AWS Secret Key for Grafana CloudWatch data source"
  type        = string
  default     = ""
  sensitive   = true
}

variable "grafana_api_key" {
  description = "Grafana Cloud API key for OTLP authentication"
  type        = string
  default     = ""
  sensitive   = true
}

variable "grafana_otlp_endpoint" {
  description = "Grafana Cloud OTLP endpoint"
  type        = string
  default     = "https://otlp-gateway-prod-us-central-0.grafana.net/otlp"
}

variable "grafana_prometheus_endpoint" {
  description = "Grafana Cloud Prometheus remote write endpoint"
  type        = string
  default     = ""
}

variable "grafana_loki_endpoint" {
  description = "Grafana Cloud Loki logs endpoint"
  type        = string
  default     = ""
}

variable "grafana_tempo_endpoint" {
  description = "Grafana Cloud Tempo traces endpoint"
  type        = string
  default     = ""
}

# OpenTelemetry Configuration
variable "otel_trace_sampling_ratio" {
  description = "OpenTelemetry trace sampling ratio (0.0 to 1.0)"
  type        = string
  default     = "0.1"  # 10% sampling for cost optimization
}

variable "otel_log_retention_days" {
  description = "OpenTelemetry logs retention period in days"
  type        = number
  default     = 14
}

variable "enable_otel_auto_instrumentation" {
  description = "Enable OpenTelemetry auto-instrumentation for Lambda functions"
  type        = bool
  default     = true
}

variable "enable_custom_metrics" {
  description = "Enable custom business metrics collection"
  type        = bool
  default     = true
}

variable "otel_resource_attributes" {
  description = "Additional OpenTelemetry resource attributes as key-value pairs"
  type        = map(string)
  default = {
    "service.namespace" = "stock-analytics"
    "business.domain"   = "financial-trading"
    "data.classification" = "sensitive"
  }
}

# AWS X-Ray Configuration
variable "enable_xray_tracing" {
  description = "Enable AWS X-Ray tracing for Lambda functions"
  type        = bool
  default     = true
}

variable "xray_sampling_rate" {
  description = "AWS X-Ray sampling rate (0.0 to 1.0)"
  type        = number
  default     = 0.1  # 10% sampling
}

# Monitoring and Alerting
variable "observability_alert_email" {
  description = "Email address for observability alerts"
  type        = string
  default     = ""
}

variable "enable_performance_alarms" {
  description = "Enable CloudWatch alarms for performance monitoring"
  type        = bool
  default     = true
}

variable "lambda_error_threshold" {
  description = "Threshold for Lambda error rate alarms"
  type        = number
  default     = 5
}

variable "lambda_duration_threshold" {
  description = "Threshold for Lambda duration alarms in milliseconds"
  type        = number
  default     = 10000  # 10 seconds
}

# Cost Optimization
variable "observability_cost_budget" {
  description = "Monthly budget for observability costs in USD"
  type        = number
  default     = 50
}

variable "enable_otel_batch_processing" {
  description = "Enable batch processing for OpenTelemetry data to reduce costs"
  type        = bool
  default     = true
}

variable "metrics_resolution" {
  description = "CloudWatch metrics resolution in seconds (60 or 1 for high-resolution)"
  type        = number
  default     = 60
}

# Data Retention
variable "traces_retention_days" {
  description = "X-Ray traces retention period in days"
  type        = number
  default     = 30
}

variable "metrics_retention_days" {
  description = "CloudWatch metrics retention period in days"
  type        = number
  default     = 90
}

# Environment-specific Configuration
variable "observability_environment" {
  description = "Environment name for observability tagging"
  type        = string
  default     = "production"
}

variable "enable_debug_logging" {
  description = "Enable debug logging for OpenTelemetry (development only)"
  type        = bool
  default     = false
}

# Business Metrics Configuration
variable "business_metrics_namespace" {
  description = "CloudWatch namespace for custom business metrics"
  type        = string
  default     = "StockAnalytics/Business"
}

variable "enable_ml_model_metrics" {
  description = "Enable ML model performance metrics collection"
  type        = bool
  default     = true
}

variable "enable_trading_metrics" {
  description = "Enable trading performance metrics collection"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_sensitive_data_filtering" {
  description = "Enable filtering of sensitive data in telemetry"
  type        = bool
  default     = true
}

variable "pii_scrubbing_patterns" {
  description = "Regex patterns for scrubbing PII from logs and traces"
  type        = list(string)
  default = [
    "api_key=\\w+",
    "password=\\w+",
    "ssn=\\d{3}-\\d{2}-\\d{4}",
    "credit_card=\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}"
  ]
}