# SigNoz Cloud Configuration Variables for Stock Analytics Engine

# SigNoz Cloud Configuration
variable "signoz_otlp_endpoint" {
  description = "SigNoz Cloud OTLP endpoint (region-specific)"
  type        = string
  default     = "ingest.us.signoz.cloud:443"
  validation {
    condition = can(regex("^ingest\\.(us|eu|in)\\.signoz\\.cloud:443$", var.signoz_otlp_endpoint))
    error_message = "SigNoz OTLP endpoint must be a valid SigNoz Cloud endpoint (ingest.{region}.signoz.cloud:443)."
  }
}

variable "signoz_ingestion_key" {
  description = "SigNoz Cloud ingestion key for authentication"
  type        = string
  sensitive   = true
  validation {
    condition     = length(var.signoz_ingestion_key) > 0
    error_message = "SigNoz ingestion key cannot be empty."
  }
}

variable "signoz_api_key" {
  description = "SigNoz Cloud API key for management operations"
  type        = string
  default     = ""
  sensitive   = true
}

variable "signoz_organization_id" {
  description = "SigNoz Cloud organization ID"
  type        = string
  default     = ""
}

# OpenTelemetry Configuration (inherits from existing variables)
variable "enable_signoz_integration" {
  description = "Enable SigNoz Cloud integration"
  type        = bool
  default     = true
}

variable "enable_rds_monitoring" {
  description = "Enable RDS PostgreSQL monitoring with SigNoz"
  type        = bool
  default     = true
}

variable "enable_lambda_monitoring" {
  description = "Enable Lambda function monitoring with SigNoz"
  type        = bool
  default     = true
}

# Infrastructure Configuration
variable "enable_ecs_otel_collector" {
  description = "Deploy OpenTelemetry Collector on ECS for enhanced metrics collection"
  type        = bool
  default     = false
}

variable "otel_collector_cpu" {
  description = "CPU allocation for ECS OTEL Collector task"
  type        = number
  default     = 512
  validation {
    condition = contains([256, 512, 1024, 2048, 4096], var.otel_collector_cpu)
    error_message = "CPU must be one of: 256, 512, 1024, 2048, 4096."
  }
}

variable "otel_collector_memory" {
  description = "Memory allocation for ECS OTEL Collector task"
  type        = number
  default     = 1024
  validation {
    condition = var.otel_collector_memory >= 512 && var.otel_collector_memory <= 8192
    error_message = "Memory must be between 512 and 8192 MB."
  }
}

# CloudWatch Exporter Configuration
variable "cloudwatch_exporter_image" {
  description = "Docker image for Prometheus CloudWatch Exporter"
  type        = string
  default     = "prom/cloudwatch-exporter:v0.15.5"
}

variable "cloudwatch_scrape_interval" {
  description = "CloudWatch metrics scrape interval in seconds"
  type        = number
  default     = 300
  validation {
    condition = var.cloudwatch_scrape_interval >= 60 && var.cloudwatch_scrape_interval <= 3600
    error_message = "Scrape interval must be between 60 and 3600 seconds."
  }
}

# Data Collection Configuration
variable "postgresql_metrics_collection_interval" {
  description = "PostgreSQL metrics collection interval in seconds"
  type        = number
  default     = 60
  validation {
    condition = var.postgresql_metrics_collection_interval >= 30 && var.postgresql_metrics_collection_interval <= 300
    error_message = "Collection interval must be between 30 and 300 seconds."
  }
}

variable "logs_poll_interval" {
  description = "CloudWatch logs polling interval"
  type        = string
  default     = "1m"
  validation {
    condition = can(regex("^[0-9]+[smh]$", var.logs_poll_interval))
    error_message = "Poll interval must be in format like '1m', '30s', '2h'."
  }
}

# Cost Optimization
variable "signoz_data_retention_days" {
  description = "Data retention period in SigNoz (days)"
  type        = number
  default     = 15
  validation {
    condition = var.signoz_data_retention_days >= 1 && var.signoz_data_retention_days <= 90
    error_message = "Retention period must be between 1 and 90 days."
  }
}

variable "enable_log_sampling" {
  description = "Enable log sampling to reduce ingestion costs"
  type        = bool
  default     = true
}

variable "log_sampling_rate" {
  description = "Log sampling rate (0.0 to 1.0)"
  type        = number
  default     = 0.1
  validation {
    condition = var.log_sampling_rate >= 0.0 && var.log_sampling_rate <= 1.0
    error_message = "Sampling rate must be between 0.0 and 1.0."
  }
}

variable "enable_batch_processing" {
  description = "Enable batch processing for telemetry data"
  type        = bool
  default     = true
}

variable "batch_size" {
  description = "Batch size for telemetry data processing"
  type        = number
  default     = 1000
  validation {
    condition = var.batch_size >= 100 && var.batch_size <= 10000
    error_message = "Batch size must be between 100 and 10000."
  }
}

# Security Configuration
variable "enable_tls_verification" {
  description = "Enable TLS certificate verification for SigNoz endpoints"
  type        = bool
  default     = true
}

variable "enable_sensitive_data_filtering" {
  description = "Enable filtering of sensitive data in telemetry"
  type        = bool
  default     = true
}

variable "pii_filtering_patterns" {
  description = "Regex patterns for filtering PII from logs and traces"
  type        = list(string)
  default = [
    ".*(password|secret|token|key|api_key).*",
    ".*\\b\\d{3}-\\d{2}-\\d{4}\\b.*",  # SSN pattern
    ".*\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b.*"  # Credit card pattern
  ]
}

# Monitoring Configuration
variable "enable_otel_collector_monitoring" {
  description = "Enable monitoring of OpenTelemetry Collector itself"
  type        = bool
  default     = true
}

variable "otel_collector_health_check_interval" {
  description = "Health check interval for OTEL Collector in seconds"
  type        = number
  default     = 30
}

variable "enable_cloudwatch_integration" {
  description = "Keep CloudWatch integration alongside SigNoz for backup monitoring"
  type        = bool
  default     = true
}

# Alerting Configuration
variable "enable_signoz_alerts" {
  description = "Enable alerting through SigNoz"
  type        = bool
  default     = true
}

variable "alert_notification_channels" {
  description = "Notification channels for SigNoz alerts"
  type        = list(string)
  default     = []
}

# Resource Tagging
variable "signoz_resource_tags" {
  description = "Additional tags for SigNoz-related resources"
  type        = map(string)
  default = {
    "monitoring.platform" = "signoz"
    "data.classification" = "telemetry"
    "cost.center"        = "observability"
  }
}

# Development and Debugging
variable "enable_debug_logging" {
  description = "Enable debug logging for OpenTelemetry components"
  type        = bool
  default     = false
}

variable "otel_log_level" {
  description = "Log level for OpenTelemetry components"
  type        = string
  default     = "info"
  validation {
    condition = contains(["debug", "info", "warn", "error"], var.otel_log_level)
    error_message = "Log level must be one of: debug, info, warn, error."
  }
}

# Migration Configuration
variable "enable_dual_shipping" {
  description = "Enable dual shipping to both Grafana and SigNoz during migration"
  type        = bool
  default     = false
}

variable "migration_phase" {
  description = "Current migration phase: planning, testing, dual-shipping, cutover, cleanup"
  type        = string
  default     = "planning"
  validation {
    condition = contains(["planning", "testing", "dual-shipping", "cutover", "cleanup"], var.migration_phase)
    error_message = "Migration phase must be one of: planning, testing, dual-shipping, cutover, cleanup."
  }
}