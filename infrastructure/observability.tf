# OpenTelemetry and Observability Infrastructure for Stock Analytics Engine
# Integrates with Grafana Cloud and AWS native monitoring

# ADOT (AWS Distro for OpenTelemetry) Collector Lambda Layer
locals {
  # Use our custom OpenTelemetry Lambda Layer ARN
  adot_layer_arn = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:layer:stock-analytics-otel-python:1"

  # OpenTelemetry Python instrumentations - use our custom layer
  otel_python_layer_arn = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:layer:stock-analytics-otel-python:1"

  # Common OpenTelemetry environment variables
  otel_base_config = {
    # OpenTelemetry configuration
    OTEL_PROPAGATORS                 = "tracecontext,baggage,xray"
    OTEL_PYTHON_DISTRO               = "aws_distro"
    OTEL_PYTHON_CONFIGURATOR         = "aws_lambda_configurator"
    OTEL_LAMBDA_DISABLE_AWS_CONTEXT_PROPAGATION = "false"

    # Grafana Cloud OTLP endpoint configuration
    OTEL_EXPORTER_OTLP_ENDPOINT      = var.grafana_otlp_endpoint
    OTEL_EXPORTER_OTLP_HEADERS       = "Authorization=Basic ${base64encode("${var.grafana_instance_id}:${var.grafana_api_key}")}"
    OTEL_EXPORTER_OTLP_PROTOCOL      = "grpc"

    # Service naming and resource attributes
    OTEL_SERVICE_NAME                = "stock-analytics-engine"
    OTEL_RESOURCE_ATTRIBUTES         = "service.namespace=stock-analytics,deployment.environment=${var.environment}"

    # Sampling configuration for cost optimization
    OTEL_TRACES_SAMPLER              = "traceidratio"
    OTEL_TRACES_SAMPLER_ARG          = var.otel_trace_sampling_ratio

    # Metrics configuration
    OTEL_METRICS_EXPORTER            = "otlp"
    OTEL_LOGS_EXPORTER              = "otlp"

    # AWS X-Ray integration
    AWS_LAMBDA_EXEC_WRAPPER         = "/opt/otel-instrument"
    _AWS_LAMBDA_TELEMETRY_LOG_FD    = "1"
  }
}

# CloudWatch OTEL Collector Configuration
resource "aws_ssm_parameter" "otel_collector_config" {
  name  = "/stock-analytics/otel/collector-config"
  type  = "String"
  tier  = "Advanced"
  value = templatefile("${path.module}/otel-collector-config.yaml", {
    grafana_otlp_endpoint = var.grafana_otlp_endpoint
    grafana_auth_header  = base64encode("${var.grafana_instance_id}:${var.grafana_api_key}")
    environment          = var.environment
  })

  description = "OpenTelemetry Collector configuration for Grafana Cloud integration"

  tags = merge(
    {
      Name = "otel-collector-config"
      Purpose = "observability"
    },
    var.additional_tags
  )
}

# Custom metrics namespace for business logic
resource "aws_cloudwatch_log_group" "otel_metrics" {
  name              = "/aws/otel/stock-analytics"
  retention_in_days = var.otel_log_retention_days

  tags = merge(
    {
      Name = "otel-metrics-logs"
      Purpose = "observability"
    },
    var.additional_tags
  )
}

# Lambda Layer for OpenTelemetry Python instrumentation
resource "aws_lambda_layer_version" "otel_python_instrumentation" {
  filename   = "${path.module}/otel-python-layer.zip"
  layer_name = "stock-analytics-otel-python"

  compatible_runtimes = ["python3.11"]
  description         = "OpenTelemetry Python instrumentation for Stock Analytics"

  depends_on = [data.archive_file.otel_python_layer]
}

# Archive for custom OpenTelemetry Python layer with instrumentations
data "archive_file" "otel_python_layer" {
  type        = "zip"
  output_path = "${path.module}/otel-python-layer.zip"

  source {
    content = templatefile("${path.module}/otel-layer-requirements.txt", {})
    filename = "requirements.txt"
  }

  source {
    content = templatefile("${path.module}/otel-python-installer.sh", {})
    filename = "install.sh"
  }
}

# Enhanced IAM policy for OpenTelemetry operations
resource "aws_iam_role_policy" "lambda_otel_permissions" {
  name = "lambda-otel-observability-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets",
          "xray:GetSamplingStatisticSummaries"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = [
          aws_ssm_parameter.otel_collector_config.arn,
          "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/stock-analytics/otel/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          aws_cloudwatch_log_group.otel_metrics.arn,
          "${aws_cloudwatch_log_group.otel_metrics.arn}:*"
        ]
      }
    ]
  })
}

# CloudWatch Dashboard for OpenTelemetry Metrics
resource "aws_cloudwatch_dashboard" "otel_observability" {
  dashboard_name = "StockAnalytics-OpenTelemetry-Overview"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", "stock-data-ingestion"],
            ["AWS/Lambda", "Duration", "FunctionName", "ml-model-inference-lowcost"],
            ["AWS/Lambda", "Duration", "FunctionName", "stock-recommendations-api"],
            ["AWS/Lambda", "Duration", "FunctionName", "dual-prediction-reporting-api"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Lambda Function Durations"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Errors", "FunctionName", "stock-data-ingestion"],
            ["AWS/Lambda", "Errors", "FunctionName", "ml-model-inference-lowcost"],
            ["AWS/Lambda", "Errors", "FunctionName", "stock-recommendations-api"],
            ["AWS/Lambda", "Errors", "FunctionName", "dual-prediction-reporting-api"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Lambda Function Errors"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 24
        height = 6

        properties = {
          metrics = [
            ["StockAnalytics/DataIngestion", "SymbolsProcessed"],
            ["StockAnalytics/DataIngestion", "APICallsUsed"],
            ["StockAnalytics/MLInference", "PredictionsGenerated"],
            ["StockAnalytics/MLInference", "AccuracyScore"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Business Metrics"
          period  = 300
        }
      }
    ]
  })
}

# Application Performance Monitoring alarms
resource "aws_cloudwatch_metric_alarm" "high_lambda_error_rate" {
  for_each = toset([
    "stock-data-ingestion",
    "ml-model-inference-lowcost",
    "stock-recommendations-api",
    "dual-prediction-reporting-api"
  ])

  alarm_name          = "high-error-rate-${each.value}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "High error rate detected for ${each.value}"
  alarm_actions       = [aws_sns_topic.observability_alerts.arn]

  dimensions = {
    FunctionName = each.value
  }

  tags = merge(
    {
      Name = "high-error-rate-${each.value}"
      Purpose = "observability"
    },
    var.additional_tags
  )
}

# SNS topic for observability alerts
resource "aws_sns_topic" "observability_alerts" {
  name = "stock-analytics-observability-alerts"

  tags = merge(
    {
      Name = "observability-alerts"
      Purpose = "monitoring"
    },
    var.additional_tags
  )
}

# SNS subscription for observability alerts
resource "aws_sns_topic_subscription" "observability_alerts_email" {
  count     = var.observability_alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.observability_alerts.arn
  protocol  = "email"
  endpoint  = var.observability_alert_email
}

# X-Ray service map and tracing
resource "aws_xray_sampling_rule" "stock_analytics_sampling" {
  rule_name      = "StockAnalyticsSampling"
  priority       = 9000
  version        = 1
  reservoir_size = 1
  fixed_rate     = var.xray_sampling_rate
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "stock-analytics-*"
  resource_arn   = "*"

  tags = merge(
    {
      Name = "stock-analytics-sampling"
      Purpose = "observability"
    },
    var.additional_tags
  )
}