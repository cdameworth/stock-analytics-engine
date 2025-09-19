# Minimal SigNoz Observability Integration
# Core monitoring for AWS services without complex configurations

# CloudWatch Log Groups for SigNoz integration
resource "aws_cloudwatch_log_group" "signoz_api_gateway_logs" {
  name              = "/aws/apigateway/stock-recommendations-signoz"
  retention_in_days = var.api_gateway_log_retention

  tags = merge(
    {
      Name = "signoz-api-gateway-logs"
      Service = "api-gateway"
      ObservabilityTarget = "signoz"
    },
    var.additional_tags
  )
}

# SNS Topic for SigNoz Observability Alerts
resource "aws_sns_topic" "signoz_observability_alerts" {
  name = "stock-analytics-signoz-observability-alerts"

  tags = merge(
    {
      Name = "signoz-observability-alerts"
      Service = "sns"
      Purpose = "signoz-alerting"
    },
    var.additional_tags
  )
}

resource "aws_sns_topic_subscription" "signoz_observability_alerts_email" {
  count     = var.cost_alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.signoz_observability_alerts.arn
  protocol  = "email"
  endpoint  = var.cost_alert_email
}

# Basic CloudWatch Alarms for SigNoz integration
resource "aws_cloudwatch_metric_alarm" "signoz_api_gateway_errors" {
  alarm_name          = "signoz-api-gateway-errors-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4XXError"
  namespace           = "AWS/ApiGateway"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "API Gateway 4XX errors are high (SigNoz monitoring)"
  alarm_actions       = [aws_sns_topic.signoz_observability_alerts.arn]

  dimensions = {
    ApiName   = aws_api_gateway_rest_api.stock_recommendations_api.name
    Stage     = aws_api_gateway_stage.stock_recommendations_api_stage.stage_name
  }

  tags = merge(
    {
      Name = "signoz-api-gateway-errors"
      Service = "api-gateway"
      AlertType = "error"
      MonitoringTarget = "signoz"
    },
    var.additional_tags
  )
}

resource "aws_cloudwatch_metric_alarm" "signoz_elasticache_cpu" {
  alarm_name          = "signoz-elasticache-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "ElastiCache CPU utilization is high (SigNoz monitoring)"
  alarm_actions       = [aws_sns_topic.signoz_observability_alerts.arn]

  dimensions = {
    CacheClusterId = "${aws_elasticache_replication_group.stock_analytics_valkey.replication_group_id}-001"
  }

  tags = merge(
    {
      Name = "signoz-elasticache-cpu"
      Service = "elasticache"
      AlertType = "performance"
      MonitoringTarget = "signoz"
    },
    var.additional_tags
  )
}

resource "aws_cloudwatch_metric_alarm" "signoz_rds_cpu" {
  alarm_name          = "signoz-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "RDS CPU utilization is high (SigNoz monitoring)"
  alarm_actions       = [aws_sns_topic.signoz_observability_alerts.arn]

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.stock_analytics_aurora.cluster_identifier
  }

  tags = merge(
    {
      Name = "signoz-rds-cpu"
      Service = "rds"
      AlertType = "performance"
      MonitoringTarget = "signoz"
    },
    var.additional_tags
  )
}

# Data source for CloudWatch to SigNoz forwarder
data "archive_file" "cloudwatch_signoz_forwarder" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/cloudwatch_signoz_forwarder.py"
  output_path = "${path.module}/cloudwatch_signoz_forwarder.zip"
}

# Lambda Log Forwarding Function for SigNoz
resource "aws_lambda_function" "signoz_log_forwarder" {
  count            = var.enable_signoz_integration ? 1 : 0
  filename         = data.archive_file.cloudwatch_signoz_forwarder.output_path
  source_code_hash = data.archive_file.cloudwatch_signoz_forwarder.output_base64sha256
  function_name    = "signoz-log-forwarder"
  role             = aws_iam_role.signoz_log_forwarder_role[0].arn
  handler          = "cloudwatch_signoz_forwarder.lambda_handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      SIGNOZ_ENDPOINT = var.signoz_otlp_endpoint
      SIGNOZ_TOKEN    = var.signoz_ingestion_key
      LOG_LEVEL       = "INFO"
    }
  }

  tags = merge(
    {
      Name = "signoz-log-forwarder"
      Service = "observability"
      Purpose = "log-forwarding"
      MonitoringTarget = "signoz"
    },
    var.additional_tags
  )
}

resource "aws_iam_role" "signoz_log_forwarder_role" {
  count = var.enable_signoz_integration ? 1 : 0
  name  = "signoz-log-forwarder-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    {
      Name = "signoz-log-forwarder-role"
      Service = "observability"
      MonitoringTarget = "signoz"
    },
    var.additional_tags
  )
}

resource "aws_iam_role_policy" "signoz_log_forwarder_policy" {
  count = var.enable_signoz_integration ? 1 : 0
  name  = "signoz-log-forwarder-policy"
  role  = aws_iam_role.signoz_log_forwarder_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# CloudWatch Dashboard for SigNoz Integration Status
resource "aws_cloudwatch_dashboard" "signoz_observability_status" {
  dashboard_name = "SigNoz-Observability-Status"

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
            ["AWS/ApiGateway", "Count", "ApiName", aws_api_gateway_rest_api.stock_recommendations_api.name],
            [".", "4XXError", ".", "."],
            [".", "5XXError", ".", "."],
            [".", "Latency", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "API Gateway Metrics (SigNoz Monitored)"
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
            ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", "${aws_elasticache_replication_group.stock_analytics_valkey.replication_group_id}-001"],
            [".", "DatabaseMemoryUsagePercentage", ".", "."],
            [".", "CurrConnections", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "ElastiCache Metrics (SigNoz Monitored)"
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
            ["AWS/RDS", "CPUUtilization", "DBClusterIdentifier", aws_rds_cluster.stock_analytics_aurora.cluster_identifier],
            [".", "DatabaseConnections", ".", "."],
            [".", "FreeStorageSpace", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "RDS Metrics (SigNoz Monitored)"
          period  = 300
        }
      }
    ]
  })

  depends_on = [
    aws_api_gateway_rest_api.stock_recommendations_api,
    aws_elasticache_replication_group.stock_analytics_valkey,
    aws_rds_cluster.stock_analytics_aurora
  ]
}

# Output information about SigNoz integration
# CloudWatch Log Subscription Filters for RDS and SNS
resource "aws_cloudwatch_log_subscription_filter" "rds_logs_to_signoz" {
  count           = var.enable_signoz_integration ? 1 : 0
  name            = "rds-logs-to-signoz"
  log_group_name  = "/aws/rds/instance/${aws_rds_cluster.stock_analytics_aurora.cluster_identifier}/postgresql"
  filter_pattern  = ""  # Forward all logs
  destination_arn = aws_lambda_function.signoz_log_forwarder[0].arn

  depends_on = [aws_lambda_permission.allow_cloudwatch_rds_logs]
}

resource "aws_cloudwatch_log_subscription_filter" "lambda_logs_to_signoz" {
  count           = var.enable_signoz_integration ? 1 : 0
  name            = "lambda-logs-to-signoz"
  log_group_name  = "/aws/lambda/stock-recommendations-api"
  filter_pattern  = ""  # Forward all logs
  destination_arn = aws_lambda_function.signoz_log_forwarder[0].arn

  depends_on = [aws_lambda_permission.allow_cloudwatch_lambda_logs]
}

# Lambda permissions for CloudWatch log subscription filters
resource "aws_lambda_permission" "allow_cloudwatch_rds_logs" {
  count         = var.enable_signoz_integration ? 1 : 0
  statement_id  = "AllowExecutionFromCloudWatchRDS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.signoz_log_forwarder[0].function_name
  principal     = "logs.amazonaws.com"
  source_arn    = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/rds/instance/${aws_rds_cluster.stock_analytics_aurora.cluster_identifier}/postgresql:*"
}

resource "aws_lambda_permission" "allow_cloudwatch_lambda_logs" {
  count         = var.enable_signoz_integration ? 1 : 0
  statement_id  = "AllowExecutionFromCloudWatchLambda"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.signoz_log_forwarder[0].function_name
  principal     = "logs.amazonaws.com"
  source_arn    = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/stock-recommendations-api:*"
}

output "signoz_observability_status" {
  description = "SigNoz observability integration status"
  value = {
    api_gateway_monitoring = "CloudWatch alarms configured for SigNoz integration"
    elasticache_monitoring = "CPU and memory alarms configured"
    rds_monitoring = "Database performance alarms configured"
    sns_topic_arn = aws_sns_topic.signoz_observability_alerts.arn
    dashboard_name = aws_cloudwatch_dashboard.signoz_observability_status.dashboard_name
    lambda_functions_with_otel = [
      "stock-data-ingestion",
      "ml-model-inference-lowcost",
      "stock-recommendations-api",
      "dual-accuracy-tracker",
      "dual-prediction-reporting-api"
    ]
    signoz_endpoint = var.enable_signoz_integration ? var.signoz_otlp_endpoint : "not configured"
    log_forwarder = var.enable_signoz_integration ? aws_lambda_function.signoz_log_forwarder[0].function_name : "disabled"
    log_subscriptions = var.enable_signoz_integration ? [
      "RDS PostgreSQL logs",
      "Lambda function logs"
    ] : []
  }
}