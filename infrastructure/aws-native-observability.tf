# AWS Native Observability Integration for Stock Analytics Engine
# Configures CloudWatch, X-Ray, and other AWS observability services

# Enhanced CloudWatch Dashboards for comprehensive monitoring
resource "aws_cloudwatch_dashboard" "stock_analytics_comprehensive" {
  dashboard_name = "StockAnalytics-Comprehensive-Monitoring"

  dashboard_body = jsonencode({
    widgets = [
      # Lambda Performance Overview
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 24
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Duration", "FunctionName", "stock-data-ingestion"],
            [".", ".", ".", "ml-model-inference-lowcost"],
            [".", ".", ".", "stock-recommendations-api"],
            [".", ".", ".", "dual-prediction-reporting-api"],
            [".", "Invocations", ".", "stock-data-ingestion"],
            [".", ".", ".", "ml-model-inference-lowcost"],
            [".", ".", ".", "stock-recommendations-api"],
            [".", "Errors", ".", "stock-data-ingestion"],
            [".", ".", ".", "ml-model-inference-lowcost"],
            [".", ".", ".", "stock-recommendations-api"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Lambda Functions - Performance Overview"
          period  = 300
          yAxis = {
            left = {
              min = 0
            }
          }
        }
      },

      # API Gateway Metrics
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiName", "stock-recommendations-api"],
            [".", "Latency", ".", "."],
            [".", "4XXError", ".", "."],
            [".", "5XXError", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "API Gateway Performance"
          period  = 300
        }
      },

      # DynamoDB Performance
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/DynamoDB", "ConsumedReadCapacityUnits", "TableName", "stock-recommendations"],
            [".", "ConsumedWriteCapacityUnits", ".", "."],
            [".", "SuccessfulRequestLatency", ".", ".", "Operation", "Query"],
            [".", ".", ".", ".", ".", "PutItem"],
            [".", ".", ".", ".", ".", "GetItem"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "DynamoDB Performance"
          period  = 300
        }
      },

      # ElastiCache Valkey Metrics
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", "stock-analytics-valkey-lowcost-001"],
            [".", "DatabaseMemoryUsagePercentage", ".", "."],
            [".", "CacheHits", ".", "."],
            [".", "CacheMisses", ".", "."],
            [".", "NetworkBytesIn", ".", "."],
            [".", "NetworkBytesOut", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "ElastiCache Valkey Performance"
          period  = 300
        }
      },

      # Aurora RDS Performance
      {
        type   = "metric"
        x      = 12
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBClusterIdentifier", "stock-analytics-aurora"],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."],
            [".", "FreeableMemory", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Aurora PostgreSQL Performance"
          period  = 300
        }
      },

      # Business Metrics
      {
        type   = "metric"
        x      = 0
        y      = 18
        width  = 24
        height = 6

        properties = {
          metrics = [
            ["StockAnalytics/DataIngestion", "SymbolsProcessed"],
            [".", "APICallsUsed"],
            [".", "APICallsRemaining"],
            ["StockAnalytics/MLInference", "PredictionsGenerated"],
            [".", "AccuracyScore"],
            ["StockAnalytics/Business", "TradingRecommendations"],
            [".", "PortfolioValue"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "Stock Analytics Business Metrics"
          period  = 300
        }
      }
    ]
    period = 300
    start  = "-PT3H"
    end    = "PT0H"
  })
}

# X-Ray Service Map Configuration
resource "aws_xray_sampling_rule" "stock_analytics_detailed" {
  rule_name      = "StockAnalyticsDetailed"
  priority       = 8000
  version        = 1
  reservoir_size = 2
  fixed_rate     = var.xray_sampling_rate
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "stock-*"
  resource_arn   = "*"

  tags = merge(
    {
      Name = "stock-analytics-detailed-sampling"
      Purpose = "detailed-observability"
    },
    var.additional_tags
  )
}

# CloudWatch Composite Alarms for System Health
resource "aws_cloudwatch_composite_alarm" "system_health" {
  alarm_name        = "StockAnalytics-SystemHealth"
  alarm_description = "Composite alarm monitoring overall system health"

  alarm_rule = join(" OR ", [
    "ALARM(${aws_cloudwatch_metric_alarm.lambda_error_rate_critical.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.api_gateway_high_latency.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.dynamodb_throttling.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.valkey_memory_high.alarm_name})"
  ])

  alarm_actions = [aws_sns_topic.critical_alerts.arn]
  ok_actions    = [aws_sns_topic.critical_alerts.arn]

  tags = merge(
    {
      Name = "system-health-composite"
      Severity = "critical"
    },
    var.additional_tags
  )
}

# Critical Lambda Error Rate Alarm
resource "aws_cloudwatch_metric_alarm" "lambda_error_rate_critical" {
  alarm_name          = "StockAnalytics-Lambda-ErrorRate-Critical"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  threshold           = "0.05"  # 5% error rate

  metric_query {
    id          = "e1"
    return_data = true

    metric {
      metric_name = "Errors"
      namespace   = "AWS/Lambda"
      period      = "300"
      stat        = "Sum"

      dimensions = {
        FunctionName = "stock-data-ingestion"
      }
    }
  }

  metric_query {
    id          = "e2"
    return_data = false

    metric {
      metric_name = "Invocations"
      namespace   = "AWS/Lambda"
      period      = "300"
      stat        = "Sum"

      dimensions = {
        FunctionName = "stock-data-ingestion"
      }
    }
  }

  metric_query {
    id          = "error_rate"
    return_data = true
    expression  = "e1/e2"
    label       = "Error Rate"
  }

  alarm_description = "Critical error rate in stock data ingestion Lambda"
  alarm_actions     = [aws_sns_topic.critical_alerts.arn]

  tags = merge(
    {
      Name = "lambda-error-rate-critical"
      Severity = "critical"
    },
    var.additional_tags
  )
}

# API Gateway High Latency Alarm
resource "aws_cloudwatch_metric_alarm" "api_gateway_high_latency" {
  alarm_name          = "StockAnalytics-APIGateway-HighLatency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "Latency"
  namespace           = "AWS/ApiGateway"
  period              = "300"
  statistic           = "Average"
  threshold           = "5000"  # 5 seconds
  alarm_description   = "High latency detected in API Gateway"
  alarm_actions       = [aws_sns_topic.observability_alerts.arn]

  dimensions = {
    ApiName = "stock-recommendations-api"
  }

  tags = merge(
    {
      Name = "api-gateway-high-latency"
      Severity = "warning"
    },
    var.additional_tags
  )
}

# DynamoDB Throttling Alarm
resource "aws_cloudwatch_metric_alarm" "dynamodb_throttling" {
  alarm_name          = "StockAnalytics-DynamoDB-Throttling"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "UserErrors"
  namespace           = "AWS/DynamoDB"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "DynamoDB throttling detected"
  alarm_actions       = [aws_sns_topic.critical_alerts.arn]

  dimensions = {
    TableName = "stock-recommendations"
  }

  tags = merge(
    {
      Name = "dynamodb-throttling"
      Severity = "critical"
    },
    var.additional_tags
  )
}

# Valkey Memory Usage Alarm
resource "aws_cloudwatch_metric_alarm" "valkey_memory_high" {
  alarm_name          = "StockAnalytics-Valkey-MemoryHigh"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"  # 80% memory usage
  alarm_description   = "High memory usage in Valkey cluster"
  alarm_actions       = [aws_sns_topic.observability_alerts.arn]

  dimensions = {
    CacheClusterId = "stock-analytics-valkey-lowcost-001"
  }

  tags = merge(
    {
      Name = "valkey-memory-high"
      Severity = "warning"
    },
    var.additional_tags
  )
}

# SNS Topic for Critical Alerts
resource "aws_sns_topic" "critical_alerts" {
  name = "stock-analytics-critical-alerts"

  tags = merge(
    {
      Name = "critical-alerts"
      Purpose = "monitoring"
    },
    var.additional_tags
  )
}

# SNS subscription for critical alerts
resource "aws_sns_topic_subscription" "critical_alerts_email" {
  count     = var.observability_alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.critical_alerts.arn
  protocol  = "email"
  endpoint  = var.observability_alert_email
}

# CloudWatch Insights Queries for Business Intelligence
resource "aws_cloudwatch_query_definition" "business_performance_analysis" {
  name = "Stock-Analytics-Business-Performance"

  log_group_names = [
    "/aws/lambda/stock-data-ingestion",
    "/aws/lambda/ml-model-inference-lowcost",
    "/aws/lambda/stock-recommendations-api"
  ]

  query_string = <<-EOT
    fields @timestamp, @message, @requestId
    | filter @message like /prediction_accuracy/ or @message like /symbols_processed/ or @message like /api_calls_used/
    | parse @message /prediction_accuracy=(?<accuracy>[0-9.]+)/
    | parse @message /symbols_processed=(?<symbols>[0-9]+)/
    | parse @message /api_calls_used=(?<api_calls>[0-9]+)/
    | stats
        avg(accuracy) as avg_accuracy,
        sum(symbols) as total_symbols,
        sum(api_calls) as total_api_calls
      by bin(5m)
    | sort @timestamp desc
  EOT
}

resource "aws_cloudwatch_query_definition" "performance_bottleneck_analysis" {
  name = "Stock-Analytics-Performance-Bottlenecks"

  log_group_names = [
    "/aws/lambda/stock-data-ingestion",
    "/aws/lambda/ml-model-inference-lowcost",
    "/aws/lambda/stock-recommendations-api"
  ]

  query_string = <<-EOT
    fields @timestamp, @duration, @requestId, @message
    | filter @type = "REPORT"
    | stats
        avg(@duration) as avg_duration,
        max(@duration) as max_duration,
        min(@duration) as min_duration,
        count() as execution_count
      by bin(5m)
    | sort @timestamp desc
  EOT
}

# Custom Metrics for Business Logic
resource "aws_cloudwatch_log_metric_filter" "ml_accuracy_metric" {
  name           = "MLModelAccuracy"
  log_group_name = "/aws/lambda/ml-model-inference-lowcost"
  pattern        = "[timestamp, requestId, level=\"INFO\", message=\"MODEL_ACCURACY\", accuracy_score]"

  metric_transformation {
    name      = "AccuracyScore"
    namespace = "StockAnalytics/MLInference"
    value     = "$accuracy_score"
  }
}

resource "aws_cloudwatch_log_metric_filter" "symbols_processed_metric" {
  name           = "SymbolsProcessed"
  log_group_name = "/aws/lambda/stock-data-ingestion"
  pattern        = "[timestamp, requestId, level=\"INFO\", message=\"SYMBOLS_PROCESSED\", count]"

  metric_transformation {
    name      = "SymbolsProcessed"
    namespace = "StockAnalytics/DataIngestion"
    value     = "$count"
  }
}

resource "aws_cloudwatch_log_metric_filter" "api_calls_metric" {
  name           = "APICallsUsed"
  log_group_name = "/aws/lambda/stock-data-ingestion"
  pattern        = "[timestamp, requestId, level=\"INFO\", message=\"API_CALLS_USED\", calls_used, calls_remaining]"

  metric_transformation {
    name      = "APICallsUsed"
    namespace = "StockAnalytics/DataIngestion"
    value     = "$calls_used"
  }
}

# Cost Optimization: CloudWatch Logs Retention Policies
resource "aws_cloudwatch_log_group" "cost_optimized_logs" {
  for_each = {
    "/aws/apigateway/access-logs"     = 7
    "/aws/lambda/debug-logs"          = 3
    "/aws/xray/traces"               = 14
    "/aws/otel/collector"            = 7
  }

  name              = each.key
  retention_in_days = each.value

  tags = merge(
    {
      Name = each.key
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# EventBridge Rules for Observability Events
resource "aws_cloudwatch_event_rule" "observability_health_check" {
  name                = "stock-analytics-health-check"
  description         = "Regular health check for stock analytics system"
  schedule_expression = "rate(5 minutes)"

  tags = merge(
    {
      Name = "observability-health-check"
      Purpose = "monitoring"
    },
    var.additional_tags
  )
}

# Lambda function for health check (simple example)
data "archive_file" "health_check_lambda" {
  type        = "zip"
  output_path = "${path.module}/health_check.zip"

  source {
    content = <<-EOT
import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Simple health check - verify key services are accessible
        dynamodb = boto3.client('dynamodb')
        elasticache = boto3.client('elasticache')

        # Check DynamoDB table
        dynamodb.describe_table(TableName='stock-recommendations')

        # Check ElastiCache cluster
        elasticache.describe_replication_groups(
            ReplicationGroupId='stock-analytics-valkey-lowcost'
        )

        logger.info("Health check passed - all services accessible")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'healthy',
                'timestamp': context.aws_request_id,
                'services': {
                    'dynamodb': 'healthy',
                    'elasticache': 'healthy'
                }
            })
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")

        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': context.aws_request_id
            })
        }
    EOT
    filename = "health_check.py"
  }
}

resource "aws_lambda_function" "health_check" {
  filename         = data.archive_file.health_check_lambda.output_path
  source_code_hash = data.archive_file.health_check_lambda.output_base64sha256
  function_name    = "stock-analytics-health-check"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "health_check.lambda_handler"
  runtime          = "python3.11"
  timeout          = 30

  environment {
    variables = {
      ENVIRONMENT = var.environment
    }
  }

  tags = merge(
    {
      Name = "health-check"
      Purpose = "monitoring"
    },
    var.additional_tags
  )
}

# EventBridge target for health check
resource "aws_cloudwatch_event_target" "health_check_target" {
  rule      = aws_cloudwatch_event_rule.observability_health_check.name
  target_id = "HealthCheckTarget"
  arn       = aws_lambda_function.health_check.arn
}

# Lambda permission for EventBridge health check
resource "aws_lambda_permission" "allow_eventbridge_health_check" {
  statement_id  = "AllowExecutionFromEventBridgeHealthCheck"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.health_check.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.observability_health_check.arn
}