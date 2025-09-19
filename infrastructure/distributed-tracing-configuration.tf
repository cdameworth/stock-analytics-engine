# Comprehensive Distributed Tracing Configuration
# Implements end-to-end X-Ray tracing across EventBridge, API Gateway, and Lambda functions

# API Gateway X-Ray tracing will be enabled via the main stage definition

# Configure X-Ray sampling rules for the stock analytics application
resource "aws_xray_sampling_rule" "stock_analytics_sampling" {
  rule_name      = "StockAnalyticsHighImportance"
  priority       = 1000
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.1  # 10% sampling rate
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "stock-analytics*"  # Matches all our services
  resource_arn   = "*"

  # Custom attributes for financial services
  attributes = {
    "business.domain" = "financial"
    "system.type"     = "stock-analytics"
  }
}

# High priority sampling for critical business operations
resource "aws_xray_sampling_rule" "critical_operations_sampling" {
  rule_name      = "CriticalFinancialOperations"
  priority       = 500
  version        = 1
  reservoir_size = 2
  fixed_rate     = 1.0  # 100% sampling for critical operations
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"

  # Target specific critical operations
  attributes = {
    "operation.criticality" = "high"
    "business.domain"       = "financial"
  }
}

# EventBridge sampling rule for event-driven flows
resource "aws_xray_sampling_rule" "eventbridge_sampling" {
  rule_name      = "EventBridgeFinancialEvents"
  priority       = 800
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.3  # 30% sampling for EventBridge events
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"

  attributes = {
    "source.type"         = "eventbridge"
    "business.domain"     = "financial"
  }
}

# Local variables for X-Ray configuration
locals {
  # Common X-Ray environment variables for all Lambda functions
  xray_environment_base = {
    # Enable X-Ray tracing
    _X_AMZN_TRACE_ID                    = ""

    # X-Ray specific configuration
    AWS_XRAY_TRACING_NAME               = "stock-analytics"
    AWS_XRAY_DEBUG_MODE                 = var.enable_debug_logging ? "TRUE" : "FALSE"
    AWS_XRAY_SDK_ENABLED                = "true"

    # Configure X-Ray for financial domain
    XRAY_TRACE_ID_HEADER                = "X-Amzn-Trace-Id"

    # Business-aware tracing configuration
    ENABLE_DISTRIBUTED_TRACING          = "true"
    TRACE_BUSINESS_CONTEXT              = "true"
    FINANCIAL_DOMAIN_TRACING            = "true"
  }

  # Function-specific X-Ray configurations
  xray_service_configurations = {
    stock_data_ingestion = {
      AWS_XRAY_TRACING_NAME = "stock-data-ingestion"
      TRACE_OPERATION_TYPE  = "data_ingestion"
      BUSINESS_CRITICALITY  = "high"
    }

    stock_recommendations_api = {
      AWS_XRAY_TRACING_NAME = "stock-recommendations-api"
      TRACE_OPERATION_TYPE  = "api_request"
      BUSINESS_CRITICALITY  = "critical"
    }

    ml_model_inference = {
      AWS_XRAY_TRACING_NAME = "ml-model-inference"
      TRACE_OPERATION_TYPE  = "ml_inference"
      BUSINESS_CRITICALITY  = "high"
    }

    price_prediction_model = {
      AWS_XRAY_TRACING_NAME = "price-prediction-model"
      TRACE_OPERATION_TYPE  = "prediction"
      BUSINESS_CRITICALITY  = "high"
    }

    time_prediction_model = {
      AWS_XRAY_TRACING_NAME = "time-prediction-model"
      TRACE_OPERATION_TYPE  = "prediction"
      BUSINESS_CRITICALITY  = "medium"
    }

    dual_accuracy_tracker = {
      AWS_XRAY_TRACING_NAME = "dual-accuracy-tracker"
      TRACE_OPERATION_TYPE  = "analytics"
      BUSINESS_CRITICALITY  = "medium"
    }

    dual_prediction_reporting_api = {
      AWS_XRAY_TRACING_NAME = "dual-prediction-reporting-api"
      TRACE_OPERATION_TYPE  = "reporting"
      BUSINESS_CRITICALITY  = "medium"
    }
  }

  # Complete environment variables combining OTEL, X-Ray, and SigNoz
  complete_tracing_environment = {
    stock_data_ingestion = merge(
      local.get_lambda_environment.stock_data_ingestion,
      local.xray_environment_base,
      local.xray_service_configurations.stock_data_ingestion
    )

    stock_recommendations_api = merge(
      local.get_lambda_environment.stock_recommendations_api,
      local.xray_environment_base,
      local.xray_service_configurations.stock_recommendations_api
    )

    ml_model_inference = merge(
      local.get_lambda_environment.ml_model_inference,
      local.xray_environment_base,
      local.xray_service_configurations.ml_model_inference
    )

    dual_accuracy_tracker = merge(
      local.get_lambda_environment.dual_accuracy_tracker,
      local.xray_environment_base,
      local.xray_service_configurations.dual_accuracy_tracker
    )

    dual_prediction_reporting_api = merge(
      local.get_lambda_environment.dual_prediction_reporting_api,
      local.xray_environment_base,
      local.xray_service_configurations.dual_prediction_reporting_api
    )
  }
}

# CloudWatch Log Group for X-Ray trace insights
resource "aws_cloudwatch_log_group" "xray_trace_insights" {
  name              = "/aws/xray/stock-analytics-insights"
  retention_in_days = var.xray_log_retention_days

  tags = merge(
    {
      Name = "x-ray-trace-insights"
      Service = "observability"
      TraceType = "distributed"
    },
    var.additional_tags
  )
}

# CloudWatch Insights Queries for X-Ray analysis
resource "aws_cloudwatch_query_definition" "financial_trace_analysis" {
  name = "Stock Analytics - Financial Trace Analysis"

  log_group_names = [
    aws_cloudwatch_log_group.xray_trace_insights.name
  ]

  query_string = <<EOF
fields @timestamp, @message
| filter @message like /stock-analytics/
| filter @message like /financial/
| stats count() by bin(5m)
| sort @timestamp desc
| limit 100
EOF
}

resource "aws_cloudwatch_query_definition" "eventbridge_trace_flow" {
  name = "Stock Analytics - EventBridge Trace Flow"

  log_group_names = [
    "/aws/lambda/stock-data-ingestion",
    "/aws/lambda/ml-model-inference-lowcost",
    "/aws/lambda/price-prediction-model",
    "/aws/lambda/time-prediction-model"
  ]

  query_string = <<EOF
fields @timestamp, @requestId, @message
| filter @message like /eventbridge/ or @message like /X-Amzn-Trace-Id/
| sort @timestamp asc
| limit 200
EOF
}

# X-Ray service map configuration
resource "aws_cloudwatch_dashboard" "xray_service_map" {
  dashboard_name = "StockAnalytics-XRay-ServiceMap"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2
        properties = {
          markdown = "# Stock Analytics - Distributed Tracing Dashboard\n\nComprehensive view of X-Ray traces across EventBridge, API Gateway, and Lambda functions"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/X-Ray", "TracesReceived"],
            [".", "TracesProcessed"],
            [".", "LatencyHigh"],
            [".", "ErrorRate"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "X-Ray Trace Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 2
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/ApiGateway", "Latency", "ApiName", aws_api_gateway_rest_api.stock_recommendations_api.name],
            [".", "4XXError", ".", "."],
            [".", "5XXError", ".", "."],
            [".", "Count", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = data.aws_region.current.name
          title   = "API Gateway with X-Ray Tracing"
          period  = 300
        }
      }
    ]
  })
}

# Output X-Ray configuration details
output "distributed_tracing_configuration" {
  description = "Distributed tracing configuration details"
  value = {
    api_gateway_xray_enabled = aws_api_gateway_stage.stock_recommendations_api_stage_with_tracing.xray_tracing_enabled
    xray_sampling_rules = [
      aws_xray_sampling_rule.stock_analytics_sampling.rule_name,
      aws_xray_sampling_rule.critical_operations_sampling.rule_name,
      aws_xray_sampling_rule.eventbridge_sampling.rule_name
    ]
    cloudwatch_insights_queries = [
      aws_cloudwatch_query_definition.financial_trace_analysis.name,
      aws_cloudwatch_query_definition.eventbridge_trace_flow.name
    ]
    service_map_dashboard = aws_cloudwatch_dashboard.xray_service_map.dashboard_name
    trace_insights_log_group = aws_cloudwatch_log_group.xray_trace_insights.name
  }
}