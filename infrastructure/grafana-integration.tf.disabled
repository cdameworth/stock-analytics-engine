# Grafana Cloud Integration Configuration for Stock Analytics Engine
# Configures data sources, dashboards, and alerting via Terraform

# Grafana provider configuration
provider "grafana" {
  url  = "https://${var.grafana_instance_id}.grafana.net"
  auth = var.grafana_api_key
}

# AWS CloudWatch data source in Grafana
resource "grafana_data_source" "cloudwatch" {
  type = "cloudwatch"
  name = "AWS CloudWatch - Stock Analytics"

  json_data_encoded = jsonencode({
    authType = "keys"
    defaultRegion = data.aws_region.current.name
    customMetricsNamespaces = "AWS/Lambda,AWS/DynamoDB,AWS/ElastiCache,AWS/ApiGateway,StockAnalytics/DataIngestion,StockAnalytics/MLInference,StockAnalytics/Business"
    tracesDs = grafana_data_source.xray.uid
    logGroups = [
      {
        logGroupName = "/aws/lambda/stock-data-ingestion"
        region = data.aws_region.current.name
      }
    ]
    derivedFields = [
      {
        name = "X-Ray Trace"
        datasourceUid = grafana_data_source.xray.uid
        matcherRegex = "XRAY TraceId: (1-[0-9a-f]{8}-[0-9a-f]{24})"
        url = "$${__value.raw}"
        urlDisplayLabel = "View X-Ray trace"
      }
    ]
  })

  secure_json_data_encoded = jsonencode({
    accessKey = var.grafana_aws_access_key
    secretKey = var.grafana_aws_secret_key
  })

  depends_on = [aws_iam_role.grafana_cloudwatch_role, grafana_data_source.xray]
}

# AWS X-Ray data source in Grafana
resource "grafana_data_source" "xray" {
  type = "aws-x-ray-datasource"
  name = "AWS X-Ray - Stock Analytics"

  json_data_encoded = jsonencode({
    authType = "keys"
    defaultRegion = data.aws_region.current.name
  })

  secure_json_data_encoded = jsonencode({
    accessKey = var.grafana_aws_access_key
    secretKey = var.grafana_aws_secret_key
  })
}

# Prometheus data source for OpenTelemetry metrics
resource "grafana_data_source" "prometheus" {
  count = var.grafana_prometheus_endpoint != "" ? 1 : 0

  type = "prometheus"
  name = "Prometheus - OpenTelemetry"
  url  = var.grafana_prometheus_endpoint

  json_data_encoded = jsonencode({
    httpMethod = "GET"
    customQueryParameters = ""
    prometheusType = "Prometheus"
    prometheusVersion = "2.40.0"
  })

  secure_json_data_encoded = jsonencode({
    basicAuthPassword = var.grafana_api_key
  })

  basic_auth_enabled  = true
  basic_auth_username = var.grafana_instance_id
}

# Loki data source for logs
resource "grafana_data_source" "loki" {
  count = var.grafana_loki_endpoint != "" ? 1 : 0

  type = "loki"
  name = "Loki - Logs"
  url  = var.grafana_loki_endpoint

  json_data_encoded = jsonencode({
    maxLines = 1000
  })

  secure_json_data_encoded = jsonencode({
    basicAuthPassword = var.grafana_api_key
  })

  basic_auth_enabled  = true
  basic_auth_username = var.grafana_instance_id
}

# Tempo data source for traces
resource "grafana_data_source" "tempo" {
  count = var.grafana_tempo_endpoint != "" ? 1 : 0

  type = "tempo"
  name = "Tempo - Traces"
  url  = var.grafana_tempo_endpoint

  json_data_encoded = jsonencode({
    tracesToLogs = {
      datasourceUid = var.grafana_loki_endpoint != "" ? grafana_data_source.loki[0].uid : ""
      tags = ["traceId"]
    }
    tracesToMetrics = {
      datasourceUid = var.grafana_prometheus_endpoint != "" ? grafana_data_source.prometheus[0].uid : ""
      tags = [{key = "service.name", value = "service"}]
    }
    nodeGraph = {
      enabled = true
    }
  })

  secure_json_data_encoded = jsonencode({
    basicAuthPassword = var.grafana_api_key
  })

  basic_auth_enabled  = true
  basic_auth_username = var.grafana_instance_id
}

# IAM role for Grafana to access CloudWatch
resource "aws_iam_role" "grafana_cloudwatch_role" {
  name = "grafana-cloudwatch-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = var.grafana_instance_id
          }
        }
      }
    ]
  })

  tags = merge(
    {
      Name = "grafana-cloudwatch-access"
      Purpose = "observability"
    },
    var.additional_tags
  )
}

# IAM policy for Grafana CloudWatch access
resource "aws_iam_role_policy" "grafana_cloudwatch_policy" {
  name = "grafana-cloudwatch-access-policy"
  role = aws_iam_role.grafana_cloudwatch_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:DescribeAlarmsForMetric",
          "cloudwatch:DescribeAlarmHistory",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:ListMetrics",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetInsightRuleReport"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/*",
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/apigateway/*",
          aws_cloudwatch_log_group.otel_metrics.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeRegions",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "xray:BatchGetTraces",
          "xray:GetTraceSummaries",
          "xray:GetServiceGraph",
          "xray:GetTimeSeriesServiceStatistics"
        ]
        Resource = "*"
      }
    ]
  })
}

# Stock Analytics Overview Dashboard
resource "grafana_dashboard" "stock_analytics_overview" {
  config_json = jsonencode({
    dashboard = {
      id       = null
      title    = "Stock Analytics Engine - Overview"
      tags     = ["stock-analytics", "lambda", "overview"]
      timezone = "browser"
      panels = [
        {
          id          = 1
          title       = "Lambda Function Invocations"
          type        = "stat"
          gridPos     = { h = 8, w = 12, x = 0, y = 0 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "AWS/Lambda"
              metricName = "Invocations"
              dimensions = {
                FunctionName = "stock-data-ingestion"
              }
              statistic = "Sum"
              period    = "300"
            }
          ]
        },
        {
          id          = 2
          title       = "Lambda Function Errors"
          type        = "stat"
          gridPos     = { h = 8, w = 12, x = 12, y = 0 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "AWS/Lambda"
              metricName = "Errors"
              dimensions = {
                FunctionName = "stock-data-ingestion"
              }
              statistic = "Sum"
              period    = "300"
            }
          ]
        },
        {
          id          = 3
          title       = "API Gateway Request Count"
          type        = "timeseries"
          gridPos     = { h = 8, w = 24, x = 0, y = 8 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "AWS/ApiGateway"
              metricName = "Count"
              dimensions = {
                ApiName = "stock-recommendations-api"
              }
              statistic = "Sum"
              period    = "300"
            }
          ]
        },
        {
          id          = 4
          title       = "DynamoDB Operations"
          type        = "timeseries"
          gridPos     = { h = 8, w = 12, x = 0, y = 16 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "AWS/DynamoDB"
              metricName = "ConsumedReadCapacityUnits"
              dimensions = {
                TableName = "stock-recommendations"
              }
              statistic = "Sum"
              period    = "300"
            }
          ]
        },
        {
          id          = 5
          title       = "ElastiCache Valkey Performance"
          type        = "timeseries"
          gridPos     = { h = 8, w = 12, x = 12, y = 16 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "AWS/ElastiCache"
              metricName = "CacheHits"
              dimensions = {
                CacheClusterId = "stock-analytics-valkey-lowcost-001"
              }
              statistic = "Sum"
              period    = "300"
            }
          ]
        }
      ]
      time = {
        from = "now-1h"
        to   = "now"
      }
      refresh = "10s"
    }
  })
}

# Business Metrics Dashboard
resource "grafana_dashboard" "business_metrics" {
  config_json = jsonencode({
    dashboard = {
      id       = null
      title    = "Stock Analytics - Business Metrics"
      tags     = ["stock-analytics", "business", "ml"]
      timezone = "browser"
      panels = [
        {
          id          = 1
          title       = "ML Model Accuracy"
          type        = "stat"
          gridPos     = { h = 8, w = 8, x = 0, y = 0 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "StockAnalytics/MLInference"
              metricName = "AccuracyScore"
              statistic  = "Average"
              period     = "3600"
            }
          ]
          fieldConfig = {
            defaults = {
              color = {
                mode = "thresholds"
              }
              thresholds = {
                steps = [
                  { color = "red", value = 0 },
                  { color = "yellow", value = 0.5 },
                  { color = "green", value = 0.7 }
                ]
              }
              unit = "percentunit"
            }
          }
        },
        {
          id          = 2
          title       = "Symbols Processed Today"
          type        = "stat"
          gridPos     = { h = 8, w = 8, x = 8, y = 0 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "StockAnalytics/DataIngestion"
              metricName = "SymbolsProcessed"
              statistic  = "Sum"
              period     = "3600"
            }
          ]
        },
        {
          id          = 3
          title       = "API Calls Remaining"
          type        = "gauge"
          gridPos     = { h = 8, w = 8, x = 16, y = 0 }
          dataSource  = grafana_data_source.cloudwatch.name
          targets = [
            {
              namespace  = "StockAnalytics/DataIngestion"
              metricName = "APICallsRemaining"
              statistic  = "Maximum"
              period     = "300"
            }
          ]
          fieldConfig = {
            defaults = {
              color = {
                mode = "thresholds"
              }
              thresholds = {
                steps = [
                  { color = "red", value = 0 },
                  { color = "yellow", value = 100 },
                  { color = "green", value = 1000 }
                ]
              }
              max = 4500
              min = 0
            }
          }
        }
      ]
      time = {
        from = "now-24h"
        to   = "now"
      }
      refresh = "30s"
    }
  })
}

# Alert rules for critical metrics
resource "grafana_rule_group" "stock_analytics_alerts" {
  name             = "Stock Analytics Alerts"
  folder_uid       = grafana_folder.stock_analytics.uid
  interval_seconds = 60

  rule {
    name      = "High Lambda Error Rate"
    condition = "C"

    data {
      ref_id = "A"
      relative_time_range {
        from = 300
        to   = 0
      }
      datasource_uid = grafana_data_source.cloudwatch.uid
      model = jsonencode({
        namespace  = "AWS/Lambda"
        metricName = "Errors"
        statistic  = "Sum"
        dimensions = {
          FunctionName = "stock-data-ingestion"
        }
      })
    }

    data {
      ref_id = "B"
      relative_time_range {
        from = 300
        to   = 0
      }
      datasource_uid = grafana_data_source.cloudwatch.uid
      model = jsonencode({
        namespace  = "AWS/Lambda"
        metricName = "Invocations"
        statistic  = "Sum"
        dimensions = {
          FunctionName = "stock-data-ingestion"
        }
      })
    }

    data {
      ref_id = "C"
      relative_time_range {
        from = 0
        to   = 0
      }
      datasource_uid = "__expr__"
      model = jsonencode({
        type       = "math"
        expression = "$A / $B"
      })
    }

    annotations = {
      summary = "High error rate detected in Lambda function"
      description = "Error rate for stock-data-ingestion Lambda function exceeds 10%"
    }

    labels = {
      severity = "critical"
      service  = "stock-analytics"
    }

    for = "2m"
    no_data_state = "NoData"
    exec_err_state = "Alerting"
  }
}

# Grafana folder for organization
resource "grafana_folder" "stock_analytics" {
  title = "Stock Analytics Engine"
}

# Notification policy for alerts
resource "grafana_notification_policy" "stock_analytics_policy" {
  group_by      = ["alertname", "grafana_folder"]
  contact_point = grafana_contact_point.email[0].name

  group_wait      = "10s"
  group_interval  = "5m"
  repeat_interval = "12h"

  policy {
    matcher {
      label = "severity"
      match = "="
      value = "critical"
    }
    contact_point   = grafana_contact_point.email[0].name
    group_interval  = "1m"
    repeat_interval = "5m"
  }
}

# Email contact point for alerts
resource "grafana_contact_point" "email" {
  count = var.observability_alert_email != "" ? 1 : 0
  name  = "stock-analytics-email"

  email {
    addresses = [var.observability_alert_email]
    subject   = "Stock Analytics Alert: {{ .GroupLabels.alertname }}"
    message   = "Alert: {{ .GroupLabels.alertname }}\nSeverity: {{ .CommonLabels.severity }}\nDescription: {{ .CommonAnnotations.description }}"
  }
}