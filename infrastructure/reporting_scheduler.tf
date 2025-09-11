# Scheduled Reporting System for Stock Analytics
# Sends morning validation and evening summary reports

# SNS Topic for email notifications
resource "aws_sns_topic" "stock_analytics_reports" {
  name = "stock-analytics-daily-reports"

  tags = {
    Name        = "Stock Analytics Daily Reports"
    Environment = "production"
    Purpose     = "DailyReports"
  }
}

# SNS Topic subscription (you'll need to confirm via email)
resource "aws_sns_topic_subscription" "email_reports" {
  topic_arn = aws_sns_topic.stock_analytics_reports.arn
  protocol  = "email"
  endpoint  = var.notification_email # You'll need to set this variable
}

# Package the report sender function
data "archive_file" "report_sender" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/report_sender.py"
  output_path = "${path.module}/report_sender.zip"
}

# Lambda function for sending formatted reports
resource "aws_lambda_function" "report_sender" {
  filename         = data.archive_file.report_sender.output_path
  function_name    = "stock-analytics-report-sender"
  role             = aws_iam_role.report_sender_role.arn
  handler          = "report_sender.lambda_handler"
  source_code_hash = data.archive_file.report_sender.output_base64sha256
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      SNS_TOPIC_ARN     = aws_sns_topic.stock_analytics_reports.arn
      REPORTING_API_URL = aws_lambda_function_url.reporting_api_url.function_url
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.report_sender_policy
  ]
}

# Lambda URL for the existing reporting API (if not already created)
resource "aws_lambda_function_url" "reporting_api_url" {
  function_name      = aws_lambda_function.dual_prediction_reporting.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_origins     = ["*"]
    allow_methods     = ["GET"]
    allow_headers     = ["*"]
    max_age           = 86400
  }
}

# IAM Role for Report Sender Lambda
resource "aws_iam_role" "report_sender_role" {
  name = "stock-analytics-report-sender-role"

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
}

# IAM Policy for Report Sender
resource "aws_iam_policy" "report_sender_policy" {
  name = "stock-analytics-report-sender-policy"

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
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.stock_analytics_reports.arn
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
          "lambda:InvokeFunctionUrl"
        ]
        Resource = aws_lambda_function.dual_prediction_reporting.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "report_sender_policy" {
  role       = aws_iam_role.report_sender_role.name
  policy_arn = aws_iam_policy.report_sender_policy.arn
}

# EventBridge Rule for Morning Report (6:30 AM EST weekdays)
resource "aws_cloudwatch_event_rule" "morning_report" {
  name                = "stock-analytics-morning-report"
  description         = "Trigger morning validation report at 6:30 AM EST"
  schedule_expression = "cron(30 11 ? * MON-FRI *)" # 11:30 UTC = 6:30 AM EST

  tags = {
    Name = "Morning Report Trigger"
    Type = "Scheduled"
  }
}

# EventBridge Rule for Evening Report (6:00 PM EST weekdays)
resource "aws_cloudwatch_event_rule" "evening_report" {
  name                = "stock-analytics-evening-report"
  description         = "Trigger evening summary report at 6:00 PM EST"
  schedule_expression = "cron(0 23 ? * MON-FRI *)" # 23:00 UTC = 6:00 PM EST

  tags = {
    Name = "Evening Report Trigger"
    Type = "Scheduled"
  }
}

# EventBridge Target for Morning Report
resource "aws_cloudwatch_event_target" "morning_report_target" {
  rule      = aws_cloudwatch_event_rule.morning_report.name
  target_id = "MorningReportLambdaTarget"
  arn       = aws_lambda_function.report_sender.arn

  input = jsonencode({
    report_type = "morning"
    timeframe   = "24h"
  })
}

# EventBridge Target for Evening Report
resource "aws_cloudwatch_event_target" "evening_report_target" {
  rule      = aws_cloudwatch_event_rule.evening_report.name
  target_id = "EveningReportLambdaTarget"
  arn       = aws_lambda_function.report_sender.arn

  input = jsonencode({
    report_type = "evening"
    timeframe   = "market_hours"
  })
}

# Lambda Permissions for EventBridge
resource "aws_lambda_permission" "morning_report_permission" {
  statement_id  = "AllowExecutionFromEventBridgeMorning"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.report_sender.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.morning_report.arn
}

resource "aws_lambda_permission" "evening_report_permission" {
  statement_id  = "AllowExecutionFromEventBridgeEvening"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.report_sender.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.evening_report.arn
}

# CloudWatch Log Group for Report Sender
resource "aws_cloudwatch_log_group" "report_sender_logs" {
  name              = "/aws/lambda/stock-analytics-report-sender"
  retention_in_days = 7
}

# Output the SNS topic ARN and Lambda URL
output "sns_topic_arn" {
  value       = aws_sns_topic.stock_analytics_reports.arn
  description = "ARN of the SNS topic for email reports"
}

output "reporting_api_url" {
  value       = aws_lambda_function_url.reporting_api_url.function_url
  description = "URL of the reporting API"
}