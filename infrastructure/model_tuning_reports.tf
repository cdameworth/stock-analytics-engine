# Model Tuning Reporter Infrastructure
# Deploys Lambda function for sending detailed model tuning reports

# Lambda function for model tuning reports
data "archive_file" "model_tuning_reporter" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/model_tuning_reporter.py"
  output_path = "${path.module}/model_tuning_reporter.zip"
}

resource "aws_lambda_function" "model_tuning_reporter" {
  filename         = data.archive_file.model_tuning_reporter.output_path
  source_code_hash = data.archive_file.model_tuning_reporter.output_base64sha256
  function_name    = "model-tuning-reporter"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "model_tuning_reporter.lambda_handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 512

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      SNS_TOPIC_ARN           = aws_sns_topic.cost_alerts.arn
      TUNING_HISTORY_TABLE    = "model-tuning-history"
      ACCURACY_METRICS_TABLE  = "prediction-accuracy-metrics"
      PRICE_PREDICTIONS_TABLE = "price-predictions"
      TIME_PREDICTIONS_TABLE  = "time-to-hit-predictions"
    }
  }

  tags = merge(
    {
      Name = "model-tuning-reporter"
      Type = "reporting"
    },
    var.additional_tags
  )
}

# Update existing tuning functions with reporter environment variable
resource "aws_lambda_function" "price_model_tuning_updated" {
  filename         = "${path.module}/price_model_tuning.zip"
  source_code_hash = data.archive_file.price_model_tuning.output_base64sha256
  function_name    = "price-model-tuning"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "price_model_tuning.lambda_handler"
  runtime          = "python3.11"
  timeout          = 900  # 15 minutes for tuning
  memory_size      = 1024

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      PRICE_PREDICTIONS_TABLE = "price-predictions"
      TUNING_HISTORY_TABLE    = "model-tuning-history"
      PRICE_PREDICTION_FUNCTION = "price-prediction-model"
      TUNING_REPORTER_FUNCTION = aws_lambda_function.model_tuning_reporter.function_name
    }
  }

  tags = merge(
    {
      Name = "price-model-tuning"
      Type = "model-tuning"
    },
    var.additional_tags
  )
}

resource "aws_lambda_function" "time_model_tuning_updated" {
  filename         = "${path.module}/time_model_tuning.zip"
  source_code_hash = data.archive_file.time_model_tuning.output_base64sha256
  function_name    = "time-model-tuning"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "time_model_tuning.lambda_handler"
  runtime          = "python3.11"
  timeout          = 900  # 15 minutes for tuning
  memory_size      = 1024

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      TIME_PREDICTIONS_TABLE = "time-to-hit-predictions"
      TUNING_HISTORY_TABLE   = "model-tuning-history"
      TIME_PREDICTION_FUNCTION = "time-to-hit-predictor"
      TUNING_REPORTER_FUNCTION = aws_lambda_function.model_tuning_reporter.function_name
    }
  }

  tags = merge(
    {
      Name = "time-model-tuning"
      Type = "model-tuning"
    },
    var.additional_tags
  )
}

# Archive files for tuning functions
data "archive_file" "price_model_tuning" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/price_model_tuning.py"
  output_path = "${path.module}/price_model_tuning.zip"
}

data "archive_file" "time_model_tuning" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/time_model_tuning.py"
  output_path = "${path.module}/time_model_tuning.zip"
}

# DynamoDB table for tuning history
resource "aws_dynamodb_table" "model_tuning_history" {
  name           = "model-tuning-history"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "model_type"
  range_key      = "timestamp"

  attribute {
    name = "model_type"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  # TTL for automatic cleanup after 90 days
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = merge(
    {
      Name = "model-tuning-history"
      Type = "reporting"
    },
    var.additional_tags
  )
}

# Lambda permissions for invoking the tuning reporter
resource "aws_lambda_permission" "price_tuning_invoke_reporter" {
  statement_id  = "AllowPriceTuningInvokeReporter"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.model_tuning_reporter.function_name
  principal     = "lambda.amazonaws.com"
  source_arn    = aws_lambda_function.price_model_tuning_updated.arn
}

resource "aws_lambda_permission" "time_tuning_invoke_reporter" {
  statement_id  = "AllowTimeTuningInvokeReporter"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.model_tuning_reporter.function_name
  principal     = "lambda.amazonaws.com"
  source_arn    = aws_lambda_function.time_model_tuning_updated.arn
}

# IAM policy for tuning functions to invoke reporter
resource "aws_iam_role_policy" "lambda_tuning_invoke_reporter" {
  name = "lambda-tuning-invoke-reporter"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          aws_lambda_function.model_tuning_reporter.arn,
          "${aws_lambda_function.model_tuning_reporter.arn}:*"
        ]
      }
    ]
  })
}

# IAM policy for reporter to access DynamoDB and SNS
resource "aws_iam_role_policy" "lambda_tuning_reporter_permissions" {
  name = "lambda-tuning-reporter-permissions"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.model_tuning_history.arn,
          "${aws_dynamodb_table.model_tuning_history.arn}/index/*",
          "arn:aws:dynamodb:*:*:table/prediction-accuracy-metrics",
          "arn:aws:dynamodb:*:*:table/price-predictions",
          "arn:aws:dynamodb:*:*:table/time-to-hit-predictions"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = [
          aws_sns_topic.cost_alerts.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })
}

# Update EventBridge rules to trigger tuning functions that now include reporting
resource "aws_cloudwatch_event_rule" "weekly_price_model_tuning_with_reports" {
  name                = "weekly-price-model-tuning"
  description         = "Trigger price model tuning with reports every Sunday at 3 AM UTC"
  schedule_expression = "cron(0 3 ? * SUN *)"
  state               = "ENABLED"

  tags = merge(
    {
      Name = "weekly-price-model-tuning"
      Type = "scheduled-tuning"
    },
    var.additional_tags
  )
}

resource "aws_cloudwatch_event_rule" "weekly_time_model_tuning_with_reports" {
  name                = "weekly-time-model-tuning"
  description         = "Trigger time model tuning with reports every Sunday at 4 AM UTC"
  schedule_expression = "cron(0 4 ? * SUN *)"
  state               = "ENABLED"

  tags = merge(
    {
      Name = "weekly-time-model-tuning"
      Type = "scheduled-tuning"
    },
    var.additional_tags
  )
}

# EventBridge targets for tuning functions
resource "aws_cloudwatch_event_target" "price_model_tuning_target" {
  rule      = aws_cloudwatch_event_rule.weekly_price_model_tuning_with_reports.name
  target_id = "PriceModelTuningTarget"
  arn       = aws_lambda_function.price_model_tuning_updated.arn

  input = jsonencode({
    action = "full_tuning_cycle"
    lookback_days = 30
    report_results = true
  })
}

resource "aws_cloudwatch_event_target" "time_model_tuning_target" {
  rule      = aws_cloudwatch_event_rule.weekly_time_model_tuning_with_reports.name
  target_id = "TimeModelTuningTarget"
  arn       = aws_lambda_function.time_model_tuning_updated.arn

  input = jsonencode({
    action = "full_tuning_cycle"
    lookback_days = 30
    report_results = true
  })
}

# Lambda permissions for EventBridge to invoke tuning functions
resource "aws_lambda_permission" "allow_eventbridge_price_tuning_reports" {
  statement_id  = "AllowExecutionFromEventBridgePriceTuningReports"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.price_model_tuning_updated.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_price_model_tuning_with_reports.arn
}

resource "aws_lambda_permission" "allow_eventbridge_time_tuning_reports" {
  statement_id  = "AllowExecutionFromEventBridgeTimeTuningReports"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.time_model_tuning_updated.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_time_model_tuning_with_reports.arn
}

# Outputs
output "model_tuning_reporter_function_name" {
  description = "Name of the model tuning reporter Lambda function"
  value       = aws_lambda_function.model_tuning_reporter.function_name
}

output "tuning_history_table_name" {
  description = "Name of the model tuning history DynamoDB table"
  value       = aws_dynamodb_table.model_tuning_history.name
}

output "tuning_schedule_info" {
  description = "Information about tuning schedules"
  value = {
    price_model_schedule = "Sundays at 3 AM UTC with email reports"
    time_model_schedule  = "Sundays at 4 AM UTC with email reports"
    next_price_tuning    = "Next Sunday at 3 AM UTC"
    next_time_tuning     = "Next Sunday at 4 AM UTC"
    report_email         = "cdameworth@gmail.com"
  }
}