# Dual Prediction System Deployment
# Deploy price prediction, time prediction, accuracy tracking, and tuning systems

# Local values for dual prediction system
locals {
  # OTEL layer ARN for dual prediction functions
  otel_layer_arn = var.enable_signoz_integration ? local.custom_otel_layer_arn : ""

  # Base OTEL configuration for dual prediction functions
  otel_base_config = var.enable_signoz_integration ? {
    OTEL_SERVICE_NAME = "dual-prediction"
    OTEL_EXPORTER_OTLP_ENDPOINT  = "https://${var.signoz_otlp_endpoint}"
    OTEL_EXPORTER_OTLP_HEADERS   = "signoz-ingestion-key=${var.signoz_ingestion_key}"
    ENABLE_BUSINESS_TRACING      = "true"
  } : {}
}

# Create deployment packages for dual prediction functions
resource "null_resource" "package_dual_prediction_functions" {
  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<EOF
      cd ../lambda_functions
      
      # Package each dual prediction function
      zip -r ../infrastructure/price_prediction_model.zip price_prediction_model.py
      zip -r ../infrastructure/time_to_hit_predictor.zip time_to_hit_predictor_slim.py
      zip -r ../infrastructure/dual_accuracy_tracker.zip dual_accuracy_tracker_simple.py
      zip -r ../infrastructure/price_model_tuning.zip price_model_tuning.py
      zip -r ../infrastructure/time_model_tuning.zip time_model_tuning.py
      zip -r ../infrastructure/dual_prediction_reporting_api.zip dual_prediction_reporting_api.py
    EOF
  }
}

# DynamoDB Tables for Dual Prediction System

# Price predictions table
resource "aws_dynamodb_table" "price_predictions" {
  name         = "price-predictions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prediction_id"

  attribute {
    name = "prediction_id"
    type = "S"
  }

  attribute {
    name = "symbol"
    type = "S"
  }

  attribute {
    name = "prediction_timestamp"
    type = "S"
  }

  global_secondary_index {
    name            = "SymbolTimestampIndex"
    hash_key        = "symbol"
    range_key       = "prediction_timestamp"
    projection_type = "ALL"
  }

  tags = {
    Name      = "Price Predictions"
    Component = "DualPredictionSystem"
  }
}

# Time predictions table  
resource "aws_dynamodb_table" "time_predictions" {
  name         = "time-to-hit-predictions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prediction_id"

  attribute {
    name = "prediction_id"
    type = "S"
  }

  attribute {
    name = "symbol"
    type = "S"
  }

  attribute {
    name = "prediction_date"
    type = "S"
  }

  global_secondary_index {
    name            = "SymbolDateIndex"
    hash_key        = "symbol"
    range_key       = "prediction_date"
    projection_type = "ALL"
  }

  tags = {
    Name      = "Time Predictions"
    Component = "DualPredictionSystem"
  }
}

# Accuracy metrics table
resource "aws_dynamodb_table" "accuracy_metrics" {
  name         = "prediction-accuracy-metrics"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "metric_id"

  attribute {
    name = "metric_id"
    type = "S"
  }

  attribute {
    name = "model_type"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  global_secondary_index {
    name            = "ModelTypeTimestampIndex"
    hash_key        = "model_type"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  tags = {
    Name      = "Prediction Accuracy Metrics"
    Component = "DualPredictionSystem"
  }
}

# Tuning history table
resource "aws_dynamodb_table" "tuning_history" {
  name         = "model-tuning-history"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"

  attribute {
    name = "session_id"
    type = "S"
  }

  attribute {
    name = "model_type"
    type = "S"
  }

  attribute {
    name = "session_timestamp"
    type = "S"
  }

  global_secondary_index {
    name            = "ModelTypeTimestampIndex"
    hash_key        = "model_type"
    range_key       = "session_timestamp"
    projection_type = "ALL"
  }

  tags = {
    Name      = "Model Tuning History"
    Component = "DualPredictionSystem"
  }
}

# Lambda Functions

# Price Prediction Model
resource "aws_lambda_function" "price_prediction_model" {
  filename      = "price_prediction_model.zip"
  function_name = "price-prediction-model"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "price_prediction_model.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  layers = var.enable_signoz_integration ? [local.otel_layer_arn] : []

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  environment {
    variables = merge(
      {
        PRICE_PREDICTIONS_TABLE = aws_dynamodb_table.price_predictions.name
        S3_DATA_BUCKET          = aws_s3_bucket.stock_data_lake.id
        RECOMMENDATIONS_TABLE   = aws_dynamodb_table.stock_recommendations.name
      },
      local.otel_base_config,
      var.enable_signoz_integration ? {
        OTEL_SERVICE_NAME = "price-prediction-model"
      } : {}
    )
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Price Prediction Model"
    Component = "DualPredictionSystem"
  }
}

# Update existing time prediction function (already deployed, just update environment)
resource "aws_lambda_function" "time_prediction_model" {
  filename      = "time_to_hit_predictor.zip"
  function_name = "time-to-hit-predictor"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "time_to_hit_predictor_slim.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512

  layers = var.enable_signoz_integration ? [local.otel_layer_arn] : []

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = merge(
      {
        TIME_PREDICTIONS_TABLE = aws_dynamodb_table.time_predictions.name
        S3_DATA_BUCKET         = aws_s3_bucket.stock_data_lake.id
      },
      local.otel_base_config,
      var.enable_signoz_integration ? {
        OTEL_SERVICE_NAME = "time-prediction-model"
      } : {}
    )
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Time Prediction Model"
    Component = "DualPredictionSystem"
  }
}

# Dual Accuracy Tracker
resource "aws_lambda_function" "dual_accuracy_tracker" {
  filename      = "dual_accuracy_tracker.zip"
  function_name = "dual-accuracy-tracker"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "dual_accuracy_tracker_simple.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024

  layers = concat([
    "arn:aws:lambda:us-east-1:791060928878:layer:basic-python-deps:1"
  ], local.common_otel_layers)

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = merge(
      {
        PRICE_PREDICTIONS_TABLE = aws_dynamodb_table.price_predictions.name
        TIME_PREDICTIONS_TABLE  = aws_dynamodb_table.time_predictions.name
        ACCURACY_METRICS_TABLE  = aws_dynamodb_table.accuracy_metrics.name
      },
      var.enable_signoz_integration ? local.get_lambda_environment.dual_accuracy_tracker : {}
    )
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Dual Accuracy Tracker"
    Component = "DualPredictionSystem"
  }
}

# Price Model Tuning
resource "aws_lambda_function" "price_model_tuning" {
  filename      = "price_model_tuning.zip"
  function_name = "price-model-tuning"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "price_model_tuning.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      PRICE_PREDICTIONS_TABLE   = aws_dynamodb_table.price_predictions.name
      TUNING_HISTORY_TABLE      = aws_dynamodb_table.tuning_history.name
      PRICE_PREDICTION_FUNCTION = aws_lambda_function.price_prediction_model.function_name
    }
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Price Model Tuning"
    Component = "DualPredictionSystem"
  }
}

# Time Model Tuning
resource "aws_lambda_function" "time_model_tuning" {
  filename      = "time_model_tuning.zip"
  function_name = "time-model-tuning"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "time_model_tuning.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      TIME_PREDICTIONS_TABLE   = aws_dynamodb_table.time_predictions.name
      TUNING_HISTORY_TABLE     = aws_dynamodb_table.tuning_history.name
      TIME_PREDICTION_FUNCTION = aws_lambda_function.time_prediction_model.function_name
    }
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Time Model Tuning"
    Component = "DualPredictionSystem"
  }
}

# Dual Prediction Reporting API
resource "aws_lambda_function" "dual_prediction_reporting" {
  filename      = "dual_prediction_reporting_api.zip"
  function_name = "dual-prediction-reporting-api"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "dual_prediction_reporting_api.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512

  layers = local.common_otel_layers

  # Enable X-Ray tracing for distributed tracing
  tracing_config {
    mode = "Active"
  }

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = merge(
      {
        PRICE_PREDICTIONS_TABLE = aws_dynamodb_table.price_predictions.name
        TIME_PREDICTIONS_TABLE  = aws_dynamodb_table.time_predictions.name
        ACCURACY_METRICS_TABLE  = aws_dynamodb_table.accuracy_metrics.name
        TUNING_HISTORY_TABLE    = aws_dynamodb_table.tuning_history.name
      },
      var.enable_signoz_integration ? local.get_lambda_environment.dual_prediction_reporting_api : {}
    )
  }

  depends_on = [null_resource.package_dual_prediction_functions]

  tags = {
    Name      = "Dual Prediction Reporting API"
    Component = "DualPredictionSystem"
  }
}

# Enhanced Lambda execution policy for dual prediction system
resource "aws_iam_role_policy" "dual_prediction_policy" {
  name = "dual-prediction-system-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem"
        ]
        Resource = [
          aws_dynamodb_table.price_predictions.arn,
          aws_dynamodb_table.time_predictions.arn,
          aws_dynamodb_table.accuracy_metrics.arn,
          aws_dynamodb_table.tuning_history.arn,
          "${aws_dynamodb_table.price_predictions.arn}/index/*",
          "${aws_dynamodb_table.time_predictions.arn}/index/*",
          "${aws_dynamodb_table.accuracy_metrics.arn}/index/*",
          "${aws_dynamodb_table.tuning_history.arn}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          aws_lambda_function.price_prediction_model.arn,
          aws_lambda_function.time_prediction_model.arn,
          aws_lambda_function.dual_accuracy_tracker.arn,
          aws_lambda_function.price_model_tuning.arn,
          aws_lambda_function.time_model_tuning.arn,
          "${aws_lambda_function.price_prediction_model.arn}:*",
          "${aws_lambda_function.time_prediction_model.arn}:*",
          "${aws_lambda_function.dual_accuracy_tracker.arn}:*",
          "${aws_lambda_function.price_model_tuning.arn}:*",
          "${aws_lambda_function.time_model_tuning.arn}:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# EventBridge schedules for dual prediction system

# Daily accuracy validation
resource "aws_cloudwatch_event_rule" "daily_accuracy_validation" {
  name                = "daily-dual-accuracy-validation"
  description         = "Daily validation of price and time prediction accuracy"
  schedule_expression = "cron(0 6 * * ? *)" # 6 AM UTC daily

  tags = {
    Name      = "Daily Accuracy Validation"
    Component = "DualPredictionSystem"
  }
}

resource "aws_cloudwatch_event_target" "daily_accuracy_validation_target" {
  rule      = aws_cloudwatch_event_rule.daily_accuracy_validation.name
  target_id = "DailyAccuracyValidationTarget"
  arn       = aws_lambda_function.dual_accuracy_tracker.arn

  input = jsonencode({
    action        = "validate_all"
    lookback_days = 7
    trigger       = "automated_daily"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_accuracy_validation" {
  statement_id  = "AllowExecutionFromEventBridgeAccuracy"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dual_accuracy_tracker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_accuracy_validation.arn
}

# Weekly price model tuning
resource "aws_cloudwatch_event_rule" "weekly_price_tuning" {
  name                = "weekly-price-model-tuning"
  description         = "Weekly price model optimization and tuning"
  schedule_expression = "cron(0 3 ? * SUN *)" # 3 AM every Sunday

  tags = {
    Name      = "Weekly Price Model Tuning"
    Component = "DualPredictionSystem"
  }
}

resource "aws_cloudwatch_event_target" "weekly_price_tuning_target" {
  rule      = aws_cloudwatch_event_rule.weekly_price_tuning.name
  target_id = "WeeklyPriceTuningTarget"
  arn       = aws_lambda_function.price_model_tuning.arn

  input = jsonencode({
    action        = "full_tuning_cycle"
    lookback_days = 60
    trigger       = "automated_weekly"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_price_tuning" {
  statement_id  = "AllowExecutionFromEventBridgePrice"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.price_model_tuning.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_price_tuning.arn
}

# Weekly time model tuning
resource "aws_cloudwatch_event_rule" "weekly_time_tuning" {
  name                = "weekly-time-model-tuning"
  description         = "Weekly time model optimization and tuning"
  schedule_expression = "cron(0 4 ? * SUN *)" # 4 AM every Sunday

  tags = {
    Name      = "Weekly Time Model Tuning"
    Component = "DualPredictionSystem"
  }
}

resource "aws_cloudwatch_event_target" "weekly_time_tuning_target" {
  rule      = aws_cloudwatch_event_rule.weekly_time_tuning.name
  target_id = "WeeklyTimeTuningTarget"
  arn       = aws_lambda_function.time_model_tuning.arn

  input = jsonencode({
    action        = "full_tuning_cycle"
    lookback_days = 60
    trigger       = "automated_weekly"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_time_tuning" {
  statement_id  = "AllowExecutionFromEventBridgeTime"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.time_model_tuning.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_time_tuning.arn
}

# Note: API Gateway integration deferred until base infrastructure confirmed

# CloudWatch dashboard for dual prediction system
resource "aws_cloudwatch_dashboard" "dual_prediction_dashboard" {
  dashboard_name = "DualPredictionSystem"

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
            ["StockAnalytics/PricePrediction", "PredictionGenerated"],
            ["StockAnalytics/DualAccuracy", "ModelAccuracy", "ModelType", "price_accuracy"],
            ["StockAnalytics/DualAccuracy", "ModelAccuracy", "ModelType", "time_accuracy"],
            ["StockAnalytics/DualAccuracy", "PredictionCount", "ModelType", "price_accuracy"],
            ["StockAnalytics/DualAccuracy", "PredictionCount", "ModelType", "time_accuracy"]
          ]
          period = 300
          stat   = "Average"
          region = "us-east-1"
          title  = "Dual Prediction System Performance"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["StockAnalytics/PriceTuning", "TuningSession"],
            ["StockAnalytics/TimeTuning", "TuningSession"]
          ]
          period = 86400
          stat   = "Sum"
          region = "us-east-1"
          title  = "Model Tuning Activity"
        }
      }
    ]
  })
}

# Note: API endpoints deferred until API Gateway infrastructure confirmed

output "dual_prediction_functions" {
  description = "Lambda function names for dual prediction system"
  value = {
    price_prediction = aws_lambda_function.price_prediction_model.function_name
    time_prediction  = aws_lambda_function.time_prediction_model.function_name
    accuracy_tracker = aws_lambda_function.dual_accuracy_tracker.function_name
    price_tuning     = aws_lambda_function.price_model_tuning.function_name
    time_tuning      = aws_lambda_function.time_model_tuning.function_name
    reporting_api    = aws_lambda_function.dual_prediction_reporting.function_name
  }
}