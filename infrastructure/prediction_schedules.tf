# Prediction Model Schedules
# Automate price and time predictions with EventBridge schedules

# EventBridge rule for morning price predictions (9:30 AM EST = 14:30 UTC)
resource "aws_cloudwatch_event_rule" "morning_price_predictions" {
  name                = "morning-price-predictions"
  description         = "Trigger price predictions at market open"
  schedule_expression = "cron(30 14 ? * MON-FRI *)"
  state               = "ENABLED"

  tags = {
    Name      = "Morning Price Predictions"
    Component = "PredictionSchedules"
  }
}

# EventBridge target for morning price predictions
resource "aws_cloudwatch_event_target" "morning_price_predictions_target" {
  rule      = aws_cloudwatch_event_rule.morning_price_predictions.name
  target_id = "MorningPricePredictionsTarget"
  arn       = aws_lambda_function.price_prediction_model.arn

  input = jsonencode({
    "trigger_type"   = "scheduled"
    "market_session" = "morning"
    "symbols"        = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "CRM"]
  })
}

# EventBridge rule for afternoon predictions (2:00 PM EST = 19:00 UTC)
resource "aws_cloudwatch_event_rule" "afternoon_predictions" {
  name                = "afternoon-predictions"
  description         = "Trigger predictions at midday"
  schedule_expression = "cron(0 19 ? * MON-FRI *)"
  state               = "ENABLED"

  tags = {
    Name      = "Afternoon Predictions"
    Component = "PredictionSchedules"
  }
}

# EventBridge target for afternoon price predictions
resource "aws_cloudwatch_event_target" "afternoon_price_predictions_target" {
  rule      = aws_cloudwatch_event_rule.afternoon_predictions.name
  target_id = "AfternoonPricePredictionsTarget"
  arn       = aws_lambda_function.price_prediction_model.arn

  input = jsonencode({
    "trigger_type"   = "scheduled"
    "market_session" = "afternoon"
    "symbols"        = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "CRM"]
  })
}

# EventBridge target for afternoon time predictions
resource "aws_cloudwatch_event_target" "afternoon_time_predictions_target" {
  rule      = aws_cloudwatch_event_rule.afternoon_predictions.name
  target_id = "AfternoonTimePredictionsTarget"
  arn       = aws_lambda_function.time_prediction_model.arn

  input = jsonencode({
    "trigger_type"   = "scheduled"
    "market_session" = "afternoon"
    "symbols"        = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  })
}

# Permission for EventBridge to invoke price prediction Lambda
resource "aws_lambda_permission" "allow_eventbridge_price_predictions" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.price_prediction_model.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.morning_price_predictions.arn
}

# Permission for EventBridge to invoke price prediction Lambda (afternoon)
resource "aws_lambda_permission" "allow_eventbridge_price_predictions_afternoon" {
  statement_id  = "AllowExecutionFromEventBridgeAfternoon"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.price_prediction_model.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.afternoon_predictions.arn
}

# Permission for EventBridge to invoke time prediction Lambda
resource "aws_lambda_permission" "allow_eventbridge_time_predictions" {
  statement_id  = "AllowExecutionFromEventBridgeTime"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.time_prediction_model.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.afternoon_predictions.arn
}

# Weekly accuracy tracking schedule (Sundays at 8 AM UTC)
resource "aws_cloudwatch_event_rule" "weekly_accuracy_tracking" {
  name                = "weekly-accuracy-tracking"
  description         = "Weekly accuracy analysis and tracking"
  schedule_expression = "cron(0 8 ? * SUN *)"
  state               = "ENABLED"

  tags = {
    Name      = "Weekly Accuracy Tracking"
    Component = "PredictionSchedules"
  }
}

# EventBridge target for weekly accuracy tracking
resource "aws_cloudwatch_event_target" "weekly_accuracy_tracking_target" {
  rule      = aws_cloudwatch_event_rule.weekly_accuracy_tracking.name
  target_id = "WeeklyAccuracyTrackingTarget"
  arn       = aws_lambda_function.dual_accuracy_tracker.arn

  input = jsonencode({
    "trigger_type"    = "weekly_scheduled"
    "analysis_period" = "weekly"
  })
}

# Permission for EventBridge to invoke accuracy tracker Lambda
resource "aws_lambda_permission" "allow_eventbridge_accuracy_tracker" {
  statement_id  = "AllowExecutionFromEventBridgeAccuracy"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dual_accuracy_tracker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_accuracy_tracking.arn
}

# Output the schedule configurations
output "prediction_schedules" {
  value = {
    morning_price_predictions = {
      schedule = aws_cloudwatch_event_rule.morning_price_predictions.schedule_expression
      state    = aws_cloudwatch_event_rule.morning_price_predictions.state
    }
    afternoon_predictions = {
      schedule = aws_cloudwatch_event_rule.afternoon_predictions.schedule_expression
      state    = aws_cloudwatch_event_rule.afternoon_predictions.state
    }
    weekly_accuracy_tracking = {
      schedule = aws_cloudwatch_event_rule.weekly_accuracy_tracking.schedule_expression
      state    = aws_cloudwatch_event_rule.weekly_accuracy_tracking.state
    }
  }
  description = "Prediction model schedule configurations"
}