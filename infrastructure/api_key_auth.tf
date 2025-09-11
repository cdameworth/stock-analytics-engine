# API Key for authentication
resource "aws_api_gateway_api_key" "stock_analytics_api_key" {
  name        = "stock-analytics-api-key"
  description = "API key for stock analytics services"
  enabled     = true

  tags = merge(
    {
      Name    = "stock-analytics-api-key"
      Purpose = "API authentication for stock analytics services"
    },
    var.additional_tags
  )
}

# Usage Plan - defines rate limits and throttling
resource "aws_api_gateway_usage_plan" "stock_analytics_usage_plan" {
  name        = "stock-analytics-usage-plan"
  description = "Usage plan for stock analytics API"

  # API stages to include
  api_stages {
    api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
    stage  = aws_api_gateway_stage.stock_recommendations_api_stage.stage_name
  }

  # Rate limiting (requests per second)
  throttle_settings {
    rate_limit  = 100 # 100 requests per second
    burst_limit = 200 # Burst up to 200 requests
  }

  # Quota (requests per day/month)
  quota_settings {
    limit  = 10000 # 10,000 requests per month
    period = "MONTH"
  }

  tags = merge(
    {
      Name = "stock-analytics-usage-plan"
    },
    var.additional_tags
  )
}

# Associate API key with usage plan
resource "aws_api_gateway_usage_plan_key" "stock_analytics_usage_plan_key" {
  key_id        = aws_api_gateway_api_key.stock_analytics_api_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.stock_analytics_usage_plan.id
}

# API key authentication is managed via AWS CLI updates to existing methods
# The methods have been updated to require API keys via aws apigateway update-method commands

# Outputs
output "api_key_id" {
  description = "API Key ID for stock analytics API"
  value       = aws_api_gateway_api_key.stock_analytics_api_key.id
}

output "api_key_value" {
  description = "API Key value (keep this secret!)"
  value       = aws_api_gateway_api_key.stock_analytics_api_key.value
  sensitive   = true
}

output "api_usage_instructions" {
  description = "How to use the API with authentication"
  value = {
    header_name  = "x-api-key"
    example_curl = "curl -H 'x-api-key: YOUR_API_KEY' https://2cqomr4nb2.execute-api.us-east-1.amazonaws.com/prod/recommendations"
    rate_limits  = "100 req/sec, 200 burst, 10,000/month"
  }
}