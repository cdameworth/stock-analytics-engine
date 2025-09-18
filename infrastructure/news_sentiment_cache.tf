# DynamoDB table for news sentiment caching
resource "aws_dynamodb_table" "news_sentiment_cache" {
  name           = "news-sentiment-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "cache_key"

  attribute {
    name = "cache_key"
    type = "S"
  }

  attribute {
    name = "symbol"
    type = "S"
  }

  # TTL configuration for automatic cache cleanup
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  # Global Secondary Index for querying by symbol
  global_secondary_index {
    name            = "symbol-index"
    hash_key        = "symbol"
    projection_type = "ALL"
  }

  # Encryption at rest
  server_side_encryption {
    enabled = true
  }

  # Point-in-time recovery
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name        = "news-sentiment-cache"
    Environment = var.environment
    Project     = "stock-analytics-engine"
    Purpose     = "Cache for news sentiment analysis data"
  }
}

# IAM policy for Lambda functions to access sentiment cache table
resource "aws_iam_policy" "sentiment_cache_access" {
  name        = "sentiment-cache-access-policy"
  description = "Policy for Lambda functions to access news sentiment cache"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.news_sentiment_cache.arn,
          "${aws_dynamodb_table.news_sentiment_cache.arn}/index/*"
        ]
      }
    ]
  })
}

# Attach policy to Lambda execution role
resource "aws_iam_role_policy_attachment" "sentiment_cache_policy_attachment" {
  count      = length(var.lambda_function_names)
  role       = "lambda-execution-role"  # Existing Lambda execution role
  policy_arn = aws_iam_policy.sentiment_cache_access.arn

  depends_on = [aws_iam_policy.sentiment_cache_access]
}

# SSM parameters for news API keys
resource "aws_ssm_parameter" "newsapi_key" {
  name        = "/stock-analytics/newsapi-key"
  description = "NewsAPI.org API key for news sentiment analysis"
  type        = "SecureString"
  value       = var.newsapi_key != "" ? var.newsapi_key : "placeholder_key"

  tags = {
    Environment = var.environment
    Project     = "stock-analytics-engine"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "finnhub_key" {
  name        = "/stock-analytics/finnhub-key"
  description = "Finnhub API key for news sentiment analysis"
  type        = "SecureString"
  value       = var.finnhub_key != "" ? var.finnhub_key : "placeholder_key"

  tags = {
    Environment = var.environment
    Project     = "stock-analytics-engine"
  }

  lifecycle {
    ignore_changes = [value]
  }
}

# CloudWatch Log Group for sentiment analysis
resource "aws_cloudwatch_log_group" "sentiment_analysis_logs" {
  name              = "/aws/lambda/news-sentiment-analyzer"
  retention_in_days = 14

  tags = {
    Environment = var.environment
    Project     = "stock-analytics-engine"
  }
}

# Output the table information
output "sentiment_cache_table_name" {
  description = "Name of the DynamoDB table for sentiment caching"
  value       = aws_dynamodb_table.news_sentiment_cache.name
}

output "sentiment_cache_table_arn" {
  description = "ARN of the DynamoDB table for sentiment caching"
  value       = aws_dynamodb_table.news_sentiment_cache.arn
}