# Stock Analytics Engine - ML-powered stock recommendation service
# This application fetches stock data from Alpha Vantage API and provides ML-based recommendations

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Application = "stock-analytics-engine"
      Environment = var.environment
      Team        = "data-science"
      CostCenter  = "ml-analytics"
      Owner       = "stock-analytics-team@company.com"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API Key"
  type        = string
  default     = "YFT4NTLIWG9Z05LA"
  sensitive   = true
}

# VPC and Networking
resource "aws_vpc" "stock_analytics_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "stock-analytics-vpc"
  }
}

resource "aws_subnet" "private_subnet_1" {
  vpc_id            = aws_vpc.stock_analytics_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"
  
  tags = {
    Name = "stock-analytics-private-1"
  }
}

resource "aws_subnet" "private_subnet_2" {
  vpc_id            = aws_vpc.stock_analytics_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.aws_region}b"
  
  tags = {
    Name = "stock-analytics-private-2"
  }
}

resource "aws_subnet" "public_subnet_1" {
  vpc_id                  = aws_vpc.stock_analytics_vpc.id
  cidr_block              = "10.0.101.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  
  tags = {
    Name = "stock-analytics-public-1"
  }
}

resource "aws_subnet" "public_subnet_2" {
  vpc_id                  = aws_vpc.stock_analytics_vpc.id
  cidr_block              = "10.0.102.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  
  tags = {
    Name = "stock-analytics-public-2"
  }
}

resource "aws_internet_gateway" "stock_analytics_igw" {
  vpc_id = aws_vpc.stock_analytics_vpc.id
  
  tags = {
    Name = "stock-analytics-igw"
  }
}

resource "aws_nat_gateway" "stock_analytics_nat" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_subnet_1.id
  
  tags = {
    Name = "stock-analytics-nat"
  }
}

resource "aws_eip" "nat_eip" {
  domain = "vpc"
  
  tags = {
    Name = "stock-analytics-nat-eip"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.stock_analytics_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.stock_analytics_igw.id
  }
  
  tags = {
    Name = "stock-analytics-public-rt"
  }
}

resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.stock_analytics_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.stock_analytics_nat.id
  }
  
  tags = {
    Name = "stock-analytics-private-rt"
  }
}

resource "aws_route_table_association" "public_rta_1" {
  subnet_id      = aws_subnet.public_subnet_1.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "public_rta_2" {
  subnet_id      = aws_subnet.public_subnet_2.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "private_rta_1" {
  subnet_id      = aws_subnet.private_subnet_1.id
  route_table_id = aws_route_table.private_rt.id
}

resource "aws_route_table_association" "private_rta_2" {
  subnet_id      = aws_subnet.private_subnet_2.id
  route_table_id = aws_route_table.private_rt.id
}

# Security Groups
resource "aws_security_group" "lambda_sg" {
  name        = "stock-analytics-lambda-sg"
  description = "Security group for Lambda functions"
  vpc_id      = aws_vpc.stock_analytics_vpc.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "stock-analytics-lambda-sg"
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "stock-analytics-rds-sg"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.stock_analytics_vpc.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.lambda_sg.id]
  }
  
  tags = {
    Name = "stock-analytics-rds-sg"
  }
}

resource "aws_security_group" "redis_sg" {
  name        = "stock-analytics-redis-sg"
  description = "Security group for Redis cluster"
  vpc_id      = aws_vpc.stock_analytics_vpc.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.lambda_sg.id]
  }
  
  tags = {
    Name = "stock-analytics-redis-sg"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "stock_data_lake" {
  bucket = "stock-analytics-data-lake-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name = "stock-data-lake"
  }
}

resource "aws_s3_bucket" "ml_models" {
  bucket = "stock-analytics-ml-models-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name = "ml-models"
  }
}

resource "aws_s3_bucket" "api_logs" {
  bucket = "stock-analytics-api-logs-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name = "api-logs"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "stock_data_lake_versioning" {
  bucket = aws_s3_bucket.stock_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "stock_data_lake_encryption" {
  bucket = aws_s3_bucket.stock_data_lake.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# DynamoDB Tables
resource "aws_dynamodb_table" "stock_recommendations" {
  name           = "stock-recommendations"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "recommendation_id"
  range_key      = "timestamp"
  
  attribute {
    name = "recommendation_id"
    type = "S"
  }
  
  attribute {
    name = "timestamp"
    type = "S"
  }
  
  attribute {
    name = "symbol"
    type = "S"
  }
  
  global_secondary_index {
    name     = "symbol-timestamp-index"
    hash_key = "symbol"
    range_key = "timestamp"
  }
  
  point_in_time_recovery {
    enabled = true
  }
  
  tags = {
    Name = "stock-recommendations"
  }
}

resource "aws_dynamodb_table" "api_cache" {
  name           = "api-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "cache_key"
  
  attribute {
    name = "cache_key"
    type = "S"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  tags = {
    Name = "api-cache"
  }
}

# RDS Database
resource "aws_db_subnet_group" "stock_analytics_db_subnet_group" {
  name       = "stock-analytics-db-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
  
  tags = {
    Name = "stock-analytics-db-subnet-group"
  }
}

resource "aws_rds_cluster" "stock_analytics_aurora" {
  cluster_identifier     = "stock-analytics-aurora-cluster"
  engine                = "aurora-postgresql"
  engine_version        = "15.4"
  database_name         = "stock_analytics"
  master_username       = "stockadmin"
  manage_master_user_password = true
  
  db_subnet_group_name   = aws_db_subnet_group.stock_analytics_db_subnet_group.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  
  backup_retention_period = 7
  preferred_backup_window = "03:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  tags = {
    Name = "stock-analytics-aurora"
  }
}

resource "aws_rds_cluster_instance" "stock_analytics_aurora_instance" {
  count              = 2
  identifier         = "stock-analytics-aurora-${count.index}"
  cluster_identifier = aws_rds_cluster.stock_analytics_aurora.id
  instance_class     = "db.r6g.large"
  engine             = aws_rds_cluster.stock_analytics_aurora.engine
  engine_version     = aws_rds_cluster.stock_analytics_aurora.engine_version
  
  performance_insights_enabled = true
  monitoring_interval          = 60
  monitoring_role_arn         = aws_iam_role.rds_monitoring_role.arn
  
  tags = {
    Name = "stock-analytics-aurora-${count.index}"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "stock_analytics_redis_subnet_group" {
  name       = "stock-analytics-redis-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
  
  tags = {
    Name = "stock-analytics-redis-subnet-group"
  }
}

resource "aws_elasticache_replication_group" "stock_analytics_redis" {
  replication_group_id       = "stock-analytics-redis"
  description                = "Redis cluster for stock analytics caching"
  
  node_type                  = "cache.r7g.large"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.stock_analytics_redis_subnet_group.name
  security_group_ids = [aws_security_group.redis_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
  
  tags = {
    Name = "stock-analytics-redis"
  }
}

# SageMaker ML Model
resource "aws_sagemaker_model" "stock_prediction_model" {
  name               = "stock-prediction-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn
  
  primary_container {
    image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:0.23-1-cpu-py3"
    model_data_url = "s3://${aws_s3_bucket.ml_models.bucket}/models/stock-prediction-model.tar.gz"
    
    environment = {
      SAGEMAKER_PROGRAM                 = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY        = "s3://${aws_s3_bucket.ml_models.bucket}/code/sourcedir.tar.gz"
      SAGEMAKER_CONTAINER_LOG_LEVEL     = "20"
      SAGEMAKER_REGION                  = var.aws_region
    }
  }
  
  tags = {
    Name = "stock-prediction-model"
  }
}

resource "aws_sagemaker_endpoint_configuration" "stock_prediction_endpoint_config" {
  name = "stock-prediction-endpoint-config"
  
  production_variants {
    variant_name           = "AllTraffic"
    model_name            = aws_sagemaker_model.stock_prediction_model.name
    initial_instance_count = 2
    instance_type         = "ml.m5.large"
    initial_variant_weight = 1
  }
  
  data_capture_config {
    enable_capture                = true
    initial_sampling_percentage   = 100
    destination_s3_uri           = "s3://${aws_s3_bucket.ml_models.bucket}/model-monitor/"
    
    capture_options {
      capture_mode = "Input"
    }
    
    capture_options {
      capture_mode = "Output"
    }
    
    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }
  
  tags = {
    Name = "stock-prediction-endpoint-config"
  }
}

resource "aws_sagemaker_endpoint" "stock_prediction_endpoint" {
  name                 = "stock-prediction-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.stock_prediction_endpoint_config.name
  
  tags = {
    Name = "stock-prediction-endpoint"
  }
}

# Lambda Functions
resource "aws_lambda_function" "stock_data_ingestion" {
  filename         = "stock_data_ingestion.zip"
  function_name    = "stock-data-ingestion"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "index.handler"
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 1024
  
  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }
  
  environment {
    variables = {
      ALPHA_VANTAGE_API_KEY = var.alpha_vantage_api_key
      S3_BUCKET            = aws_s3_bucket.stock_data_lake.bucket
      REDIS_ENDPOINT       = aws_elasticache_replication_group.stock_analytics_redis.primary_endpoint_address
      DYNAMODB_TABLE       = aws_dynamodb_table.api_cache.name
    }
  }
  
  dead_letter_config {
    target_arn = aws_sqs_queue.stock_data_dlq.arn
  }
  
  tracing_config {
    mode = "Active"
  }
  
  tags = {
    Name = "stock-data-ingestion"
  }
}

resource "aws_lambda_function" "ml_model_inference" {
  filename         = "ml_model_inference.zip"
  function_name    = "ml-model-inference"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "index.handler"
  runtime         = "python3.11"
  timeout         = 900
  memory_size     = 2048
  
  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }
  
  environment {
    variables = {
      SAGEMAKER_ENDPOINT     = aws_sagemaker_endpoint.stock_prediction_endpoint.name
      DYNAMODB_TABLE         = aws_dynamodb_table.stock_recommendations.name
      S3_BUCKET             = aws_s3_bucket.stock_data_lake.bucket
      RDS_ENDPOINT          = aws_rds_cluster.stock_analytics_aurora.endpoint
      REDIS_ENDPOINT        = aws_elasticache_replication_group.stock_analytics_redis.primary_endpoint_address
    }
  }
  
  dead_letter_config {
    target_arn = aws_sqs_queue.ml_inference_dlq.arn
  }
  
  tracing_config {
    mode = "Active"
  }
  
  tags = {
    Name = "ml-model-inference"
  }
}

resource "aws_lambda_function" "stock_recommendations_api" {
  filename         = "stock_recommendations_api.zip"
  function_name    = "stock-recommendations-api"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "index.handler"
  runtime         = "python3.11"
  timeout         = 30
  memory_size     = 512
  
  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }
  
  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.stock_recommendations.name
      REDIS_ENDPOINT = aws_elasticache_replication_group.stock_analytics_redis.primary_endpoint_address
    }
  }
  
  tracing_config {
    mode = "Active"
  }
  
  tags = {
    Name = "stock-recommendations-api"
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "stock_analytics_api" {
  name        = "stock-analytics-api"
  description = "API for stock analytics and recommendations"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  tags = {
    Name = "stock-analytics-api"
  }
}

resource "aws_api_gateway_resource" "recommendations" {
  rest_api_id = aws_api_gateway_rest_api.stock_analytics_api.id
  parent_id   = aws_api_gateway_rest_api.stock_analytics_api.root_resource_id
  path_part   = "recommendations"
}

resource "aws_api_gateway_method" "get_recommendations" {
  rest_api_id   = aws_api_gateway_rest_api.stock_analytics_api.id
  resource_id   = aws_api_gateway_resource.recommendations.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.stock_analytics_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.get_recommendations.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.stock_recommendations_api.invoke_arn
}

resource "aws_api_gateway_deployment" "stock_analytics_api_deployment" {
  depends_on = [
    aws_api_gateway_method.get_recommendations,
    aws_api_gateway_integration.lambda_integration,
  ]
  
  rest_api_id = aws_api_gateway_rest_api.stock_analytics_api.id
  stage_name  = var.environment
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_lambda_permission" "allow_api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stock_recommendations_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.stock_analytics_api.execution_arn}/*/*"
}

# SQS Queues
resource "aws_sqs_queue" "stock_data_processing" {
  name                      = "stock-data-processing"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 1209600
  receive_wait_time_seconds = 20
  
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.stock_data_dlq.arn
    maxReceiveCount     = 3
  })
  
  tags = {
    Name = "stock-data-processing"
  }
}

resource "aws_sqs_queue" "stock_data_dlq" {
  name = "stock-data-processing-dlq"
  
  tags = {
    Name = "stock-data-processing-dlq"
  }
}

resource "aws_sqs_queue" "ml_inference_dlq" {
  name = "ml-inference-dlq"
  
  tags = {
    Name = "ml-inference-dlq"
  }
}

# SNS Topics
resource "aws_sns_topic" "stock_alerts" {
  name = "stock-alerts"
  
  tags = {
    Name = "stock-alerts"
  }
}

resource "aws_sns_topic" "system_alerts" {
  name = "system-alerts"
  
  tags = {
    Name = "system-alerts"
  }
}

# EventBridge Rules
resource "aws_cloudwatch_event_rule" "stock_data_ingestion_schedule" {
  name                = "stock-data-ingestion-schedule"
  description         = "Trigger stock data ingestion every 15 minutes during market hours"
  schedule_expression = "rate(15 minutes)"
  
  tags = {
    Name = "stock-data-ingestion-schedule"
  }
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.stock_data_ingestion_schedule.name
  target_id = "StockDataIngestionTarget"
  arn       = aws_lambda_function.stock_data_ingestion.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stock_data_ingestion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stock_data_ingestion_schedule.arn
}

# CloudWatch Resources
resource "aws_cloudwatch_log_group" "lambda_logs" {
  for_each = toset([
    "/aws/lambda/stock-data-ingestion",
    "/aws/lambda/ml-model-inference", 
    "/aws/lambda/stock-recommendations-api"
  ])
  
  name              = each.value
  retention_in_days = 14
  
  tags = {
    Name = each.value
  }
}

resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/stock-analytics-api"
  retention_in_days = 30
  
  tags = {
    Name = "stock-analytics-api-logs"
  }
}

resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name              = "/aws/elasticache/redis/slow-log"
  retention_in_days = 7
  
  tags = {
    Name = "redis-slow-log"
  }
}

# CloudWatch Dashboards
resource "aws_cloudwatch_dashboard" "stock_analytics_dashboard" {
  dashboard_name = "StockAnalyticsEngine"
  
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
            ["AWS/Lambda", "Duration", "FunctionName", "stock-data-ingestion"],
            [".", "Errors", ".", "."],
            [".", "Invocations", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Lambda Performance Metrics"
          period  = 300
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
            ["AWS/ApiGateway", "Count", "ApiName", "stock-analytics-api"],
            [".", "Latency", ".", "."],
            [".", "4XXError", ".", "."],
            [".", "5XXError", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "API Gateway Metrics"
          period  = 300
        }
      }
    ]
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "lambda_error_rate" {
  alarm_name          = "stock-analytics-lambda-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors lambda error rate"
  
  dimensions = {
    FunctionName = aws_lambda_function.stock_data_ingestion.function_name
  }
  
  alarm_actions = [aws_sns_topic.system_alerts.arn]
  
  tags = {
    Name = "lambda-error-rate-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_gateway_4xx_errors" {
  alarm_name          = "stock-analytics-api-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4XXError"
  namespace           = "AWS/ApiGateway"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors API Gateway 4XX errors"
  
  dimensions = {
    ApiName = aws_api_gateway_rest_api.stock_analytics_api.name
  }
  
  alarm_actions = [aws_sns_topic.system_alerts.arn]
  
  tags = {
    Name = "api-gateway-4xx-errors-alarm"
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "lambda_execution_role" {
  name = "stock-analytics-lambda-execution-role"
  
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
  
  tags = {
    Name = "lambda-execution-role"
  }
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "stock-analytics-lambda-policy"
  role = aws_iam_role.lambda_execution_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface",
          "ec2:AttachNetworkInterface",
          "ec2:DetachNetworkInterface"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.stock_data_lake.arn}/*",
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      },
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
          aws_dynamodb_table.stock_recommendations.arn,
          aws_dynamodb_table.api_cache.arn,
          "${aws_dynamodb_table.stock_recommendations.arn}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = aws_sagemaker_endpoint.stock_prediction_endpoint.arn
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage"
        ]
        Resource = [
          aws_sqs_queue.stock_data_processing.arn,
          aws_sqs_queue.stock_data_dlq.arn,
          aws_sqs_queue.ml_inference_dlq.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = [
          aws_sns_topic.stock_alerts.arn,
          aws_sns_topic.system_alerts.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "stock-analytics-sagemaker-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "sagemaker-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_role_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role" "rds_monitoring_role" {
  name = "stock-analytics-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "rds-monitoring-role"
  }
}

resource "aws_iam_role_policy_attachment" "rds_monitoring_role_policy" {
  role       = aws_iam_role.rds_monitoring_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# X-Ray Tracing
resource "aws_xray_sampling_rule" "stock_analytics_sampling" {
  rule_name      = "stock-analytics-sampling"
  priority       = 9000
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.1
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"
}

# Outputs
output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = "https://${aws_api_gateway_rest_api.stock_analytics_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}"
}

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.stock_prediction_endpoint.name
}

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = aws_rds_cluster.stock_analytics_aurora.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.stock_analytics_redis.primary_endpoint_address
  sensitive   = true
}