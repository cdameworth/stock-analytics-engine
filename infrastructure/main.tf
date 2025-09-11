# Lower-cost overrides for Stock Analytics Engine

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# Random ID for bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 8
}

# Secrets Manager - Premium API Key
resource "aws_secretsmanager_secret" "alpha_vantage_premium_api_key" {
  name        = "stock-analytics-alpha-vantage-premium-api-key"
  description = "Premium Alpha Vantage API Key - 75 calls/minute"

  tags = {
    Application   = "stock-analytics-engine"
    Environment   = var.environment
    CostOptimized = "false"
    Tier          = "premium"
  }
}

# Legacy secret for backwards compatibility
resource "aws_secretsmanager_secret" "alpha_vantage_api_key" {
  name        = "stock-analytics-alpha-vantage-api-key"
  description = "Alpha Vantage API Key for Stock Analytics Engine"

  tags = {
    Application   = "stock-analytics-engine"
    Environment   = var.environment
    CostOptimized = "true"
  }
}

resource "aws_secretsmanager_secret_version" "alpha_vantage_api_key_version" {
  secret_id     = aws_secretsmanager_secret.alpha_vantage_api_key.id
  secret_string = var.alpha_vantage_api_key
}

# IAM Policies - Updated for Premium API access
resource "aws_iam_policy" "lambda_secrets_access_premium" {
  name = "stock-analytics-lambda-secrets-access-premium"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.alpha_vantage_api_key.arn,
          aws_secretsmanager_secret.alpha_vantage_premium_api_key.arn
        ]
      }
    ]
  })
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    {
      Name = "stock-analytics-vpc"
    },
    var.additional_tags
  )
}

resource "aws_subnet" "private_subnet_1" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = merge(
    {
      Name = "private-subnet-1"
    },
    var.additional_tags
  )
}

resource "aws_subnet" "private_subnet_2" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = merge(
    {
      Name = "private-subnet-2"
    },
    var.additional_tags
  )
}

# Security Groups
resource "aws_security_group" "valkey_sg" {
  name_prefix = "valkey-sg-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
    description = "Valkey access from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "valkey-security-group"
    },
    var.additional_tags
  )
}

resource "aws_security_group" "lambda_sg" {
  name_prefix = "lambda-sg-"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "lambda-security-group"
    },
    var.additional_tags
  )
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "rds-sg-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  tags = merge(
    {
      Name = "rds-security-group"
    },
    var.additional_tags
  )
}

# IAM Roles
resource "aws_iam_role" "lambda_execution_role" {
  name               = "lambda-execution-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = var.additional_tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_secrets_access_attach_premium" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = aws_iam_policy.lambda_secrets_access_premium.arn
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = var.additional_tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Add S3 access for model artifacts
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "sagemaker-s3-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*",
          aws_s3_bucket.stock_data_lake.arn,
          "${aws_s3_bucket.stock_data_lake.arn}/*"
        ]
      }
    ]
  })
}
# S3 Buckets
resource "aws_s3_bucket" "stock_data_lake" {
  bucket = "stock-analytics-data-lake-${random_id.bucket_suffix.hex}"

  tags = merge(
    {
      Name = "stock-data-lake"
    },
    var.additional_tags
  )
}

resource "aws_s3_bucket" "ml_models" {
  bucket = "stock-analytics-ml-models-${random_id.bucket_suffix.hex}"

  tags = merge(
    {
      Name = "ml-models-bucket"
    },
    var.additional_tags
  )
}

resource "aws_s3_bucket_lifecycle_configuration" "stock_data_lake_lifecycle" {
  bucket = aws_s3_bucket.stock_data_lake.id

  rule {
    id     = "move-to-glacier"
    status = "Enabled"

    filter {
      prefix = "" # Apply to all objects
    }

    transition {
      days          = var.s3_transition_glacier_days
      storage_class = "GLACIER"
    }

    expiration {
      days = var.s3_expiration_days
    }
  }
}

# RDS Aurora
resource "aws_db_subnet_group" "aurora" {
  name       = "aurora-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]

  tags = merge(
    {
      Name = "aurora-subnet-group"
    },
    var.additional_tags
  )
}

resource "aws_rds_cluster" "stock_analytics_aurora" {
  cluster_identifier          = "stock-analytics-aurora"
  engine                      = "aurora-postgresql"
  engine_version              = "15.4"
  database_name               = "stockanalytics"
  master_username             = "postgres"
  manage_master_user_password = true

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.aurora.name

  skip_final_snapshot = true
  deletion_protection = var.db_deletion_protection

  tags = merge(
    {
      Name = "stock-analytics-aurora"
    },
    var.additional_tags
  )
}

# Primary Aurora instance for Tier 1 capacity
resource "aws_rds_cluster_instance" "stock_analytics_aurora_instance" {
  count              = var.db_instance_class == "db.t4g.small" ? 1 : 2 # Single instance for current, dual for tiers
  identifier         = var.db_instance_class == "db.t4g.small" ? "stock-analytics-aurora-lowcost-${count.index}" : "stock-analytics-aurora-tier-${count.index}"
  cluster_identifier = aws_rds_cluster.stock_analytics_aurora.id
  engine             = aws_rds_cluster.stock_analytics_aurora.engine
  engine_version     = aws_rds_cluster.stock_analytics_aurora.engine_version

  performance_insights_enabled = var.db_instance_class != "db.t4g.small"
  monitoring_interval          = 0 # Disabled to avoid monitoring role requirement
  auto_minor_version_upgrade   = var.db_instance_class != "db.t4g.small"

  instance_class        = var.db_instance_class
  copy_tags_to_snapshot = var.db_instance_class != "db.t4g.small"
  promotion_tier        = count.index

  tags = merge(
    {
      Name = var.db_instance_class == "db.t4g.small" ? "stock-analytics-aurora-lowcost-${count.index}" : "stock-analytics-aurora-tier-${count.index}"
      Role = count.index == 0 ? "primary" : "replica"
    },
    var.additional_tags
  )
}

# ElastiCache Valkey
resource "aws_elasticache_subnet_group" "stock_analytics_valkey_subnet_group" {
  name       = "stock-analytics-valkey-subnet-group"
  subnet_ids = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]

  tags = var.additional_tags
}

resource "aws_elasticache_parameter_group" "valkey_params" {
  family = "valkey7"
  name   = var.valkey_node_type == "cache.t4g.micro" ? "valkey-params-lowcost" : "valkey-params-tier"

  parameter {
    name  = "maxmemory-policy"
    value = var.valkey_node_type == "cache.t4g.micro" ? "volatile-lru" : "allkeys-lru"
  }

  dynamic "parameter" {
    for_each = var.valkey_node_type == "cache.t4g.micro" ? [] : [1]
    content {
      name  = "timeout"
      value = "300"
    }
  }

  dynamic "parameter" {
    for_each = var.valkey_node_type == "cache.t4g.micro" ? [] : [1]
    content {
      name  = "tcp-keepalive"
      value = "60"
    }
  }

  tags = merge(
    {
      Name = var.valkey_node_type == "cache.t4g.micro" ? "valkey-parameter-group-lowcost" : "valkey-parameter-group-tier"
    },
    var.additional_tags
  )
}

resource "aws_elasticache_replication_group" "stock_analytics_valkey" {
  replication_group_id = var.valkey_node_type == "cache.t4g.micro" ? "stock-analytics-valkey-lowcost" : "stock-analytics-valkey-tier"
  description          = var.valkey_node_type == "cache.t4g.micro" ? "Low-cost Valkey cluster" : "High-performance Valkey cluster for stock analytics caching"

  engine               = "valkey"
  engine_version       = "7.2"
  node_type            = var.valkey_node_type
  num_cache_clusters   = var.valkey_num_cache_clusters
  port                 = 6379
  parameter_group_name = aws_elasticache_parameter_group.valkey_params.name

  automatic_failover_enabled = var.valkey_num_cache_clusters > 1
  multi_az_enabled           = var.valkey_num_cache_clusters > 1

  subnet_group_name  = aws_elasticache_subnet_group.stock_analytics_valkey_subnet_group.name
  security_group_ids = [aws_security_group.valkey_sg.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = var.valkey_node_type == "cache.t4g.micro" ? 1 : 5
  snapshot_window          = "05:00-06:00"
  maintenance_window       = "sun:06:00-sun:07:00"

  tags = merge(
    {
      Name             = var.valkey_node_type == "cache.t4g.micro" ? "stock-analytics-valkey-lowcost" : "stock-analytics-valkey-tier"
      Engine           = "valkey"
      HighAvailability = var.valkey_num_cache_clusters > 1 ? "true" : "false"
    },
    var.additional_tags
  )
}

# SageMaker (conditional on not being completely disabled)
resource "aws_sagemaker_model" "stock_prediction_model" {
  count              = var.disable_sagemaker_entirely ? 0 : 1
  name               = "stock-prediction-model-v2-compatible"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    # Use the correct public SageMaker Scikit-learn image
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
    model_data_url = "s3://${aws_s3_bucket.ml_models.bucket}/stock_prediction_simple.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
    }
  }

  tags = var.additional_tags
}

resource "aws_sagemaker_endpoint_configuration" "stock_prediction_endpoint_config" {
  count = var.disable_sagemaker_entirely ? 0 : 1
  name  = var.sagemaker_instance_type == "ml.t2.small" ? "stock-prediction-endpoint-config-lowcost" : "stock-prediction-endpoint-config-tier"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = var.disable_sagemaker_entirely ? null : aws_sagemaker_model.stock_prediction_model[0].name
    initial_instance_count = var.enable_sagemaker_serverless ? null : var.sagemaker_instance_count
    instance_type          = var.enable_sagemaker_serverless ? null : var.sagemaker_instance_type

    dynamic "serverless_config" {
      for_each = var.enable_sagemaker_serverless ? [1] : []
      content {
        memory_size_in_mb = var.sagemaker_serverless_memory_mb
        max_concurrency   = var.sagemaker_serverless_max_concurrency
      }
    }
  }

  # Enable data capture for higher tiers
  dynamic "data_capture_config" {
    for_each = var.sagemaker_instance_type != "ml.t2.small" && !var.enable_sagemaker_serverless ? [1] : []
    content {
      enable_capture              = true
      initial_sampling_percentage = 20
      destination_s3_uri          = "s3://${aws_s3_bucket.ml_models.bucket}/model-monitor/"

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
  }

  tags = merge(
    {
      Name             = var.sagemaker_instance_type == "ml.t2.small" ? "stock-prediction-endpoint-config-lowcost" : "stock-prediction-endpoint-config-tier"
      HighAvailability = var.sagemaker_instance_count > 1 ? "true" : "false"
    },
    var.additional_tags
  )
}

# SageMaker endpoint removed - no longer in use
# resource "aws_sagemaker_endpoint" "stock_prediction_endpoint" {
#   count = var.disable_sagemaker_entirely ? 0 : 1
#   name                 = var.sagemaker_instance_type == "ml.t2.small" ? "stock-prediction-endpoint-lowcost" : "stock-prediction-endpoint-tier"
#   endpoint_config_name = var.disable_sagemaker_entirely ? null : aws_sagemaker_endpoint_configuration.stock_prediction_endpoint_config[0].name
# 
#   tags = merge(
#     {
#       Name = var.sagemaker_instance_type == "ml.t2.small" ? "stock-prediction-endpoint-lowcost" : "stock-prediction-endpoint-tier"
#       HighAvailability = var.sagemaker_instance_count > 1 ? "true" : "false"
#     },
#     var.additional_tags
#   )
# }

# Lambda Function
data "archive_file" "ml_model_inference" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/ml_model_inference.py"
  output_path = "${path.module}/ml_model_inference.zip"
}

data "archive_file" "stock_data_ingestion" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/stock_data_ingestion.py"
  output_path = "${path.module}/stock_data_ingestion.zip"
}

data "archive_file" "stock_recommendations_api" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/stock_recommendations_api.py"
  output_path = "${path.module}/stock_recommendations_api.zip"
}

# Lambda Function - ML Model Inference
resource "aws_lambda_function" "ml_model_inference" {
  filename         = data.archive_file.ml_model_inference.output_path
  source_code_hash = data.archive_file.ml_model_inference.output_base64sha256
  function_name    = var.lambda_memory_size > 512 ? "ml-model-inference-tier" : "ml-model-inference-lowcost"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "ml_model_inference.lambda_handler"
  runtime          = "python3.11"
  timeout          = var.lambda_timeout
  memory_size      = var.lambda_memory_size

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      SAGEMAKER_ENDPOINT               = "disabled" # SageMaker endpoint removed
      DYNAMODB_TABLE                   = aws_dynamodb_table.stock_recommendations.name
      S3_BUCKET                        = aws_s3_bucket.stock_data_lake.bucket
      RDS_ENDPOINT                     = aws_rds_cluster.stock_analytics_aurora.endpoint
      VALKEY_ENDPOINT                  = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
      REDIS_ENDPOINT                   = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
      ALPHA_VANTAGE_API_KEY_SECRET_ARN = var.use_premium_api_key ? aws_secretsmanager_secret.alpha_vantage_premium_api_key.arn : aws_secretsmanager_secret.alpha_vantage_api_key.arn
      PREMIUM_API_CALLS_PER_MINUTE     = var.premium_api_calls_per_minute
      USE_PREMIUM_API_KEY              = var.use_premium_api_key
    }
  }

  ephemeral_storage {
    size = 512
  }

  # Use AWS-provided NumPy layer (for x86_64)
  #layers = [
  #  "arn:aws:lambda:us-east-1:668099181075:layer:AWSLambda-Python311-SciPy1x:112"
  #]

  architectures = var.enable_graviton_instances ? ["arm64"] : ["x86_64"]

  tracing_config {
    mode = "Active"
  }

  tags = merge(
    {
      Name            = var.lambda_memory_size > 512 ? "ml-model-inference-tier" : "ml-model-inference-lowcost"
      HighPerformance = var.lambda_memory_size > 512 ? "true" : "false"
    },
    var.additional_tags
  )
}

# Provisioned Concurrency for ML Model Inference Lambda
resource "aws_lambda_provisioned_concurrency_config" "ml_model_inference_concurrency" {
  count                             = var.enable_lambda_provisioned_concurrency ? 1 : 0
  function_name                     = aws_lambda_function.ml_model_inference.function_name
  provisioned_concurrent_executions = var.lambda_provisioned_concurrency_count
  qualifier                         = "$LATEST"
}

# Lambda Function - Stock Data Ingestion (UPDATE THIS SECTION)
resource "aws_lambda_function" "stock_data_ingestion" {
  filename         = data.archive_file.stock_data_ingestion.output_path
  source_code_hash = data.archive_file.stock_data_ingestion.output_base64sha256
  function_name    = "stock-data-ingestion"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "stock_data_ingestion.lambda_handler"
  runtime          = "python3.11"
  timeout          = var.lambda_timeout
  memory_size      = var.lambda_memory_size

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  layers = [
    "arn:aws:lambda:us-east-1:791060928878:layer:valkey-redis-py311:1"
  ]

  environment {
    variables = {
      ALPHA_VANTAGE_API_KEY_SECRET_ARN = var.use_premium_api_key ? aws_secretsmanager_secret.alpha_vantage_premium_api_key.arn : aws_secretsmanager_secret.alpha_vantage_api_key.arn
      S3_BUCKET                        = aws_s3_bucket.stock_data_lake.bucket
      DYNAMODB_TABLE                   = aws_dynamodb_table.stock_recommendations.name
      VALKEY_ENDPOINT                  = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
      ML_INFERENCE_FUNCTION_NAME       = aws_lambda_function.ml_model_inference.function_name
      MAX_SYMBOLS_PER_RUN              = var.use_premium_api_key ? "30" : "12"
      PER_CALL_TIMEOUT                 = var.use_premium_api_key ? "4" : "8"
      PREMIUM_API_CALLS_PER_MINUTE     = var.premium_api_calls_per_minute
      USE_PREMIUM_API_KEY              = var.use_premium_api_key
      COMPREHENSIVE_STOCK_COVERAGE     = var.use_premium_api_key ? "true" : "false"
    }
  }

  tags = merge(
    {
      Name            = "stock-data-ingestion"
      HighPerformance = var.lambda_memory_size > 512 ? "true" : "false"
    },
    var.additional_tags
  )
}

# Provisioned Concurrency for Stock Data Ingestion Lambda
resource "aws_lambda_provisioned_concurrency_config" "stock_data_ingestion_concurrency" {
  count                             = var.enable_lambda_provisioned_concurrency ? 1 : 0
  function_name                     = aws_lambda_function.stock_data_ingestion.function_name
  provisioned_concurrent_executions = var.lambda_provisioned_concurrency_count
  qualifier                         = "$LATEST"
}


# Lambda Function - Stock Recommendations API (UPDATE THIS SECTION)
resource "aws_lambda_function" "stock_recommendations_api" {
  filename         = data.archive_file.stock_recommendations_api.output_path
  source_code_hash = data.archive_file.stock_recommendations_api.output_base64sha256
  function_name    = "stock-recommendations-api"
  role             = aws_iam_role.lambda_execution_role.arn
  handler          = "stock_recommendations_api.lambda_handler"
  runtime          = "python3.11"
  timeout          = 30
  memory_size      = 512

  vpc_config {
    subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      DYNAMODB_TABLE  = aws_dynamodb_table.stock_recommendations.name
      VALKEY_ENDPOINT = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
      REDIS_ENDPOINT  = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
    }
  }

  tags = merge(
    {
      Name          = "stock-recommendations-api"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# Internet Gateway
resource "aws_internet_gateway" "igw" {
  count  = var.enable_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.main.id

  tags = merge(
    {
      Name = "stock-analytics-igw"
    },
    var.additional_tags
  )
}

resource "aws_subnet" "public_nat_subnet" {
  count                   = var.enable_nat_gateway ? 1 : 0
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.3.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  tags                    = merge({ Name = "public-nat-subnet" }, var.additional_tags)
}

# Elastic IP for NAT
resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? 1 : 0
  domain = "vpc"

  tags = merge(
    {
      Name = "stock-analytics-nat-eip"
    },
    var.additional_tags
  )
}

# NAT Gateway
resource "aws_nat_gateway" "nat_gw" {
  count         = var.enable_nat_gateway ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public_nat_subnet[0].id

  tags = merge(
    {
      Name = "stock-analytics-nat-gw"
    },
    var.additional_tags
  )

  depends_on = [aws_internet_gateway.igw]
}

# Public route table (route to IGW)
resource "aws_route_table" "public_rt" {
  count  = var.enable_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw[0].id
  }

  tags = merge(
    {
      Name = "public-rt"
    },
    var.additional_tags
  )
}

# Associate public subnet
resource "aws_route_table_association" "public_assoc_1" {
  count          = var.enable_nat_gateway ? 1 : 0
  subnet_id      = aws_subnet.public_nat_subnet[0].id
  route_table_id = aws_route_table.public_rt[0].id
}

# Private route table (default route via NAT)
resource "aws_route_table" "private_rt" {
  count  = var.enable_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw[0].id
  }

  tags = merge(
    {
      Name = "private-rt"
    },
    var.additional_tags
  )
}

# Associate existing private subnets to private RT
resource "aws_route_table_association" "private_assoc_1" {
  count          = var.enable_nat_gateway ? 1 : 0
  subnet_id      = aws_subnet.private_subnet_1.id
  route_table_id = aws_route_table.private_rt[0].id
}

resource "aws_route_table_association" "private_assoc_2" {
  count          = var.enable_nat_gateway ? 1 : 0
  subnet_id      = aws_subnet.private_subnet_2.id
  route_table_id = aws_route_table.private_rt[0].id
}

# Add Lambda permissions for DynamoDB and SageMaker
resource "aws_iam_role_policy" "lambda_sagemaker_dynamodb" {
  name = "lambda-sagemaker-dynamodb-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "arn:aws:sagemaker:*:*:endpoint/disabled" # SageMaker endpoint removed
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Scan",
          "dynamodb:Query"
        ]
        Resource = [
          aws_dynamodb_table.stock_recommendations.arn,
          "${aws_dynamodb_table.stock_recommendations.arn}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "arn:aws:lambda:*:*:function:ml-model-inference*"
      }
    ]
  })
}

# CloudWatch Custom Metrics Policy for Enhanced Observability
resource "aws_iam_role_policy" "lambda_cloudwatch_metrics" {
  name = "lambda-cloudwatch-custom-metrics-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = [
              "StockAnalytics/DataIngestion",
              "StockAnalytics/MLInference",
              "AWS/Lambda"
            ]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "lambda-s3-ingestion-access"
  role = aws_iam_role.lambda_execution_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowPutGetListDataLake"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.stock_data_lake.arn,
          "${aws_s3_bucket.stock_data_lake.arn}/*"
        ]
      }
    ]
  })
}

# Add missing DynamoDB table for your Lambda functions
resource "aws_dynamodb_table" "stock_recommendations" {
  name         = "stock-recommendations"
  billing_mode = "PAY_PER_REQUEST" # Cost-optimized
  hash_key     = "recommendation_id"

  attribute {
    name = "recommendation_id"
    type = "S"
  }

  attribute {
    name = "symbol"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  # GSI for querying by symbol
  global_secondary_index {
    name            = "symbol-timestamp-index"
    hash_key        = "symbol"
    range_key       = "timestamp"
    projection_type = "ALL" # Add this required argument
  }

  # TTL for automatic cleanup
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = merge(
    {
      Name          = "stock-recommendations"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# API Gateway
resource "aws_api_gateway_rest_api" "stock_recommendations_api" {
  name        = "stock-recommendations-api"
  description = "API for stock recommendations"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = merge(
    {
      Name          = "stock-recommendations-api"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# API Gateway Resources
resource "aws_api_gateway_resource" "recommendations" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_rest_api.stock_recommendations_api.root_resource_id
  path_part   = "recommendations"
}

resource "aws_api_gateway_resource" "recommendations_symbol" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_resource.recommendations.id
  path_part   = "{symbol}"
}

# API Gateway Resource - /dual-predictions
resource "aws_api_gateway_resource" "dual_predictions" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_rest_api.stock_recommendations_api.root_resource_id
  path_part   = "dual-predictions"
}

# API Gateway Resource - /dual-predictions/analytics
resource "aws_api_gateway_resource" "dual_predictions_analytics" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_resource.dual_predictions.id
  path_part   = "analytics"
}

# API Gateway Resource - /analytics (for frontend compatibility)
resource "aws_api_gateway_resource" "analytics" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_rest_api.stock_recommendations_api.root_resource_id
  path_part   = "analytics"
}

# API Gateway Resource - /analytics/dashboard
resource "aws_api_gateway_resource" "analytics_dashboard" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  parent_id   = aws_api_gateway_resource.analytics.id
  path_part   = "dashboard"
}

# API Gateway Methods - GET /recommendations
resource "aws_api_gateway_method" "get_recommendations" {
  rest_api_id      = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id      = aws_api_gateway_resource.recommendations.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true

  request_parameters = {
    "method.request.querystring.limit" = false
    "method.request.querystring.page"  = false
  }
}

# API Gateway Methods - GET /recommendations/{symbol}
resource "aws_api_gateway_method" "get_recommendation_by_symbol" {
  rest_api_id      = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id      = aws_api_gateway_resource.recommendations_symbol.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true

  request_parameters = {
    "method.request.path.symbol" = true
  }
}

# API Gateway Integration - GET /recommendations
resource "aws_api_gateway_integration" "get_recommendations_integration" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.get_recommendations.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.stock_recommendations_api.invoke_arn
}

# API Gateway Integration - GET /recommendations/{symbol}
resource "aws_api_gateway_integration" "get_recommendation_by_symbol_integration" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations_symbol.id
  http_method = aws_api_gateway_method.get_recommendation_by_symbol.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.stock_recommendations_api.invoke_arn
}

# Lambda permissions for API Gateway
resource "aws_lambda_permission" "api_gateway_get_recommendations" {
  statement_id  = "AllowExecutionFromAPIGateway-recommendations"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stock_recommendations_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.stock_recommendations_api.execution_arn}/*/*"
}

# API Gateway Method - GET /dual-predictions/analytics
resource "aws_api_gateway_method" "get_dual_predictions_analytics" {
  rest_api_id      = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id      = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true
}

# API Gateway Method - GET /analytics/dashboard
resource "aws_api_gateway_method" "get_analytics_dashboard" {
  rest_api_id      = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id      = aws_api_gateway_resource.analytics_dashboard.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true
}

# API Gateway Integration - GET /dual-predictions/analytics
resource "aws_api_gateway_integration" "get_dual_predictions_analytics_integration" {
  rest_api_id             = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id             = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method             = aws_api_gateway_method.get_dual_predictions_analytics.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:lambda:path/2015-03-31/functions/arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:dual-prediction-reporting-api/invocations"
}

# API Gateway Integration - GET /analytics/dashboard
resource "aws_api_gateway_integration" "get_analytics_dashboard_integration" {
  rest_api_id             = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id             = aws_api_gateway_resource.analytics_dashboard.id
  http_method             = aws_api_gateway_method.get_analytics_dashboard.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:lambda:path/2015-03-31/functions/arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:dual-prediction-reporting-api/invocations"
}

# Lambda permission for dual prediction API
resource "aws_lambda_permission" "api_gateway_dual_predictions" {
  statement_id  = "AllowExecutionFromAPIGateway-dual-predictions"
  action        = "lambda:InvokeFunction"
  function_name = "dual-prediction-reporting-api"
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.stock_recommendations_api.execution_arn}/*/*"
}







# CORS Integrations - Mock integrations for OPTIONS
resource "aws_api_gateway_integration" "options_recommendations" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method
  type        = "MOCK"
  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}





# CORS Method Responses
resource "aws_api_gateway_method_response" "options_recommendations" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}





# CORS Integration Responses
resource "aws_api_gateway_integration_response" "options_recommendations" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,x-api-key,Authorization'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}





# CORS Response Headers for actual API responses
resource "aws_api_gateway_method_response" "get_recommendations_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.get_recommendations.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_method_response" "get_recommendations_symbol_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations_symbol.id
  http_method = aws_api_gateway_method.get_recommendation_by_symbol.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_method_response" "get_dual_predictions_analytics_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method = aws_api_gateway_method.get_dual_predictions_analytics.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

# Method Response - GET /analytics/dashboard
resource "aws_api_gateway_method_response" "get_analytics_dashboard_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.analytics_dashboard.id
  http_method = aws_api_gateway_method.get_analytics_dashboard.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

# Integration Response - GET /analytics/dashboard
resource "aws_api_gateway_integration_response" "get_analytics_dashboard_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.analytics_dashboard.id
  http_method = aws_api_gateway_method.get_analytics_dashboard.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
}

# OPTIONS method for analytics dashboard CORS
resource "aws_api_gateway_method" "options_analytics_dashboard" {
  rest_api_id   = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id   = aws_api_gateway_resource.analytics_dashboard.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_analytics_dashboard" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.analytics_dashboard.id
  http_method = aws_api_gateway_method.options_analytics_dashboard.http_method
  type        = "MOCK"
  request_templates = {
    "application/json" = "{\"statusCode\":200}"
  }
}

resource "aws_api_gateway_method_response" "options_analytics_dashboard_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.analytics_dashboard.id
  http_method = aws_api_gateway_method.options_analytics_dashboard.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_analytics_dashboard_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.analytics_dashboard.id
  http_method = aws_api_gateway_method.options_analytics_dashboard.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }


  depends_on = [aws_api_gateway_method_response.options_analytics_dashboard_200]
}

# CORS Integration Responses for actual API responses
resource "aws_api_gateway_integration_response" "get_recommendations_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.get_recommendations.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
}

resource "aws_api_gateway_integration_response" "get_recommendations_symbol_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations_symbol.id
  http_method = aws_api_gateway_method.get_recommendation_by_symbol.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
}

resource "aws_api_gateway_integration_response" "get_dual_predictions_analytics_200" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method = aws_api_gateway_method.get_dual_predictions_analytics.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
}

# CORS OPTIONS method for dual-predictions/analytics
resource "aws_api_gateway_method" "options_dual_predictions_analytics" {
  rest_api_id   = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id   = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_dual_predictions_analytics_integration" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method = aws_api_gateway_method.options_dual_predictions_analytics.http_method
  type        = "MOCK"
  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_dual_predictions_analytics_response" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method = aws_api_gateway_method.options_dual_predictions_analytics.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_dual_predictions_analytics_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.dual_predictions_analytics.id
  http_method = aws_api_gateway_method.options_dual_predictions_analytics.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,x-api-key,Authorization'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }

  depends_on = [aws_api_gateway_method_response.options_dual_predictions_analytics_response]
}

# API Gateway Deployment
resource "aws_api_gateway_deployment" "stock_recommendations_api_deployment" {
  depends_on = [
    aws_api_gateway_integration.get_recommendations_integration,
    aws_api_gateway_integration.get_recommendation_by_symbol_integration,
    aws_api_gateway_integration.get_dual_predictions_analytics_integration,
    aws_api_gateway_integration.options_dual_predictions_analytics_integration,
  ]

  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id

  # Trigger redeployment when any method/integration changes
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.recommendations.id,
      aws_api_gateway_method.get_recommendations.id,
      aws_api_gateway_integration.get_recommendations_integration.id,
      aws_api_gateway_resource.recommendations_symbol.id,
      aws_api_gateway_method.get_recommendation_by_symbol.id,
      aws_api_gateway_integration.get_recommendation_by_symbol_integration.id,
      aws_api_gateway_resource.dual_predictions.id,
      aws_api_gateway_resource.dual_predictions_analytics.id,
      aws_api_gateway_method.get_dual_predictions_analytics.id,
      aws_api_gateway_integration.get_dual_predictions_analytics_integration.id,
      aws_api_gateway_method.options_dual_predictions_analytics.id,
      aws_api_gateway_integration.options_dual_predictions_analytics_integration.id,
      aws_api_gateway_resource.analytics_dashboard.id,
      aws_api_gateway_method.options_analytics_dashboard.id,
      aws_api_gateway_integration.options_analytics_dashboard.id,
      "cors-fix-2025-09-05-v4",
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

# API Gateway Stage
resource "aws_api_gateway_stage" "stock_recommendations_api_stage" {
  deployment_id = aws_api_gateway_deployment.stock_recommendations_api_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.stock_recommendations_api.id
  stage_name    = "prod"

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs_lowcost.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      caller         = "$context.identity.caller"
      user           = "$context.identity.user"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  # Ensure the account settings are configured before creating the stage
  depends_on = [aws_api_gateway_account.api_gateway_account]

  tags = merge(
    {
      Name          = "stock-recommendations-api-prod"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# API Gateway Custom Domain (optional - only if domain_name is provided)
resource "aws_api_gateway_domain_name" "stock_api_domain" {
  count           = var.domain_name != "" ? 1 : 0
  domain_name     = var.domain_name
  certificate_arn = var.certificate_arn

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = merge(
    {
      Name          = "stock-api-custom-domain"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# API Gateway Base Path Mapping
resource "aws_api_gateway_base_path_mapping" "stock_api_mapping" {
  count       = var.domain_name != "" ? 1 : 0
  api_id      = aws_api_gateway_rest_api.stock_recommendations_api.id
  stage_name  = aws_api_gateway_stage.stock_recommendations_api_stage.stage_name
  domain_name = aws_api_gateway_domain_name.stock_api_domain[0].domain_name
  base_path   = "v1" # Maps to your-domain.com/v1/recommendations
}

# Route 53 DNS Record (optional - only if hosted_zone_id is provided)
resource "aws_route53_record" "stock_api_dns" {
  count   = var.domain_name != "" && var.hosted_zone_id != "" ? 1 : 0
  zone_id = var.hosted_zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_api_gateway_domain_name.stock_api_domain[0].regional_domain_name
    zone_id                = aws_api_gateway_domain_name.stock_api_domain[0].regional_zone_id
    evaluate_target_health = true
  }
}

# CORS support (if needed for web frontend)
resource "aws_api_gateway_method" "options_recommendations" {
  rest_api_id   = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id   = aws_api_gateway_resource.recommendations.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_recommendations_integration" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method

  type = "MOCK"

  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# Fix CORS integration response (use response_parameters instead of response_headers)
resource "aws_api_gateway_method_response" "options_recommendations_response" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

# Fix CORS integration response - remove the line that says "response_headers"
resource "aws_api_gateway_integration_response" "options_recommendations_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.stock_recommendations_api.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options_recommendations.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }

  depends_on = [aws_api_gateway_method_response.options_recommendations_response]
}

# MAXIMIZED EventBridge scheduling - Market hours every 5 minutes
resource "aws_cloudwatch_event_rule" "stock_data_ingestion_schedule" {
  name                = "stock-data-ingestion-market-hours"
  description         = "Trigger stock data ingestion every 5 minutes during market hours"
  schedule_expression = var.stock_data_ingestion_schedule # Every 5 minutes during market hours

  tags = merge(
    {
      Name              = "stock-data-ingestion-market-hours"
      CostOptimized     = "true"
      MaximizedCoverage = "true"
    },
    var.additional_tags
  )
}

# Additional evening processing for international markets and extended coverage
resource "aws_cloudwatch_event_rule" "stock_data_ingestion_evening" {
  name                = "stock-data-ingestion-evening"
  description         = "Evening stock data processing for international markets and extended coverage"
  schedule_expression = "cron(*/10 22-23,0-6 ? * MON-FRI *)" # Every 10 minutes from 5-11 PM EST and overnight

  tags = merge(
    {
      Name              = "stock-data-ingestion-evening"
      CostOptimized     = "true"
      MaximizedCoverage = "true"
      Schedule          = "evening"
    },
    var.additional_tags
  )
}

# EventBridge targets for MAXIMIZED coverage
resource "aws_cloudwatch_event_target" "stock_data_ingestion_target" {
  rule      = aws_cloudwatch_event_rule.stock_data_ingestion_schedule.name
  target_id = "StockDataIngestionMarketHours"
  arn       = aws_lambda_function.stock_data_ingestion.arn

  input = jsonencode({
    source          = "eventbridge-market-hours"
    processing_mode = "market_hours"
    trigger         = "scheduled_market_hours"
  })
}

resource "aws_cloudwatch_event_target" "stock_data_ingestion_evening_target" {
  rule      = aws_cloudwatch_event_rule.stock_data_ingestion_evening.name
  target_id = "StockDataIngestionEvening"
  arn       = aws_lambda_function.stock_data_ingestion.arn

  input = jsonencode({
    source          = "eventbridge-evening"
    processing_mode = "evening_extended"
    trigger         = "scheduled_evening"
  })
}

# Lambda permissions for EventBridge to invoke the function
resource "aws_lambda_permission" "allow_eventbridge_stock_ingestion" {
  statement_id  = "AllowExecutionFromEventBridgeMarketHours"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stock_data_ingestion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stock_data_ingestion_schedule.arn
}

resource "aws_lambda_permission" "allow_eventbridge_stock_ingestion_evening" {
  statement_id  = "AllowExecutionFromEventBridgeEvening"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stock_data_ingestion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stock_data_ingestion_evening.arn
}

# Optional: Trigger ML inference after data ingestion
resource "aws_cloudwatch_event_rule" "trigger_ml_inference" {
  name        = "trigger-ml-inference"
  description = "Trigger ML inference after stock data ingestion"

  event_pattern = jsonencode({
    source      = ["custom.stock-analytics"]
    detail-type = ["Stock Data Ingestion Complete"]
    detail = {
      status = ["SUCCESS"]
    }
  })

  tags = merge(
    {
      Name          = "trigger-ml-inference"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# IAM role for API Gateway CloudWatch logging
resource "aws_iam_role" "api_gateway_cloudwatch_role" {
  name = "api-gateway-cloudwatch-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    {
      Name          = "api-gateway-cloudwatch-role"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

resource "aws_iam_user_policy" "admin_allow_lambda_layer_get" {
  user = "admin"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowGetSpecificSciPyLayer"
        Effect = "Allow"
        Action = [
          "lambda:GetLayerVersion"
        ]
        Resource = "arn:aws:lambda:us-east-1:668099181075:layer:AWSLambda-Python311-SciPy1x:112"
      }
    ]
  })
}

# Attach the managed policy for API Gateway CloudWatch logging
resource "aws_iam_role_policy_attachment" "api_gateway_cloudwatch_policy" {
  role       = aws_iam_role.api_gateway_cloudwatch_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs"
}

# Set the CloudWatch role ARN in API Gateway account settings
resource "aws_api_gateway_account" "api_gateway_account" {
  cloudwatch_role_arn = aws_iam_role.api_gateway_cloudwatch_role.arn

  depends_on = [aws_iam_role_policy_attachment.api_gateway_cloudwatch_policy]
}

# CloudWatch Log Group for API Gateway (make sure this exists)
resource "aws_cloudwatch_log_group" "api_gateway_logs_lowcost" {
  name              = "/aws/apigateway/stock-recommendations-api"
  retention_in_days = var.api_gateway_log_retention

  tags = merge(
    {
      Name          = "api-gateway-logs-lowcost"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# Attach the managed policy for API Gateway CloudWatch logging
resource "aws_cloudwatch_event_target" "trigger_ml_inference_target" {
  rule      = aws_cloudwatch_event_rule.trigger_ml_inference.name
  target_id = "TriggerMLInferenceTarget"
  arn       = aws_lambda_function.ml_model_inference.arn

  input_transformer {
    input_paths = {
      symbols = "$.detail.symbols"
    }
    input_template = jsonencode({
      source  = "eventbridge-trigger"
      symbols = "<symbols>"
      trigger = "post-ingestion"
    })
  }
}

resource "aws_lambda_permission" "allow_eventbridge_ml_inference" {
  statement_id  = "AllowExecutionFromEventBridgeMLInference"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ml_model_inference.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.trigger_ml_inference.arn
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "lambda_logs_lowcost" {
  for_each = toset([
    "/aws/lambda/stock-data-ingestion",
    "/aws/lambda/ml-model-inference-lowcost",
    "/aws/lambda/stock-recommendations-api"
  ])

  name              = each.value
  retention_in_days = var.lambda_log_retention

  tags = merge(
    {
      Name          = each.value
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# Cost Monitoring
resource "aws_cloudwatch_metric_alarm" "cost_alarm" {
  alarm_name          = "stock-analytics-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400" # 24 hours
  statistic           = "Maximum"
  threshold           = var.cost_alarm_threshold
  alarm_description   = "This metric monitors estimated charges for stock analytics engine"
  alarm_actions       = [aws_sns_topic.cost_alerts.arn]

  dimensions = {
    Currency = "USD"
  }

  tags = merge(
    {
      Name          = "stock-analytics-cost-alarm"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# SNS Topic for cost alerts
resource "aws_sns_topic" "cost_alerts" {
  name = "stock-analytics-cost-alerts"

  tags = merge(
    {
      Name          = "stock-analytics-cost-alerts"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}

# SNS Topic Subscription (add your email)
resource "aws_sns_topic_subscription" "cost_alerts_email" {
  count     = var.cost_alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "email"
  endpoint  = var.cost_alert_email
}

# SSM Parameter to store the current API Gateway URL
resource "aws_ssm_parameter" "api_gateway_url" {
  name      = "/stock-analytics/api/gateway-url"
  type      = "String"
  value     = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}"
  overwrite = true

  description = "Current API Gateway URL for stock recommendations API"

  tags = merge(
    {
      Name    = "stock-analytics-api-url"
      Purpose = "URL distribution to dependent applications"
    },
    var.additional_tags
  )
}

# S3 bucket for API configuration (alternative method)
resource "aws_s3_bucket" "api_config" {
  bucket = "stock-analytics-api-config-${random_id.bucket_suffix.hex}"

  tags = merge(
    {
      Name    = "stock-analytics-api-config"
      Purpose = "API configuration distribution"
    },
    var.additional_tags
  )
}

# Upload API configuration as JSON to S3
resource "aws_s3_object" "api_config_json" {
  bucket = aws_s3_bucket.api_config.id
  key    = "api-config.json"

  content = jsonencode({
    api_gateway_url = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}"
    endpoints = {
      get_all_recommendations    = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations"
      get_recommendation         = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations/{symbol}"
      dual_predictions_analytics = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/dual-predictions/analytics"
      analytics_dashboard        = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/analytics/dashboard"
    }
    last_updated = timestamp()
    version      = "1.0"
  })

  content_type = "application/json"

  tags = merge(
    {
      Name    = "api-config"
      Purpose = "Current API endpoints configuration"
    },
    var.additional_tags
  )
}

# Daily cost report Lambda (optional)
resource "aws_cloudwatch_event_rule" "daily_cost_report" {
  name                = "daily-cost-report"
  description         = "Trigger daily cost report"
  schedule_expression = "cron(0 13 * * ? *)" # Daily at 8 AM EST (13:00 UTC)

  tags = merge(
    {
      Name          = "daily-cost-report"
      CostOptimized = "true"
    },
    var.additional_tags
  )
}