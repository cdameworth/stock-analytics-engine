output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
}

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = aws_rds_cluster.stock_analytics_aurora.endpoint
  sensitive   = true
}

output "valkey_endpoint" {
  description = "Valkey primary endpoint"
  value       = aws_elasticache_replication_group.stock_analytics_valkey.primary_endpoint_address
  sensitive   = true
}

output "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  value       = "disabled" # SageMaker endpoint removed
}

output "s3_data_lake_bucket" {
  description = "S3 data lake bucket name"
  value       = aws_s3_bucket.stock_data_lake.bucket
}

output "s3_ml_models_bucket" {
  description = "S3 ML models bucket name"
  value       = aws_s3_bucket.ml_models.bucket
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.ml_model_inference.function_name
}

output "secrets_manager_arn" {
  description = "Secrets Manager ARN for API key"
  value       = aws_secretsmanager_secret.alpha_vantage_api_key.arn
  sensitive   = true
}

output "cost_alarm_name" {
  description = "CloudWatch cost alarm name"
  value       = aws_cloudwatch_metric_alarm.cost_alarm.alarm_name
}

output "api_gateway_url" {
  description = "URL of the Stock Recommendations API"
  value       = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations"
}

output "api_endpoints" {
  description = "Available API endpoints"
  value = {
    get_all_recommendations    = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations"
    get_recommendation         = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations/{symbol}"
    ai_analytics_dashboard     = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/analytics/dashboard"
    ai_analytics_detailed      = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/analytics/detailed"
    ai_analytics_history       = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/analytics/history"
    dual_predictions_analytics = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/dual-predictions/analytics"
  }
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.domain_name != "" ? "https://${var.domain_name}/v1" : "Not configured - use api_gateway_url"
}

output "custom_domain_endpoints" {
  description = "Custom domain API endpoints (if configured)"
  value = var.domain_name != "" ? {
    get_all_recommendations = "https://${var.domain_name}/v1/recommendations"
    get_recommendation      = "https://${var.domain_name}/v1/recommendations/{symbol}"
    } : {
    get_all_recommendations = "Not configured - custom domain not set"
    get_recommendation      = "Not configured - custom domain not set"
  }
}

output "domain_configuration_status" {
  description = "Status of custom domain configuration"
  value = var.domain_name != "" ? (
    var.certificate_arn != "" ? (
      var.hosted_zone_id != "" ?
      "✅ Fully configured with custom domain, SSL certificate, and DNS" :
      "⚠️  Partially configured - missing Route 53 hosted zone ID"
    ) : "⚠️  Partially configured - missing SSL certificate ARN"
  ) : "❌ Not configured - using default API Gateway URL"
}

output "api_url_distribution_methods" {
  description = "Methods for dependent applications to get current API URL"
  value = {
    ssm_parameter = {
      name        = aws_ssm_parameter.api_gateway_url.name
      description = "Use AWS CLI: aws ssm get-parameter --name '${aws_ssm_parameter.api_gateway_url.name}' --query 'Parameter.Value' --output text"
    }
    s3_config_file = {
      bucket      = aws_s3_bucket.api_config.bucket
      key         = aws_s3_object.api_config_json.key
      url         = "s3://${aws_s3_bucket.api_config.bucket}/${aws_s3_object.api_config_json.key}"
      description = "Use AWS CLI: aws s3 cp s3://${aws_s3_bucket.api_config.bucket}/${aws_s3_object.api_config_json.key} - | jq -r '.api_gateway_url'"
    }
    terraform_output = {
      description = "Use Terraform: terraform output -raw api_gateway_url"
    }
  }
}

output "ssm_parameter_name" {
  description = "SSM Parameter name containing the API Gateway URL"
  value       = aws_ssm_parameter.api_gateway_url.name
}

output "api_config_s3_location" {
  description = "S3 location of API configuration JSON"
  value       = "s3://${aws_s3_bucket.api_config.bucket}/${aws_s3_object.api_config_json.key}"
}

# Enhanced outputs for tier management
output "deployment_tier" {
  description = "Current deployment tier based on instance sizes"
  value = var.sagemaker_instance_type == "ml.t2.small" ? "current" : (
    var.sagemaker_instance_type == "ml.m5.large" ? "tier1" : (
      var.sagemaker_instance_type == "ml.m5.xlarge" ? "tier2" : "tier3"
    )
  )
  sensitive = false
}

output "premium_api_enabled" {
  description = "Whether premium Alpha Vantage API is enabled"
  value       = var.use_premium_api_key
  sensitive   = false
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost based on deployment tier"
  value = var.sagemaker_instance_type == "ml.t2.small" ? "$45" : (
    var.sagemaker_instance_type == "ml.m5.large" ? "$245" : (
      var.sagemaker_instance_type == "ml.m5.xlarge" ? "$985" : "$3200"
    )
  )
  sensitive = false
}

output "stock_coverage" {
  description = "Estimated number of stocks covered"
  value       = var.use_premium_api_key && var.lambda_memory_size > 512 ? "150+ stocks across all major sectors" : "50+ popular stocks"
  sensitive   = false
}

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    tier = var.sagemaker_instance_type == "ml.t2.small" ? "current" : (
      var.sagemaker_instance_type == "ml.m5.large" ? "tier1" : (
        var.sagemaker_instance_type == "ml.m5.xlarge" ? "tier2" : "tier3"
      )
    )
    lambda_memory           = "${var.lambda_memory_size}MB"
    sagemaker_instances     = "${var.sagemaker_instance_count}x ${var.sagemaker_instance_type}"
    database_class          = var.db_instance_class
    cache_node_type         = var.valkey_node_type
    cache_clusters          = var.valkey_num_cache_clusters
    premium_api             = var.use_premium_api_key
    provisioned_concurrency = var.enable_lambda_provisioned_concurrency
    estimated_cost = var.sagemaker_instance_type == "ml.t2.small" ? "$45/month" : (
      var.sagemaker_instance_type == "ml.m5.large" ? "$245/month" : (
        var.sagemaker_instance_type == "ml.m5.xlarge" ? "$985/month" : "$3200/month"
      )
    )
  }
}

# Swagger UI and API Documentation outputs
output "swagger_ui_url" {
  description = "URL to access Swagger UI documentation"
  value       = "http://${aws_s3_bucket_website_configuration.swagger_ui_website.website_endpoint}"
}

output "swagger_ui_cloudfront_url" {
  description = "CloudFront URL for Swagger UI (if enabled)"
  value       = var.enable_cloudfront_for_docs ? "https://${aws_cloudfront_distribution.swagger_ui_distribution[0].domain_name}" : "CloudFront not enabled"
}

output "api_documentation_urls" {
  description = "All available URLs for API documentation"
  value = {
    s3_website     = "http://${aws_s3_bucket_website_configuration.swagger_ui_website.website_endpoint}"
    swagger_spec   = "http://${aws_s3_bucket_website_configuration.swagger_ui_website.website_endpoint}/swagger.yaml"
    cloudfront_url = var.enable_cloudfront_for_docs ? "https://${aws_cloudfront_distribution.swagger_ui_distribution[0].domain_name}" : "Not enabled"
  }
}

output "documentation_bucket" {
  description = "S3 bucket hosting the API documentation"
  value       = aws_s3_bucket.swagger_ui.bucket
}

output "api_cloudfront_url" {
  description = "CloudFront URL for API caching (if enabled)"
  value       = var.enable_cloudfront_for_api ? "https://${aws_cloudfront_distribution.api_distribution[0].domain_name}" : "CloudFront not enabled"
}