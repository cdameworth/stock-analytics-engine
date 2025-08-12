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
  value       = aws_elasticache_replication_group.stock_analytics_valkey_lowcost.primary_endpoint_address
  sensitive   = true
}

output "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  value       = aws_sagemaker_endpoint.stock_prediction_endpoint_lowcost.name
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
  value       = aws_lambda_function.ml_model_inference_lowcost.function_name
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
    get_all_recommendations = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations"
    get_recommendation      = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.stock_recommendations_api_stage.stage_name}/recommendations/{symbol}"
  }
}