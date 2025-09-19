# OpenTelemetry Configuration for AWS Lambda Functions
# SigNoz-compatible configuration with business-aware tracing

# Data source for existing custom OTEL layer (will be created by script)
data "aws_lambda_layer_version" "custom_otel" {
  layer_name = "stock-analytics-otel-python-http"
  # Use the latest version
}

# Official OpenTelemetry Python Lambda layer from AWS
locals {
  # OpenTelemetry Python layer ARN from AWS (replace region dynamically)
  official_otel_layer_arn = "arn:aws:lambda:${data.aws_region.current.name}:184161586896:layer:opentelemetry-python-0_11_0:1"

  # Custom layer ARN (will be available after running create-otel-layer-http.sh)
  custom_otel_layer_arn = data.aws_lambda_layer_version.custom_otel.arn

  # Common OpenTelemetry environment variables for SigNoz
  base_otel_environment = {
    # SigNoz-required configuration
    AWS_LAMBDA_EXEC_WRAPPER      = "/opt/otel-instrument"
    OTEL_EXPORTER_OTLP_ENDPOINT  = "https://${var.signoz_otlp_endpoint}"
    OTEL_EXPORTER_OTLP_HEADERS   = "signoz-ingestion-key=${var.signoz_ingestion_key}"
    OTEL_EXPORTER_OTLP_PROTOCOL  = "grpc"
    OTEL_PROPAGATORS             = "tracecontext,baggage,xray"
    OTEL_TRACES_SAMPLER          = "parentbased_always_off"  # Use our custom business sampler

    # Service identification
    OTEL_SERVICE_NAME            = "stock-analytics"
    OTEL_SERVICE_VERSION         = "1.0.0"

    # Week 3 advanced features
    ENABLE_BUSINESS_TRACING      = "true"
    PERFORMANCE_MONITORING       = var.enable_signoz_integration ? "true" : "false"
    SIGNOZ_INTEGRATION          = var.enable_signoz_integration ? "true" : "false"

    # Sampling and performance
    OTEL_TRACES_SAMPLER_ARG      = var.otel_trace_sampling_ratio
    OTEL_LOG_LEVEL              = var.enable_debug_logging ? "debug" : "info"

    # Resource attributes for better service identification
    OTEL_RESOURCE_ATTRIBUTES     = join(",", [
      "service.name=stock-analytics",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "faas.name={{.FunctionName}}",
      "faas.version={{.Version}}"
    ])
  }

  # Common Lambda layers for all functions
  common_otel_layers = var.enable_signoz_integration ? [
    local.official_otel_layer_arn,
    local.custom_otel_layer_arn
  ] : []

  # X-Ray tracing configuration for Lambda functions
  xray_tracing_config = {
    mode = var.enable_signoz_integration ? "Active" : "PassThrough"
  }
}

# Function-specific OTEL environment variables
locals {
  # Stock data ingestion specific config
  stock_data_ingestion_otel_env = merge(local.base_otel_environment, {
    OTEL_SERVICE_NAME = "stock-data-ingestion"
    OTEL_RESOURCE_ATTRIBUTES = join(",", [
      "service.name=stock-data-ingestion",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "business.domain=data-ingestion",
      "business.criticality=high"
    ])
  })

  # ML model inference specific config
  ml_model_inference_otel_env = merge(local.base_otel_environment, {
    OTEL_SERVICE_NAME = "ml-model-inference"
    OTEL_RESOURCE_ATTRIBUTES = join(",", [
      "service.name=ml-model-inference",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "business.domain=ml-inference",
      "business.criticality=high"
    ])
  })

  # Stock recommendations API specific config
  stock_recommendations_api_otel_env = merge(local.base_otel_environment, {
    OTEL_SERVICE_NAME = "stock-recommendations-api"
    OTEL_RESOURCE_ATTRIBUTES = join(",", [
      "service.name=stock-recommendations-api",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "business.domain=api",
      "business.criticality=critical"
    ])
  })

  # Dual accuracy tracker specific config
  dual_accuracy_tracker_otel_env = merge(local.base_otel_environment, {
    OTEL_SERVICE_NAME = "dual-accuracy-tracker"
    OTEL_RESOURCE_ATTRIBUTES = join(",", [
      "service.name=dual-accuracy-tracker",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "business.domain=analytics",
      "business.criticality=medium"
    ])
  })

  # Dual prediction reporting API specific config
  dual_prediction_reporting_api_otel_env = merge(local.base_otel_environment, {
    OTEL_SERVICE_NAME = "dual-prediction-reporting-api"
    OTEL_RESOURCE_ATTRIBUTES = join(",", [
      "service.name=dual-prediction-reporting-api",
      "service.version=1.0.0",
      "deployment.environment=${terraform.workspace}",
      "cloud.provider=aws",
      "cloud.platform=aws_lambda",
      "business.domain=reporting",
      "business.criticality=medium"
    ])
  })
}

# Output the layer ARNs for reference
output "otel_configuration" {
  description = "OpenTelemetry configuration details"
  value = {
    official_layer_arn = local.official_otel_layer_arn
    custom_layer_arn   = var.enable_signoz_integration ? local.custom_otel_layer_arn : "not_enabled"
    signoz_endpoint    = var.enable_signoz_integration ? var.signoz_otlp_endpoint : "not_configured"
    tracing_enabled    = var.enable_signoz_integration
    debug_logging      = var.enable_debug_logging
    sampling_ratio     = var.otel_trace_sampling_ratio
    business_tracing   = "enabled"
  }
  sensitive = false
}

# Helper function to merge OTEL environment with existing Lambda environment
locals {
  # Helper function to create complete environment for each Lambda function
  get_lambda_environment = {
    stock_data_ingestion       = merge(var.lambda_environment_variables, local.stock_data_ingestion_otel_env)
    ml_model_inference        = merge(var.lambda_environment_variables, local.ml_model_inference_otel_env)
    stock_recommendations_api = merge(var.lambda_environment_variables, local.stock_recommendations_api_otel_env)
    dual_accuracy_tracker     = merge(var.lambda_environment_variables, local.dual_accuracy_tracker_otel_env)
    dual_prediction_reporting_api = merge(var.lambda_environment_variables, local.dual_prediction_reporting_api_otel_env)
  }
}