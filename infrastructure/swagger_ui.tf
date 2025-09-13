# Swagger UI hosting infrastructure for Stock Analytics Engine API documentation

# S3 bucket for hosting Swagger UI static files
resource "aws_s3_bucket" "swagger_ui" {
  bucket = "stock-analytics-swagger-ui-${random_id.bucket_suffix.hex}"

  tags = merge(
    {
      Name    = "stock-analytics-swagger-ui"
      Purpose = "API documentation hosting"
    },
    var.additional_tags
  )
}

# Enable static website hosting on S3 bucket
resource "aws_s3_bucket_website_configuration" "swagger_ui_website" {
  bucket = aws_s3_bucket.swagger_ui.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

# S3 bucket public access configuration for website hosting
resource "aws_s3_bucket_public_access_block" "swagger_ui_pab" {
  bucket = aws_s3_bucket.swagger_ui.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# S3 bucket policy to allow public read access for website hosting
resource "aws_s3_bucket_policy" "swagger_ui_policy" {
  bucket = aws_s3_bucket.swagger_ui.id
  depends_on = [aws_s3_bucket_public_access_block.swagger_ui_pab]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.swagger_ui.arn}/*"
      }
    ]
  })
}

# Upload OpenAPI/Swagger specification
resource "aws_s3_object" "swagger_spec" {
  bucket       = aws_s3_bucket.swagger_ui.id
  key          = "swagger.yaml"
  source       = "${path.module}/../docs/swagger.yaml"
  content_type = "application/yaml"
  etag         = filemd5("${path.module}/../docs/swagger.yaml")

  tags = {
    Name = "swagger-spec"
    Type = "api-documentation"
  }
}

# Upload Swagger UI HTML file
resource "aws_s3_object" "swagger_ui_html" {
  bucket       = aws_s3_bucket.swagger_ui.id
  key          = "index.html"
  content_type = "text/html"

  content = templatefile("${path.module}/swagger-ui-template.html", {
    api_title       = "Stock Analytics Engine API"
    swagger_spec_url = "swagger.yaml"
    api_base_url    = "https://${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/prod"
  })

  tags = {
    Name = "swagger-ui-html"
    Type = "api-documentation"
  }
}

# Upload error page
resource "aws_s3_object" "swagger_ui_error" {
  bucket       = aws_s3_bucket.swagger_ui.id
  key          = "error.html"
  content_type = "text/html"

  content = <<-EOF
<!DOCTYPE html>
<html>
<head>
    <title>Error - Stock Analytics API Docs</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .error-container { max-width: 600px; margin: 0 auto; text-align: center; }
        .error-title { color: #d73502; font-size: 24px; margin-bottom: 20px; }
        .error-message { color: #666; font-size: 16px; }
        .back-link { margin-top: 30px; }
        .back-link a { color: #1f77d0; text-decoration: none; }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-title">Page Not Found</div>
        <div class="error-message">The requested page could not be found.</div>
        <div class="back-link">
            <a href="index.html">← Back to API Documentation</a>
        </div>
    </div>
</body>
</html>
EOF

  tags = {
    Name = "swagger-ui-error"
    Type = "api-documentation"
  }
}

# CloudFront distribution for Swagger UI (optional, for better performance)
resource "aws_cloudfront_distribution" "swagger_ui_distribution" {
  count = var.enable_cloudfront_for_docs ? 1 : 0

  origin {
    domain_name = aws_s3_bucket_website_configuration.swagger_ui_website.website_endpoint
    origin_id   = "swagger-ui-s3-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled             = true
  default_root_object = "index.html"
  comment             = "Stock Analytics API Documentation"

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "swagger-ui-s3-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 300   # 5 minutes
    max_ttl     = 86400 # 24 hours
  }

  # Cache behavior for YAML files with shorter TTL for updates
  ordered_cache_behavior {
    path_pattern           = "*.yaml"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "swagger-ui-s3-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 60    # 1 minute for spec updates
    max_ttl     = 300   # 5 minutes
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = merge(
    {
      Name    = "stock-analytics-swagger-ui-cdn"
      Purpose = "API documentation delivery"
    },
    var.additional_tags
  )
}

# CloudFront distribution for API Gateway (optional, for better API performance)
resource "aws_cloudfront_distribution" "api_distribution" {
  count = var.enable_cloudfront_for_api ? 1 : 0

  origin {
    domain_name = replace(aws_api_gateway_rest_api.stock_recommendations_api.execution_arn, "arn:aws:execute-api:", "")
    domain_name = "${aws_api_gateway_rest_api.stock_recommendations_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com"
    origin_id   = "api-gateway-origin"
    origin_path = "/prod"

    custom_origin_config {
      http_port              = 443
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled = true
  comment = "Stock Analytics API Gateway Distribution"

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "api-gateway-origin"
    compress               = true
    viewer_protocol_policy = "https-only"

    forwarded_values {
      query_string = true
      headers      = ["x-api-key", "Authorization"]
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 0     # No caching for API responses by default
    max_ttl     = 86400
  }

  # Cache recommendations endpoint for short periods
  ordered_cache_behavior {
    path_pattern           = "/recommendations*"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "api-gateway-origin"
    compress               = true
    viewer_protocol_policy = "https-only"

    forwarded_values {
      query_string = true
      headers      = ["x-api-key"]
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 60    # 1 minute cache for recommendations
    max_ttl     = 300   # 5 minutes max
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = merge(
    {
      Name    = "stock-analytics-api-cdn"
      Purpose = "API performance acceleration"
    },
    var.additional_tags
  )
}