provider "aws" {
  region  = var.aws_region
  profile = "stock-analytics-admin"
}

# Backend configuration (uncomment and modify as needed)
# terraform {
#   backend "s3" {
#     bucket = "your-terraform-state-bucket"
#     key    = "stock-analytics/terraform.tfstate"
#     region = "us-east-1"
#   }
# }