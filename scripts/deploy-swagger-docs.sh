#!/bin/bash
# Deploy Swagger API Documentation for Stock Analytics Engine

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-stock-analytics-admin}"
TERRAFORM_DIR="$(dirname "$0")/../infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform first."
        exit 1
    fi

    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi

    # Check AWS profile
    if ! aws sts get-caller-identity --profile "${AWS_PROFILE}" &> /dev/null; then
        print_error "AWS profile '${AWS_PROFILE}' is not configured or invalid."
        print_status "Please configure your AWS credentials with: aws configure --profile ${AWS_PROFILE}"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to validate Swagger spec
validate_swagger_spec() {
    print_status "Validating Swagger specification..."

    local swagger_file="${TERRAFORM_DIR}/../docs/swagger.yaml"

    if [[ ! -f "$swagger_file" ]]; then
        print_error "Swagger specification file not found: $swagger_file"
        exit 1
    fi

    # Basic YAML syntax validation
    if command -v python3 &> /dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$swagger_file'))" 2>/dev/null || {
            print_error "Invalid YAML syntax in swagger.yaml"
            exit 1
        }
    fi

    print_success "Swagger specification is valid"
}

# Function to initialize and plan Terraform
terraform_plan() {
    print_status "Initializing Terraform..."
    cd "$TERRAFORM_DIR"

    export AWS_PROFILE="${AWS_PROFILE}"

    # Initialize Terraform
    terraform init -upgrade

    print_status "Planning Terraform deployment for Swagger UI..."
    terraform plan -target="aws_s3_bucket.swagger_ui" \
                   -target="aws_s3_bucket_website_configuration.swagger_ui_website" \
                   -target="aws_s3_bucket_public_access_block.swagger_ui_pab" \
                   -target="aws_s3_bucket_policy.swagger_ui_policy" \
                   -target="aws_s3_object.swagger_spec" \
                   -target="aws_s3_object.swagger_ui_html" \
                   -target="aws_s3_object.swagger_ui_error" \
                   -out="swagger-docs.tfplan"
}

# Function to apply Terraform changes
terraform_apply() {
    print_status "Deploying Swagger UI infrastructure..."
    cd "$TERRAFORM_DIR"

    export AWS_PROFILE="${AWS_PROFILE}"

    terraform apply -auto-approve "swagger-docs.tfplan"

    print_success "Swagger UI infrastructure deployed successfully"
}

# Function to get deployment outputs
get_deployment_info() {
    print_status "Retrieving deployment information..."
    cd "$TERRAFORM_DIR"

    export AWS_PROFILE="${AWS_PROFILE}"

    echo ""
    echo "=================================="
    echo "🚀 SWAGGER UI DEPLOYMENT COMPLETE"
    echo "=================================="
    echo ""

    # Get Swagger UI URL
    local swagger_url
    swagger_url=$(terraform output -raw swagger_ui_url 2>/dev/null || echo "Not available")

    if [[ "$swagger_url" != "Not available" ]]; then
        print_success "Swagger UI URL: $swagger_url"
        echo ""
        echo "📖 API Documentation is now available at:"
        echo "   $swagger_url"
        echo ""
    else
        print_warning "Could not retrieve Swagger UI URL from Terraform outputs"
    fi

    # Get API endpoints
    echo "🔗 API Endpoints documented:"
    echo "   GET /recommendations - Get all stock recommendations"
    echo "   GET /recommendations/{symbol} - Get specific stock recommendation"
    echo "   GET /analytics/dashboard - Get performance dashboard"
    echo "   GET /dual-predictions/analytics - Get dual prediction analytics"
    echo ""

    # Get API Gateway URL
    local api_gateway_url
    api_gateway_url=$(terraform output -raw api_gateway_url 2>/dev/null || echo "Not available")
    if [[ "$api_gateway_url" != "Not available" ]]; then
        echo "🌐 API Base URL: ${api_gateway_url%/recommendations}"
        echo ""
    fi

    # Get API key info
    local api_key_value
    api_key_value=$(terraform output -raw api_key_value 2>/dev/null || echo "Not available")
    if [[ "$api_key_value" != "Not available" ]]; then
        echo "🔑 API Key (keep this secure): $api_key_value"
        echo ""
        echo "📋 Example API Call:"
        echo "   curl -H 'x-api-key: $api_key_value' \\"
        echo "        '${api_gateway_url%/recommendations}/recommendations'"
        echo ""
    fi

    echo "💡 Next Steps:"
    echo "   1. Visit the Swagger UI URL to explore the API documentation"
    echo "   2. Use the API key in the 'x-api-key' header for authentication"
    echo "   3. Test the endpoints using the 'Try it out' feature in Swagger UI"
    echo ""
}

# Function to update existing infrastructure
update_existing() {
    print_status "Updating Swagger documentation in existing deployment..."
    cd "$TERRAFORM_DIR"

    export AWS_PROFILE="${AWS_PROFILE}"

    # Only update the S3 objects (documentation files)
    terraform apply -target="aws_s3_object.swagger_spec" \
                   -target="aws_s3_object.swagger_ui_html" \
                   -target="aws_s3_object.swagger_ui_error" \
                   -auto-approve

    print_success "Swagger documentation updated successfully"
}

# Function to enable CloudFront (optional)
enable_cloudfront() {
    print_status "Enabling CloudFront for better performance..."
    cd "$TERRAFORM_DIR"

    export AWS_PROFILE="${AWS_PROFILE}"

    # Check if CloudFront is already enabled
    if terraform output swagger_ui_cloudfront_url 2>/dev/null | grep -v "CloudFront not enabled" > /dev/null; then
        print_warning "CloudFront is already enabled"
        return
    fi

    print_status "This will enable CloudFront distribution for the Swagger UI"
    read -p "Continue? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "CloudFront deployment skipped"
        return
    fi

    # Apply with CloudFront enabled
    terraform apply -target="aws_cloudfront_distribution.swagger_ui_distribution" \
                   -var="enable_cloudfront_for_docs=true" \
                   -auto-approve

    print_success "CloudFront enabled for Swagger UI"

    local cloudfront_url
    cloudfront_url=$(terraform output -raw swagger_ui_cloudfront_url 2>/dev/null)
    if [[ "$cloudfront_url" != "CloudFront not enabled" ]]; then
        echo ""
        print_success "CloudFront URL: $cloudfront_url"
        print_status "Note: CloudFront deployment may take 10-15 minutes to be fully available"
    fi
}

# Function to clean up plan files
cleanup() {
    cd "$TERRAFORM_DIR"
    rm -f swagger-docs.tfplan
}

# Main execution
main() {
    echo "================================================"
    echo "🚀 Stock Analytics Engine - Swagger Docs Deploy"
    echo "================================================"
    echo ""

    local action="${1:-deploy}"

    case "$action" in
        "deploy")
            check_prerequisites
            validate_swagger_spec
            terraform_plan
            terraform_apply
            get_deployment_info
            cleanup
            ;;
        "update")
            check_prerequisites
            validate_swagger_spec
            update_existing
            get_deployment_info
            ;;
        "cloudfront")
            check_prerequisites
            enable_cloudfront
            get_deployment_info
            ;;
        "info")
            get_deployment_info
            ;;
        *)
            echo "Usage: $0 [deploy|update|cloudfront|info]"
            echo ""
            echo "Commands:"
            echo "  deploy     - Deploy complete Swagger UI infrastructure (default)"
            echo "  update     - Update only the documentation files"
            echo "  cloudfront - Enable CloudFront for better performance"
            echo "  info       - Show current deployment information"
            echo ""
            echo "Environment Variables:"
            echo "  AWS_PROFILE - AWS CLI profile to use (default: stock-analytics-admin)"
            echo ""
            exit 1
            ;;
    esac
}

# Trap to cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"