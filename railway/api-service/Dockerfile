# Multi-stage build for Stock Analytics Engine API Service
# Combines multiple Lambda functions into a unified REST API for Railway deployment

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY lambda_requirements/requirements_pandas.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    boto3==1.34.0 \
    python-dotenv==1.0.0

# Copy shared utilities first (for better caching)
COPY lambda_functions/shared /app/lambda_functions/shared

# Copy Lambda function code
COPY lambda_functions/stock_recommendations_api.py /app/lambda_functions/
COPY lambda_functions/dual_prediction_reporting_api.py /app/lambda_functions/
COPY lambda_functions/custom_stock_request_api.py /app/lambda_functions/
COPY lambda_functions/ml_inference_native.py /app/lambda_functions/
COPY lambda_functions/price_prediction_model.py /app/lambda_functions/
COPY lambda_functions/time_to_hit_predictor_slim.py /app/lambda_functions/

# Copy Flask wrapper
COPY railway/api-service/app.py /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Railway
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run with Gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app
