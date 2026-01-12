# Data Ingestion Worker Service for Railway
# Runs scheduled data collection from Alpha Vantage API

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY lambda_requirements/requirements_pandas.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    schedule==1.2.0 \
    boto3==1.34.0 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0

# Copy shared utilities
COPY lambda_functions/shared /app/lambda_functions/shared

# Copy Lambda function code
COPY lambda_functions/stock_data_ingestion.py /app/lambda_functions/

# Copy worker wrapper
COPY railway/data-ingestion/worker.py /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; print('healthy')" || exit 1

# Run worker
CMD ["python", "worker.py"]
