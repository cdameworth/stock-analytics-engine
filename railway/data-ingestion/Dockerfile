# Data Ingestion Worker Service - Railway-native
# No AWS dependencies - uses PostgreSQL for all data storage

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

# Install Python dependencies (no boto3 needed for Railway-native)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    schedule==1.2.0 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.0 \
    pytz==2024.1

# Copy shared utilities (database, error handling)
COPY lambda_functions/shared /app/lambda_functions/shared

# Copy Railway-native services
COPY railway/services /app/services

# Copy worker
COPY railway/data-ingestion/worker.py /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; print('healthy')" || exit 1

# Run worker
CMD ["python", "worker.py"]
