# Stock Analytics Engine API Service - Railway-native
# No AWS dependencies - uses PostgreSQL for all data storage

FROM python:3.11-slim

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

# Install Python dependencies (no boto3 needed for Railway-native)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.0

# Copy shared utilities (database, error handling, accuracy tracking)
COPY lambda_functions/shared /app/lambda_functions/shared

# Copy Railway-native services
COPY railway/services /app/services

# Copy Flask application
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
