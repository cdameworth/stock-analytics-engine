# Data Ingestion Worker Service for Railway
# Railway-native implementation using PostgreSQL (no AWS dependencies)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (minimal, no AWS)
RUN pip install --no-cache-dir \
    schedule==1.2.0 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    pytz==2024.1 \
    python-dotenv==1.0.0

# Copy Railway shared utilities
COPY railway/shared /app/shared

# Copy Railway-native worker
COPY railway/data-ingestion/worker_native.py /app/worker.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; print('healthy')" || exit 1

# Run worker
CMD ["python", "worker.py"]
