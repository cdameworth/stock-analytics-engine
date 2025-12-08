# Model Tuning Worker Service for Railway
# Runs scheduled ML model optimization and hyperparameter tuning

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

# Install Python dependencies with ML libraries
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    schedule==1.2.0 \
    boto3==1.34.0 \
    python-dotenv==1.0.0 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    lightgbm==4.1.0

# Copy shared utilities
COPY lambda_functions/shared /app/lambda_functions/shared

# Copy Lambda function code
COPY lambda_functions/price_model_tuning.py /app/lambda_functions/
COPY lambda_functions/time_model_tuning.py /app/lambda_functions/
COPY lambda_functions/model_tuning_reporter.py /app/lambda_functions/
COPY lambda_functions/dual_accuracy_tracker.py /app/lambda_functions/

# Copy worker wrapper
COPY railway/model-tuning/worker.py /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=120s --timeout=20s --start-period=60s --retries=3 \
    CMD python -c "import os; print('healthy')" || exit 1

# Run worker
CMD ["python", "worker.py"]
