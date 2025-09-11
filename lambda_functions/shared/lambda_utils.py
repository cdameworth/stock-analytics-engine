"""
Shared utilities for Lambda functions in the Stock Analytics Engine.
Provides common patterns for error handling, logging, response formatting, and AWS service interactions.
"""

import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import boto3
from botocore.exceptions import ClientError, BotoCoreError


# Configure structured logging
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up structured logging for Lambda functions.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter for structured logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Add console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Common response patterns
class LambdaResponse:
    """Standardized Lambda response builder."""
    
    @staticmethod
    def success(data: Any, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create a successful Lambda response."""
        return {
            'statusCode': status_code,
            'headers': headers or LambdaResponse.get_cors_headers(),
            'body': json.dumps({
                'success': True,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }, default=decimal_to_float)
        }
    
    @staticmethod
    def error(message: str, status_code: int = 500, error_code: Optional[str] = None, 
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create an error Lambda response."""
        error_data = {
            'success': False,
            'error': {
                'message': message,
                'code': error_code or 'INTERNAL_ERROR',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return {
            'statusCode': status_code,
            'headers': headers or LambdaResponse.get_cors_headers(),
            'body': json.dumps(error_data)
        }
    
    @staticmethod
    def get_cors_headers() -> Dict[str, str]:
        """Get standard CORS headers."""
        return {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
            'Content-Type': 'application/json'
        }


# Decimal conversion utility
def decimal_to_float(obj: Any) -> Any:
    """Convert Decimal objects to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Environment variable helpers
class EnvConfig:
    """Environment variable configuration helper."""
    
    @staticmethod
    def get_required(key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    @staticmethod
    def get_optional(key: str, default: str = "") -> str:
        """Get optional environment variable with default."""
        return os.environ.get(key, default)
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer environment variable with default."""
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable with default."""
        return os.environ.get(key, str(default)).lower() in ('true', '1', 'yes', 'on')


# AWS service helpers
class AWSClients:
    """Centralized AWS client management with error handling."""
    
    _clients = {}
    
    @classmethod
    def get_client(cls, service_name: str, region: Optional[str] = None) -> Any:
        """Get AWS client with caching and error handling."""
        cache_key = f"{service_name}_{region or 'default'}"

        if cache_key not in cls._clients:
            try:
                # Use default region if none specified
                client_region = region or os.environ.get('AWS_REGION', 'us-east-1')
                cls._clients[cache_key] = boto3.client(service_name, region_name=client_region)
            except Exception as e:
                # In test environments, return a mock client
                if os.environ.get('ENVIRONMENT') == 'test':
                    from unittest.mock import Mock
                    cls._clients[cache_key] = Mock()
                else:
                    raise RuntimeError(f"Failed to create {service_name} client: {str(e)}")

        return cls._clients[cache_key]
    
    @classmethod
    def get_resource(cls, service_name: str, region: Optional[str] = None) -> Any:
        """Get AWS resource with caching and error handling."""
        cache_key = f"{service_name}_resource_{region or 'default'}"

        if cache_key not in cls._clients:
            try:
                # Use default region if none specified
                client_region = region or os.environ.get('AWS_REGION', 'us-east-1')
                cls._clients[cache_key] = boto3.resource(service_name, region_name=client_region)
            except Exception as e:
                # In test environments, return a mock resource
                if os.environ.get('ENVIRONMENT') == 'test':
                    from unittest.mock import Mock
                    cls._clients[cache_key] = Mock()
                else:
                    raise RuntimeError(f"Failed to create {service_name} resource: {str(e)}")

        return cls._clients[cache_key]


# Error handling decorator
def handle_lambda_errors(logger: logging.Logger):
    """Decorator for consistent Lambda error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
            start_time = time.time()
            
            try:
                logger.info(f"Starting {func.__name__} with event: {json.dumps(event, default=str)}")
                result = func(event, context)
                
                duration = round(time.time() - start_time, 3)
                logger.info(f"Completed {func.__name__} in {duration}s")
                
                return result
                
            except ValueError as e:
                logger.error(f"Validation error in {func.__name__}: {str(e)}")
                return LambdaResponse.error(str(e), 400, "VALIDATION_ERROR")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.error(f"AWS error in {func.__name__}: {error_code} - {str(e)}")
                return LambdaResponse.error(f"AWS service error: {error_code}", 500, "AWS_ERROR")
                
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
                return LambdaResponse.error("Internal server error", 500, "INTERNAL_ERROR")
        
        return wrapper
    return decorator


# CloudWatch metrics helper
class MetricsHelper:
    """Helper for sending custom CloudWatch metrics."""
    
    def __init__(self, namespace: str = "StockAnalytics"):
        self.cloudwatch = AWSClients.get_client('cloudwatch')
        self.namespace = namespace
    
    def put_metric(self, metric_name: str, value: Union[int, float], 
                   unit: str = 'Count', dimensions: Optional[Dict[str, str]] = None) -> None:
        """Send a custom metric to CloudWatch."""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
        except Exception as e:
            # Don't fail the main function for metrics errors
            logging.getLogger(__name__).warning(f"Failed to send metric {metric_name}: {str(e)}")


# DynamoDB helpers
class DynamoDBHelper:
    """Helper for common DynamoDB operations."""
    
    def __init__(self, table_name: str):
        self.table = AWSClients.get_resource('dynamodb').Table(table_name)
    
    def put_item_safe(self, item: Dict[str, Any]) -> bool:
        """Safely put item to DynamoDB with error handling."""
        try:
            # Convert any Decimal values
            clean_item = json.loads(json.dumps(item, default=decimal_to_float))
            self.table.put_item(Item=clean_item)
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to put item to DynamoDB: {str(e)}")
            return False
    
    def get_item_safe(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely get item from DynamoDB with error handling."""
        try:
            response = self.table.get_item(Key=key)
            return response.get('Item')
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to get item from DynamoDB: {str(e)}")
            return None


# Input validation helpers
class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate that all required fields are present."""
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate and normalize stock symbol."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        normalized = symbol.upper().strip()
        if not normalized.isalpha() or len(normalized) > 10:
            raise ValueError("Invalid symbol format")
        
        return normalized
    
    @staticmethod
    def validate_positive_number(value: Any, field_name: str) -> float:
        """Validate that a value is a positive number."""
        try:
            num_value = float(value)
            if num_value <= 0:
                raise ValueError(f"{field_name} must be positive")
            return num_value
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a valid positive number")
