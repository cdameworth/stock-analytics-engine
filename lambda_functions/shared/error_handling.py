"""
Enhanced error handling and logging framework for Stock Analytics Engine.
Provides structured error handling, custom exceptions, and comprehensive logging.
"""

import json
import logging
import traceback
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union
from functools import wraps
from enum import Enum

from .lambda_utils import LambdaResponse, MetricsHelper
from .config import get_config, FeatureFlags

config = get_config()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    CACHE = "cache"
    ML_MODEL = "ml_model"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"


# Custom Exception Classes
class StockAnalyticsError(Exception):
    """Base exception for Stock Analytics Engine."""
    
    def __init__(self, message: str, error_code: str = None, 
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/response."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'timestamp': self.timestamp
        }


class ValidationError(StockAnalyticsError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        context = {}
        if field:
            context['field'] = field
        if value is not None:
            context['invalid_value'] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context
        )


class ExternalAPIError(StockAnalyticsError):
    """Raised when external API calls fail."""
    
    def __init__(self, message: str, api_name: str = None, 
                 status_code: int = None, response_body: str = None):
        context = {}
        if api_name:
            context['api_name'] = api_name
        if status_code:
            context['status_code'] = status_code
        if response_body:
            context['response_body'] = response_body[:1000]  # Limit size
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class DatabaseError(StockAnalyticsError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, table_name: str = None, 
                 operation: str = None, key: Dict[str, Any] = None):
        context = {}
        if table_name:
            context['table_name'] = table_name
        if operation:
            context['operation'] = operation
        if key:
            context['key'] = key
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class MLModelError(StockAnalyticsError):
    """Raised when ML model operations fail."""
    
    def __init__(self, message: str, model_name: str = None, 
                 model_version: str = None, input_data: Dict[str, Any] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if model_version:
            context['model_version'] = model_version
        if input_data:
            context['input_data'] = {k: str(v)[:100] for k, v in input_data.items()}
        
        super().__init__(
            message=message,
            error_code="ML_MODEL_ERROR",
            category=ErrorCategory.ML_MODEL,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class ConfigurationError(StockAnalyticsError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None, 
                 expected_type: str = None, actual_value: Any = None):
        context = {}
        if config_key:
            context['config_key'] = config_key
        if expected_type:
            context['expected_type'] = expected_type
        if actual_value is not None:
            context['actual_value'] = str(actual_value)
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            context=context
        )


# Enhanced Logger Class
class StructuredLogger:
    """Enhanced logger with structured logging and error tracking."""
    
    def __init__(self, name: str, level: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, (level or config.get_log_level()).upper()))
        self.metrics_helper = MetricsHelper("StockAnalytics/Errors") if FeatureFlags.is_metrics_enabled() else None
        
        # Configure structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Remove existing handlers to avoid duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_error(self, error: Union[Exception, StockAnalyticsError], 
                  context: Dict[str, Any] = None, 
                  include_traceback: bool = True) -> None:
        """Log error with structured information."""
        error_data = {
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {}
        }
        
        if isinstance(error, StockAnalyticsError):
            error_data.update(error.to_dict())
        
        if include_traceback:
            error_data['traceback'] = traceback.format_exc()
        
        # Log the error
        self.logger.error(json.dumps(error_data, default=str))
        
        # Send metrics if enabled
        if self.metrics_helper:
            self._send_error_metrics(error, error_data)
    
    def log_warning(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log warning with context."""
        warning_data = {
            'message': message,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.warning(json.dumps(warning_data, default=str))
    
    def log_info(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log info with context."""
        info_data = {
            'message': message,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(info_data, default=str))
    
    def log_debug(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log debug with context."""
        if config.is_development():
            debug_data = {
                'message': message,
                'context': context or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            self.logger.debug(json.dumps(debug_data, default=str))
    
    def _send_error_metrics(self, error: Exception, error_data: Dict[str, Any]) -> None:
        """Send error metrics to CloudWatch."""
        try:
            error_type = type(error).__name__
            category = error_data.get('category', 'unknown')
            severity = error_data.get('severity', 'medium')
            
            self.metrics_helper.put_metric(
                'ErrorOccurred',
                1,
                'Count',
                {
                    'ErrorType': error_type,
                    'Category': category,
                    'Severity': severity
                }
            )
        except Exception as e:
            # Don't fail the main function for metrics errors
            self.logger.warning(f"Failed to send error metrics: {str(e)}")


# Error handling decorators
def handle_errors(logger: StructuredLogger, 
                  return_error_response: bool = True,
                  reraise_critical: bool = True):
    """
    Decorator for comprehensive error handling.
    
    Args:
        logger: StructuredLogger instance
        return_error_response: Whether to return Lambda error response
        reraise_critical: Whether to reraise critical errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except ValidationError as e:
                logger.log_error(e, include_traceback=False)
                if return_error_response:
                    return LambdaResponse.error(e.message, 400, e.error_code)
                raise
            
            except ExternalAPIError as e:
                logger.log_error(e)
                if return_error_response:
                    return LambdaResponse.error(
                        "External service temporarily unavailable", 
                        503, 
                        e.error_code
                    )
                raise
            
            except DatabaseError as e:
                logger.log_error(e)
                if return_error_response:
                    return LambdaResponse.error(
                        "Data service temporarily unavailable", 
                        503, 
                        e.error_code
                    )
                raise
            
            except ConfigurationError as e:
                logger.log_error(e)
                if reraise_critical:
                    raise
                if return_error_response:
                    return LambdaResponse.error(
                        "Service configuration error", 
                        500, 
                        e.error_code
                    )
            
            except StockAnalyticsError as e:
                logger.log_error(e)
                if return_error_response:
                    status_code = 500 if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else 400
                    return LambdaResponse.error(e.message, status_code, e.error_code)
                raise
            
            except Exception as e:
                # Handle unexpected errors
                logger.log_error(e, context={'function': func.__name__})
                if return_error_response:
                    return LambdaResponse.error(
                        "An unexpected error occurred", 
                        500, 
                        "INTERNAL_ERROR"
                    )
                raise
        
        return wrapper
    return decorator


def create_error_context(function_name: str, **kwargs) -> Dict[str, Any]:
    """Create error context dictionary."""
    context = {
        'function': function_name,
        'timestamp': datetime.utcnow().isoformat()
    }
    context.update(kwargs)
    return context


def safe_execute(func, *args, default_return=None, logger: StructuredLogger = None, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        logger: Logger instance
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.log_error(e, context={'function': func.__name__})
        return default_return


# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and retry logic."""
    
    @staticmethod
    def retry_with_backoff(func, max_retries: int = 3, 
                          backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,),
                          logger: StructuredLogger = None):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplier
            exceptions: Exceptions to catch and retry
            logger: Logger instance
        """
        import time
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries:
                    if logger:
                        logger.log_error(e, context={'final_attempt': True})
                    raise
                
                wait_time = backoff_factor ** attempt
                if logger:
                    logger.log_warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s",
                        context={'error': str(e), 'attempt': attempt + 1}
                    )
                time.sleep(wait_time)
    
    @staticmethod
    def circuit_breaker(failure_threshold: int = 5, 
                       recovery_timeout: int = 60,
                       logger: StructuredLogger = None):
        """
        Circuit breaker pattern implementation.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            logger: Logger instance
        """
        # This would be implemented with state management
        # For now, it's a placeholder for the pattern
        pass
