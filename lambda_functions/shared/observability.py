"""
Observability and monitoring utilities for Stock Analytics Engine.
Provides comprehensive monitoring, tracing, and performance measurement capabilities.
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from .lambda_utils import AWSClients, MetricsHelper
from .config import get_config, FeatureFlags
from .error_handling import StructuredLogger

config = get_config()


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str]
    context: Dict[str, Any]


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    tags: Dict[str, str]
    logs: List[Dict[str, Any]]
    status: str  # 'success', 'error', 'timeout'


class PerformanceTracker:
    """Track and measure performance metrics."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics_helper = MetricsHelper(f"StockAnalytics/{service_name}")
        self.logger = StructuredLogger(f"{service_name}.performance")
        self.active_spans: Dict[str, TraceSpan] = {}
    
    @contextmanager
    def measure_operation(self, operation_name: str, 
                         dimensions: Dict[str, str] = None,
                         log_result: bool = True):
        """
        Context manager to measure operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            dimensions: Additional dimensions for metrics
            log_result: Whether to log the performance result
        """
        start_time = time.time()
        span_id = str(uuid.uuid4())
        
        # Create trace span
        span = TraceSpan(
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            tags=dimensions or {},
            logs=[],
            status='in_progress'
        )
        
        self.active_spans[span_id] = span
        
        try:
            yield span
            span.status = 'success'
            
        except Exception as e:
            span.status = 'error'
            span.logs.append({
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'error',
                'message': str(e)
            })
            raise
            
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            span.end_time = datetime.utcnow()
            span.duration_ms = duration_ms
            
            # Send performance metrics
            if FeatureFlags.is_metrics_enabled():
                self._send_performance_metrics(operation_name, duration_ms, dimensions, span.status)
            
            # Log performance if requested
            if log_result:
                self._log_performance(operation_name, duration_ms, span.status, dimensions)
            
            # Clean up span
            self.active_spans.pop(span_id, None)
    
    def record_metric(self, name: str, value: float, unit: str = 'Count',
                     dimensions: Dict[str, str] = None, 
                     context: Dict[str, Any] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            dimensions: Metric dimensions
            context: Additional context
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            dimensions=dimensions or {},
            context=context or {}
        )
        
        if FeatureFlags.is_metrics_enabled():
            self.metrics_helper.put_metric(name, value, unit, dimensions)
        
        # Log metric for debugging
        if config.is_development():
            self.logger.log_debug(f"Metric recorded: {name}={value}", context=asdict(metric))
    
    def _send_performance_metrics(self, operation_name: str, duration_ms: float,
                                 dimensions: Dict[str, str], status: str) -> None:
        """Send performance metrics to CloudWatch."""
        try:
            base_dimensions = {'Operation': operation_name, 'Status': status}
            if dimensions:
                base_dimensions.update(dimensions)
            
            # Duration metric
            self.metrics_helper.put_metric(
                'OperationDuration',
                duration_ms,
                'Milliseconds',
                base_dimensions
            )
            
            # Count metric
            self.metrics_helper.put_metric(
                'OperationCount',
                1,
                'Count',
                base_dimensions
            )
            
            # Success/failure rate
            success_value = 1 if status == 'success' else 0
            self.metrics_helper.put_metric(
                'OperationSuccess',
                success_value,
                'Count',
                base_dimensions
            )
            
        except Exception as e:
            self.logger.log_warning(f"Failed to send performance metrics: {str(e)}")
    
    def _log_performance(self, operation_name: str, duration_ms: float,
                        status: str, dimensions: Dict[str, str]) -> None:
        """Log performance information."""
        context = {
            'operation': operation_name,
            'duration_ms': round(duration_ms, 2),
            'status': status,
            'dimensions': dimensions or {}
        }
        
        if status == 'success':
            self.logger.log_info(f"Operation completed: {operation_name}", context)
        else:
            self.logger.log_warning(f"Operation failed: {operation_name}", context)


class BusinessMetricsCollector:
    """Collect business-specific metrics for the stock analytics engine."""
    
    def __init__(self):
        self.metrics_helper = MetricsHelper("StockAnalytics/Business")
        self.logger = StructuredLogger("business_metrics")
    
    def record_prediction_accuracy(self, symbol: str, prediction_type: str,
                                  accuracy: float, confidence: float) -> None:
        """Record prediction accuracy metrics."""
        dimensions = {
            'Symbol': symbol,
            'PredictionType': prediction_type
        }
        
        self.metrics_helper.put_metric('PredictionAccuracy', accuracy, 'Percent', dimensions)
        self.metrics_helper.put_metric('PredictionConfidence', confidence, 'None', dimensions)
    
    def record_recommendation_performance(self, symbol: str, recommendation: str,
                                        actual_return: float, predicted_return: float) -> None:
        """Record recommendation performance metrics."""
        dimensions = {
            'Symbol': symbol,
            'Recommendation': recommendation
        }
        
        self.metrics_helper.put_metric('ActualReturn', actual_return, 'Percent', dimensions)
        self.metrics_helper.put_metric('PredictedReturn', predicted_return, 'Percent', dimensions)
        
        # Calculate prediction error
        error = abs(actual_return - predicted_return)
        self.metrics_helper.put_metric('PredictionError', error, 'Percent', dimensions)
    
    def record_api_usage(self, endpoint: str, response_time_ms: float,
                        status_code: int, user_type: str = 'unknown') -> None:
        """Record API usage metrics."""
        dimensions = {
            'Endpoint': endpoint,
            'StatusCode': str(status_code),
            'UserType': user_type
        }
        
        self.metrics_helper.put_metric('APIResponseTime', response_time_ms, 'Milliseconds', dimensions)
        self.metrics_helper.put_metric('APIRequestCount', 1, 'Count', dimensions)
        
        # Success rate
        success = 1 if 200 <= status_code < 300 else 0
        self.metrics_helper.put_metric('APISuccessRate', success, 'Count', dimensions)
    
    def record_data_quality(self, source: str, completeness: float,
                           freshness_minutes: float, accuracy: float) -> None:
        """Record data quality metrics."""
        dimensions = {'DataSource': source}
        
        self.metrics_helper.put_metric('DataCompleteness', completeness, 'Percent', dimensions)
        self.metrics_helper.put_metric('DataFreshness', freshness_minutes, 'Minutes', dimensions)
        self.metrics_helper.put_metric('DataAccuracy', accuracy, 'Percent', dimensions)


# Decorators for automatic monitoring
def monitor_performance(operation_name: str = None, 
                       service_name: str = "Unknown",
                       dimensions: Dict[str, str] = None):
    """
    Decorator to automatically monitor function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        service_name: Name of the service
        dimensions: Additional dimensions for metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = PerformanceTracker(service_name)
            op_name = operation_name or func.__name__
            
            with tracker.measure_operation(op_name, dimensions):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_business_metrics(metric_type: str = "general"):
    """
    Decorator to automatically collect business metrics.
    
    Args:
        metric_type: Type of business metric to collect
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = BusinessMetricsCollector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Extract metrics from result if it's a dict
                if isinstance(result, dict) and metric_type == "prediction":
                    symbol = result.get('symbol', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    prediction_type = result.get('prediction_type', 'unknown')
                    
                    # Record prediction metrics
                    collector.record_prediction_accuracy(
                        symbol, prediction_type, 1.0, confidence  # Assume success for now
                    )
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration_ms = (time.time() - start_time) * 1000
                collector.logger.log_error(e, context={
                    'function': func.__name__,
                    'duration_ms': duration_ms,
                    'metric_type': metric_type
                })
                raise
        
        return wrapper
    return decorator


class HealthChecker:
    """Health check utilities for service monitoring."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = StructuredLogger(f"{service_name}.health")
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool],
                      timeout_seconds: int = 5) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            timeout_seconds: Timeout for the check
        """
        def timed_check():
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Health check '{name}' timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                return check_func()
            finally:
                signal.alarm(0)
        
        self.checks[name] = timed_check
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Dict containing health check results
        """
        results = {
            'service': self.service_name,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration_ms = (time.time() - start_time) * 1000
                
                results['checks'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'duration_ms': round(duration_ms, 2),
                    'message': 'OK' if is_healthy else 'Check failed'
                }
                
                if not is_healthy:
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'duration_ms': 0,
                    'message': str(e)
                }
                results['overall_status'] = 'unhealthy'
                
                self.logger.log_error(e, context={'health_check': name})
        
        return results


# Common health check functions
def check_database_connection(table_name: str) -> bool:
    """Check if DynamoDB table is accessible."""
    try:
        dynamodb = AWSClients.get_resource('dynamodb')
        table = dynamodb.Table(table_name)
        table.load()
        return True
    except Exception:
        return False


def check_s3_bucket_access(bucket_name: str) -> bool:
    """Check if S3 bucket is accessible."""
    try:
        s3 = AWSClients.get_client('s3')
        s3.head_bucket(Bucket=bucket_name)
        return True
    except Exception:
        return False


def check_external_api(api_url: str, timeout: int = 5) -> bool:
    """Check if external API is accessible."""
    try:
        import requests
        response = requests.get(api_url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False
