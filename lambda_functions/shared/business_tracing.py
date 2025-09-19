"""
Business-Aware OpenTelemetry Tracing for Stock Analytics
Provides intelligent sampling and financial domain tracing
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, Callable
from functools import wraps
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision

from .market_utils import (
    get_market_session, classify_symbol, get_sampling_rate,
    get_financial_attributes, should_trace_operation, MarketSession, SymbolTier
)

logger = logging.getLogger(__name__)

class BusinessAwareSampler(Sampler):
    """
    Custom OpenTelemetry sampler that implements business-aware sampling logic
    """

    def __init__(self, fallback_rate: float = 0.1):
        """
        Initialize business-aware sampler

        Args:
            fallback_rate: Default sampling rate for operations without business context
        """
        self.fallback_rate = fallback_rate

    def should_sample(
        self,
        parent_context,
        trace_id: int,
        name: str,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None
    ) -> SamplingResult:
        """
        Implement business-aware sampling logic

        Args:
            parent_context: Parent span context
            trace_id: Trace ID
            name: Span name
            kind: Span kind
            attributes: Span attributes
            links: Span links
            trace_state: Trace state

        Returns:
            SamplingResult with sampling decision
        """
        if attributes is None:
            attributes = {}

        # Always sample if parent is sampled
        if parent_context and parent_context.trace_flags.sampled:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, trace_state)

        # Extract business context from attributes
        symbol = attributes.get("finance.symbol")
        confidence_score = attributes.get("finance.confidence_score")
        is_error = attributes.get("error.type") is not None

        # Use business logic to determine sampling
        if symbol:
            sampling_rate = get_sampling_rate(
                symbol=symbol,
                is_error=is_error,
                confidence_score=confidence_score
            )
        else:
            sampling_rate = self.fallback_rate

        # Deterministic sampling based on trace_id
        should_sample = (trace_id % 100) < (sampling_rate * 100)

        decision = Decision.RECORD_AND_SAMPLE if should_sample else Decision.DROP

        return SamplingResult(decision, attributes, trace_state)

    def get_description(self) -> str:
        return f"BusinessAwareSampler(fallback_rate={self.fallback_rate})"

class FinancialTracer:
    """
    Enhanced tracer for financial operations with business context
    """

    def __init__(self, service_name: str):
        """
        Initialize financial tracer

        Args:
            service_name: Name of the service for tracing
        """
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)

    def start_financial_span(
        self,
        operation_name: str,
        symbol: Optional[str] = None,
        current_price: Optional[float] = None,
        target_price: Optional[float] = None,
        confidence_score: Optional[float] = None,
        recommendation: Optional[str] = None,
        **kwargs
    ) -> Span:
        """
        Start a span with comprehensive financial attributes

        Args:
            operation_name: Name of the operation
            symbol: Stock symbol
            current_price: Current stock price
            target_price: Predicted target price
            confidence_score: ML model confidence
            recommendation: Buy/sell/hold recommendation
            **kwargs: Additional attributes

        Returns:
            OpenTelemetry span with financial attributes
        """
        span = self.tracer.start_span(operation_name)

        # Add financial domain attributes
        if symbol:
            financial_attrs = get_financial_attributes(
                symbol=symbol,
                current_price=current_price,
                target_price=target_price,
                confidence_score=confidence_score,
                recommendation=recommendation
            )
            span.set_attributes(financial_attrs)

        # Add service context
        span.set_attributes({
            "service.name": self.service_name,
            "service.operation": operation_name,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Add any additional attributes
        if kwargs:
            span.set_attributes(kwargs)

        return span

    def trace_prediction(
        self,
        prediction_type: str,
        symbol: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        model_version: str,
        execution_time_ms: Optional[float] = None
    ) -> Span:
        """
        Create a comprehensive prediction trace

        Args:
            prediction_type: Type of prediction (price, time, etc.)
            symbol: Stock symbol
            input_data: Model input data
            output_data: Model output/prediction
            model_version: Version of the ML model
            execution_time_ms: Execution time in milliseconds

        Returns:
            OpenTelemetry span for the prediction
        """
        span_name = f"ml.{prediction_type}_prediction"
        span = self.start_financial_span(
            span_name,
            symbol=symbol,
            current_price=input_data.get('current_price'),
            target_price=output_data.get('target_price'),
            confidence_score=output_data.get('confidence'),
            recommendation=output_data.get('recommendation')
        )

        # Add ML-specific attributes
        ml_attributes = {
            "ml.prediction_type": prediction_type,
            "ml.model_version": model_version,
            "ml.input_features_count": len(input_data),
            "ml.prediction_success": True
        }

        if execution_time_ms:
            ml_attributes["ml.execution_time_ms"] = execution_time_ms

        # Add specific input/output attributes
        for key, value in input_data.items():
            if isinstance(value, (int, float, str, bool)):
                ml_attributes[f"ml.input.{key}"] = value

        for key, value in output_data.items():
            if isinstance(value, (int, float, str, bool)):
                ml_attributes[f"ml.output.{key}"] = value

        span.set_attributes(ml_attributes)
        return span

def trace_financial_operation(
    operation_name: str,
    tracer: Optional[FinancialTracer] = None
):
    """
    Decorator for tracing financial operations with business context

    Args:
        operation_name: Name of the operation to trace
        tracer: Optional FinancialTracer instance

    Returns:
        Decorated function with tracing
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract financial context from function arguments
            symbol = kwargs.get('symbol') or (args[0] if args and isinstance(args[0], str) else None)

            # Use provided tracer or get default
            if tracer is None:
                default_tracer = trace.get_tracer(__name__)
                span = default_tracer.start_span(operation_name)
            else:
                span = tracer.start_financial_span(operation_name, symbol=symbol)

            try:
                # Add function context
                span.set_attributes({
                    "function.name": func.__name__,
                    "function.args_count": len(args),
                    "function.kwargs_count": len(kwargs)
                })

                # Execute function
                start_time = datetime.utcnow()
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()

                # Add execution metrics
                execution_time = (end_time - start_time).total_seconds() * 1000
                span.set_attributes({
                    "function.execution_time_ms": execution_time,
                    "function.success": True
                })

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                # Add error context
                span.set_attributes({
                    "function.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            finally:
                span.end()

        return wrapper
    return decorator

def create_correlation_id() -> str:
    """
    Create a correlation ID for async workflow tracking

    Returns:
        UUID-based correlation ID
    """
    return str(uuid.uuid4())

def propagate_correlation_context(
    correlation_id: Optional[str] = None,
    parent_operation: Optional[str] = None,
    symbols: Optional[list] = None
) -> Dict[str, Any]:
    """
    Create correlation context for async Lambda invocations

    Args:
        correlation_id: Existing correlation ID or create new one
        parent_operation: Name of the parent operation
        symbols: List of symbols being processed

    Returns:
        Dictionary with correlation context for Lambda payloads
    """
    if correlation_id is None:
        correlation_id = create_correlation_id()

    context = {
        "correlation_id": correlation_id,
        "parent_operation": parent_operation,
        "timestamp": datetime.utcnow().isoformat(),
        "market_session": get_market_session().value
    }

    if symbols:
        context["symbols"] = symbols
        context["symbol_count"] = len(symbols)

        # Add symbol tier distribution
        tier_counts = {}
        for symbol in symbols:
            tier = classify_symbol(symbol)
            tier_counts[tier.value] = tier_counts.get(tier.value, 0) + 1

        context["symbol_tier_distribution"] = tier_counts

    return context

def extract_correlation_context(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract correlation context from Lambda event

    Args:
        event: Lambda event payload

    Returns:
        Correlation context if present, None otherwise
    """
    correlation_fields = ['correlation_id', 'parent_operation', 'timestamp', 'market_session']

    # Check if event contains correlation context
    if any(field in event for field in correlation_fields):
        return {field: event.get(field) for field in correlation_fields if field in event}

    return None

def enhance_span_with_correlation(span: Span, correlation_context: Dict[str, Any]) -> None:
    """
    Add correlation attributes to a span

    Args:
        span: OpenTelemetry span to enhance
        correlation_context: Correlation context from async workflow
    """
    correlation_attrs = {}

    for key, value in correlation_context.items():
        if key in ['correlation_id', 'parent_operation', 'market_session']:
            correlation_attrs[f"workflow.{key}"] = value
        elif key == 'symbol_count':
            correlation_attrs["workflow.symbol_count"] = value
        elif key == 'symbol_tier_distribution':
            for tier, count in value.items():
                correlation_attrs[f"workflow.{tier}_count"] = count

    span.set_attributes(correlation_attrs)

# Global financial tracer instance
_global_financial_tracer: Optional[FinancialTracer] = None

def get_financial_tracer(service_name: Optional[str] = None) -> FinancialTracer:
    """
    Get or create global financial tracer instance

    Args:
        service_name: Service name for the tracer

    Returns:
        FinancialTracer instance
    """
    global _global_financial_tracer

    if _global_financial_tracer is None:
        _global_financial_tracer = FinancialTracer(service_name or "stock_analytics")

    return _global_financial_tracer

# Convenience functions for common operations
def trace_data_ingestion(symbols: list, processing_time_ms: float) -> Span:
    """Create a span for data ingestion operations"""
    tracer = get_financial_tracer()
    span = tracer.start_financial_span("data_ingestion.process_symbols")

    span.set_attributes({
        "data_ingestion.symbol_count": len(symbols),
        "data_ingestion.processing_time_ms": processing_time_ms,
        "data_ingestion.market_session": get_market_session().value
    })

    # Add symbol tier breakdown
    tier_counts = {}
    for symbol in symbols:
        tier = classify_symbol(symbol).value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    for tier, count in tier_counts.items():
        span.set_attribute(f"data_ingestion.{tier}_symbols", count)

    return span

def trace_api_request(
    endpoint: str,
    method: str,
    symbol: Optional[str] = None,
    response_time_ms: Optional[float] = None
) -> Span:
    """Create a span for API requests"""
    tracer = get_financial_tracer()
    span = tracer.start_financial_span("api.request", symbol=symbol)

    span.set_attributes({
        "http.endpoint": endpoint,
        "http.method": method,
        "api.market_session": get_market_session().value
    })

    if response_time_ms:
        span.set_attribute("http.response_time_ms", response_time_ms)

    return span