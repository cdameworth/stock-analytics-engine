"""
EventBridge X-Ray Trace Propagation for Stock Analytics
Implements proper trace context propagation across EventBridge events and Lambda functions
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

# OpenTelemetry and X-Ray SDK imports
try:
    from aws_xray_sdk.core import xray_recorder, patch_all
    from aws_xray_sdk.core.context import Context
    from aws_xray_sdk.core.models.trace_header import TraceHeader
    XRAY_SDK_AVAILABLE = True
except ImportError:
    XRAY_SDK_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.propagate import extract, inject
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Patch AWS services for automatic X-Ray tracing
if XRAY_SDK_AVAILABLE:
    patch_all()

class EventBridgeTracer:
    """
    Enhanced EventBridge tracing with X-Ray and OpenTelemetry integration
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.propagator = TraceContextTextMapPropagator() if OTEL_AVAILABLE else None

    def extract_trace_context_from_event(self, event: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Extract trace context from EventBridge event

        Args:
            event: EventBridge event payload

        Returns:
            Trace context headers if available
        """
        trace_context = {}

        # Check for X-Ray trace ID in environment
        xray_trace_id = os.environ.get('_X_AMZN_TRACE_ID')
        if xray_trace_id:
            trace_context['X-Amzn-Trace-Id'] = xray_trace_id

        # Check for trace context in event headers (custom events)
        if 'headers' in event:
            for header_name in ['X-Amzn-Trace-Id', 'traceparent', 'tracestate']:
                if header_name in event['headers']:
                    trace_context[header_name] = event['headers'][header_name]

        # Check for trace context in event detail (EventBridge specific)
        if 'detail' in event and isinstance(event['detail'], dict):
            if 'trace_context' in event['detail']:
                trace_context.update(event['detail']['trace_context'])

        # Extract from EventBridge source attributes
        if 'source' in event and event['source'] == 'custom.stock-analytics':
            # This is our custom event, extract correlation context
            if 'detail' in event and 'correlation_id' in event['detail']:
                trace_context['correlation_id'] = event['detail']['correlation_id']

        logger.debug(f"Extracted trace context: {trace_context}")
        return trace_context if trace_context else None

    def create_trace_context_for_event(self, correlation_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create trace context for outgoing EventBridge events

        Args:
            correlation_id: Optional correlation ID for async workflow tracking

        Returns:
            Trace context headers for EventBridge event
        """
        trace_context = {}

        # Add X-Ray trace ID if available
        if XRAY_SDK_AVAILABLE:
            try:
                current_trace_id = xray_recorder.get_trace_id()
                if current_trace_id:
                    trace_context['X-Amzn-Trace-Id'] = current_trace_id
            except Exception as e:
                logger.warning(f"Could not get X-Ray trace ID: {e}")

        # Add OpenTelemetry trace context
        if OTEL_AVAILABLE and self.propagator:
            try:
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    carrier = {}
                    self.propagator.inject(carrier)
                    trace_context.update(carrier)
            except Exception as e:
                logger.warning(f"Could not inject OpenTelemetry context: {e}")

        # Add correlation ID for async workflow tracking
        if correlation_id:
            trace_context['correlation_id'] = correlation_id
        else:
            trace_context['correlation_id'] = str(uuid.uuid4())

        # Add business context
        trace_context.update({
            'service_name': self.service_name,
            'business_domain': 'financial',
            'system_type': 'stock-analytics',
            'trace_timestamp': datetime.utcnow().isoformat()
        })

        return trace_context

    def trace_eventbridge_publish(self, event_data: Dict[str, Any], event_bus_name: str = 'default') -> Dict[str, Any]:
        """
        Publish EventBridge event with proper trace propagation

        Args:
            event_data: Event data to publish
            event_bus_name: EventBridge bus name

        Returns:
            Enhanced event data with trace context
        """
        # Create trace context for the event
        trace_context = self.create_trace_context_for_event()

        # Enhance event data with trace context
        enhanced_event = {
            'Source': event_data.get('Source', 'custom.stock-analytics'),
            'DetailType': event_data.get('DetailType', 'Stock Analytics Event'),
            'Detail': json.dumps({
                **event_data.get('Detail', {}),
                'trace_context': trace_context
            }),
            'EventBusName': event_bus_name
        }

        # Add trace attributes if X-Ray is available
        if XRAY_SDK_AVAILABLE:
            try:
                subsegment = xray_recorder.current_subsegment()
                if subsegment:
                    subsegment.put_annotation('event_source', enhanced_event['Source'])
                    subsegment.put_annotation('event_type', enhanced_event['DetailType'])
                    subsegment.put_annotation('event_bus', event_bus_name)
                    subsegment.put_metadata('event_detail', event_data.get('Detail', {}))
            except Exception as e:
                logger.warning(f"Could not add X-Ray annotations: {e}")

        return enhanced_event

    def start_trace_from_eventbridge_event(self, event: Dict[str, Any], operation_name: str):
        """
        Start a new trace or continue existing trace from EventBridge event

        Args:
            event: EventBridge event
            operation_name: Name of the operation being traced

        Returns:
            Trace context manager or None
        """
        trace_context = self.extract_trace_context_from_event(event)

        if not trace_context:
            logger.info(f"No trace context found in event, starting new trace for {operation_name}")

        # Create business-aware span attributes
        span_attributes = {
            'operation.name': operation_name,
            'event.source': event.get('source', 'unknown'),
            'event.detail_type': event.get('detail-type', 'unknown'),
            'business.domain': 'financial',
            'system.component': self.service_name
        }

        # Add correlation ID if available
        if trace_context and 'correlation_id' in trace_context:
            span_attributes['workflow.correlation_id'] = trace_context['correlation_id']

        # Add EventBridge specific attributes
        if 'detail' in event:
            if isinstance(event['detail'], dict):
                # Add financial context from event detail
                for key in ['symbol', 'prediction_type', 'trigger_type', 'market_session']:
                    if key in event['detail']:
                        span_attributes[f'event.{key}'] = event['detail'][key]

        # Return span attributes for use with tracer
        return span_attributes

class LambdaEventBridgeIntegration:
    """
    Integration helper for Lambda functions triggered by EventBridge
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = EventBridgeTracer(service_name)

    def handle_eventbridge_trigger(self, event: Dict[str, Any], context, operation_name: str):
        """
        Handle Lambda function triggered by EventBridge with proper tracing

        Args:
            event: Lambda event (EventBridge trigger)
            context: Lambda context
            operation_name: Name of the operation

        Returns:
            Span attributes and correlation context
        """
        # Extract trace context and create span attributes
        span_attributes = self.tracer.start_trace_from_eventbridge_event(event, operation_name)

        # Add Lambda context information
        span_attributes.update({
            'lambda.request_id': context.aws_request_id,
            'lambda.function_name': context.function_name,
            'lambda.function_version': context.function_version,
            'lambda.memory_limit': context.memory_limit_in_mb,
            'lambda.remaining_time': context.get_remaining_time_in_millis()
        })

        # Determine trigger type
        trigger_type = 'unknown'
        if 'source' in event:
            if event['source'] == 'aws.events':
                trigger_type = 'scheduled'
            elif event['source'] == 'custom.stock-analytics':
                trigger_type = 'event_driven'

        span_attributes['trigger.type'] = trigger_type

        # Extract correlation context for async workflow tracking
        correlation_context = self.tracer.extract_trace_context_from_event(event)

        return span_attributes, correlation_context

    def publish_downstream_event(self, event_data: Dict[str, Any], event_bus_name: str = 'default'):
        """
        Publish downstream EventBridge event with trace propagation

        Args:
            event_data: Event data to publish
            event_bus_name: EventBridge bus name

        Returns:
            Enhanced event ready for EventBridge publication
        """
        return self.tracer.trace_eventbridge_publish(event_data, event_bus_name)

# Global instance for easy access
_eventbridge_integration: Optional[LambdaEventBridgeIntegration] = None

def get_eventbridge_integration(service_name: str = "stock-analytics") -> LambdaEventBridgeIntegration:
    """Get or create global EventBridge integration instance"""
    global _eventbridge_integration
    if _eventbridge_integration is None:
        _eventbridge_integration = LambdaEventBridgeIntegration(service_name)
    return _eventbridge_integration

# Convenience decorator for EventBridge-triggered Lambda functions
def trace_eventbridge_handler(operation_name: str):
    """
    Decorator for Lambda handlers triggered by EventBridge

    Args:
        operation_name: Name of the operation being performed
    """
    def decorator(handler_func):
        def wrapper(event, context):
            integration = get_eventbridge_integration()

            # Handle EventBridge trigger with tracing
            span_attributes, correlation_context = integration.handle_eventbridge_trigger(
                event, context, operation_name
            )

            # Add span attributes to event for use in handler
            enhanced_event = {
                **event,
                '_trace_attributes': span_attributes,
                '_correlation_context': correlation_context
            }

            # Execute the original handler
            try:
                result = handler_func(enhanced_event, context)

                # Add success attributes if X-Ray is available
                if XRAY_SDK_AVAILABLE:
                    try:
                        subsegment = xray_recorder.current_subsegment()
                        if subsegment:
                            subsegment.put_annotation('operation_success', True)
                            subsegment.put_annotation('operation_name', operation_name)
                    except Exception as e:
                        logger.warning(f"Could not add success annotations: {e}")

                return result

            except Exception as e:
                # Add error attributes if X-Ray is available
                if XRAY_SDK_AVAILABLE:
                    try:
                        subsegment = xray_recorder.current_subsegment()
                        if subsegment:
                            subsegment.put_annotation('operation_success', False)
                            subsegment.put_annotation('error_type', type(e).__name__)
                            subsegment.put_metadata('error_details', {
                                'message': str(e),
                                'operation': operation_name
                            })
                    except Exception as annotation_error:
                        logger.warning(f"Could not add error annotations: {annotation_error}")

                raise

        return wrapper
    return decorator