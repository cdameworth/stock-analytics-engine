#!/usr/bin/env python3
"""
CloudWatch Logs to SigNoz Forwarder
Forwards CloudWatch logs to SigNoz Cloud via OTLP/gRPC

This Lambda function receives CloudWatch log events and forwards them
to SigNoz Cloud using the OpenTelemetry Protocol over gRPC.
"""

import base64
import gzip
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import urllib3
import ssl

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# SigNoz configuration
SIGNOZ_ENDPOINT = os.environ.get('SIGNOZ_ENDPOINT', 'ingest.us.signoz.cloud:443')
SIGNOZ_TOKEN = os.environ.get('SIGNOZ_TOKEN', '')

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for forwarding CloudWatch logs to SigNoz

    Args:
        event: CloudWatch Logs event
        context: Lambda context

    Returns:
        Response dictionary
    """
    try:
        # Parse the CloudWatch Logs event
        logs_data = parse_cloudwatch_logs(event)

        if not logs_data:
            logger.warning("No log data found in event")
            return {'statusCode': 200, 'body': 'No logs to process'}

        # Forward logs to SigNoz
        success_count, error_count = forward_logs_to_signoz(logs_data)

        logger.info(f"Processed {success_count} logs successfully, {error_count} errors")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Logs forwarded successfully',
                'processed': success_count,
                'errors': error_count
            })
        }

    except Exception as e:
        logger.error(f"Error processing CloudWatch logs: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def parse_cloudwatch_logs(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse CloudWatch Logs event and extract log records

    Args:
        event: CloudWatch Logs event

    Returns:
        List of parsed log records
    """
    try:
        # Get the compressed log data from the event
        compressed_payload = base64.b64decode(event['awslogs']['data'])

        # Decompress the log data
        log_data = gzip.decompress(compressed_payload)

        # Parse JSON
        log_events = json.loads(log_data.decode('utf-8'))

        # Extract log group and stream info
        log_group = log_events.get('logGroup', '')
        log_stream = log_events.get('logStream', '')

        # Process each log event
        parsed_logs = []
        for log_event in log_events.get('logEvents', []):
            parsed_log = {
                'timestamp': log_event.get('timestamp', 0),
                'message': log_event.get('message', ''),
                'id': log_event.get('id', ''),
                'logGroup': log_group,
                'logStream': log_stream,
                'service': extract_service_name(log_group),
                'level': extract_log_level(log_event.get('message', '')),
                'source': 'aws-cloudwatch'
            }
            parsed_logs.append(parsed_log)

        return parsed_logs

    except Exception as e:
        logger.error(f"Error parsing CloudWatch logs: {str(e)}")
        return []

def extract_service_name(log_group: str) -> str:
    """
    Extract service name from CloudWatch log group name

    Args:
        log_group: CloudWatch log group name

    Returns:
        Service name
    """
    # Extract service name from log group patterns
    if '/aws/lambda/' in log_group:
        return log_group.replace('/aws/lambda/', '')
    elif '/aws/apigateway/' in log_group:
        return 'api-gateway'
    elif log_group.startswith('/aws/'):
        return log_group.split('/')[2] if len(log_group.split('/')) > 2 else 'unknown'
    else:
        return 'unknown'

def extract_log_level(message: str) -> str:
    """
    Extract log level from log message

    Args:
        message: Log message

    Returns:
        Log level (ERROR, WARN, INFO, DEBUG)
    """
    message_upper = message.upper()

    if any(word in message_upper for word in ['ERROR', 'EXCEPTION', 'FAIL', 'FATAL']):
        return 'ERROR'
    elif any(word in message_upper for word in ['WARN', 'WARNING']):
        return 'WARN'
    elif any(word in message_upper for word in ['DEBUG', 'TRACE']):
        return 'DEBUG'
    else:
        return 'INFO'

def forward_logs_to_signoz(logs_data: List[Dict[str, Any]]) -> tuple:
    """
    Forward logs to SigNoz Cloud

    Args:
        logs_data: List of log records

    Returns:
        Tuple of (success_count, error_count)
    """
    if not SIGNOZ_TOKEN:
        logger.error("SIGNOZ_TOKEN not configured")
        return 0, len(logs_data)

    success_count = 0
    error_count = 0

    # Create OTLP log payload
    otlp_payload = create_otlp_logs_payload(logs_data)

    try:
        # Send to SigNoz
        response = send_to_signoz(otlp_payload)

        if response.status == 200:
            success_count = len(logs_data)
            logger.info(f"Successfully sent {success_count} logs to SigNoz")
        else:
            error_count = len(logs_data)
            logger.error(f"Failed to send logs to SigNoz: {response.status} {response.data}")

    except Exception as e:
        error_count = len(logs_data)
        logger.error(f"Error sending logs to SigNoz: {str(e)}")

    return success_count, error_count

def create_otlp_logs_payload(logs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create OTLP logs payload for SigNoz

    Args:
        logs_data: List of log records

    Returns:
        OTLP logs payload
    """
    # Group logs by service
    logs_by_service = {}
    for log in logs_data:
        service = log.get('service', 'unknown')
        if service not in logs_by_service:
            logs_by_service[service] = []
        logs_by_service[service].append(log)

    # Create OTLP resource logs
    resource_logs = []

    for service, service_logs in logs_by_service.items():
        # Create log records for this service
        log_records = []
        for log in service_logs:
            log_record = {
                "timeUnixNano": str(int(log['timestamp'] * 1000000)),  # Convert to nanoseconds
                "severityText": log.get('level', 'INFO'),
                "severityNumber": get_severity_number(log.get('level', 'INFO')),
                "body": {
                    "stringValue": log.get('message', '')
                },
                "attributes": [
                    {"key": "log.group", "value": {"stringValue": log.get('logGroup', '')}},
                    {"key": "log.stream", "value": {"stringValue": log.get('logStream', '')}},
                    {"key": "aws.log.id", "value": {"stringValue": log.get('id', '')}},
                    {"key": "source", "value": {"stringValue": log.get('source', 'aws-cloudwatch')}},
                    {"key": "service.name", "value": {"stringValue": service}},
                    {"key": "service.namespace", "value": {"stringValue": "stock-analytics"}},
                    {"key": "deployment.environment", "value": {"stringValue": "development"}}
                ]
            }
            log_records.append(log_record)

        # Create scope logs
        scope_logs = [{
            "scope": {
                "name": "aws-cloudwatch-forwarder",
                "version": "1.0.0"
            },
            "logRecords": log_records
        }]

        # Create resource logs
        resource_log = {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service}},
                    {"key": "service.namespace", "value": {"stringValue": "stock-analytics"}},
                    {"key": "cloud.provider", "value": {"stringValue": "aws"}},
                    {"key": "cloud.platform", "value": {"stringValue": "aws_lambda"}},
                    {"key": "deployment.environment", "value": {"stringValue": "development"}}
                ]
            },
            "scopeLogs": scope_logs
        }
        resource_logs.append(resource_log)

    return {
        "resourceLogs": resource_logs
    }

def get_severity_number(level: str) -> int:
    """
    Get OTLP severity number for log level

    Args:
        level: Log level string

    Returns:
        OTLP severity number
    """
    severity_map = {
        'TRACE': 1,
        'DEBUG': 5,
        'INFO': 9,
        'WARN': 13,
        'ERROR': 17,
        'FATAL': 21
    }
    return severity_map.get(level.upper(), 9)  # Default to INFO

def send_to_signoz(payload: Dict[str, Any]) -> Any:
    """
    Send OTLP payload to SigNoz

    Args:
        payload: OTLP logs payload

    Returns:
        HTTP response
    """
    # Prepare headers
    headers = {
        'Content-Type': 'application/json',
        'signoz-access-token': SIGNOZ_TOKEN
    }

    # Create HTTP client
    http = urllib3.PoolManager(
        cert_reqs=ssl.CERT_REQUIRED,
        ca_certs=None
    )

    # Send POST request
    url = f"https://{SIGNOZ_ENDPOINT}/v1/logs"
    encoded_data = json.dumps(payload).encode('utf-8')

    response = http.request(
        'POST',
        url,
        body=encoded_data,
        headers=headers,
        timeout=30
    )

    return response

def health_check() -> Dict[str, Any]:
    """
    Health check function for monitoring

    Returns:
        Health status
    """
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'service': 'cloudwatch-signoz-forwarder',
            'timestamp': datetime.utcnow().isoformat(),
            'signoz_endpoint': SIGNOZ_ENDPOINT,
            'signoz_configured': bool(SIGNOZ_TOKEN)
        })
    }

# Test function for local development
if __name__ == "__main__":
    # Test event for local development
    test_event = {
        "awslogs": {
            "data": base64.b64encode(gzip.compress(json.dumps({
                "messageType": "DATA_MESSAGE",
                "owner": "123456789012",
                "logGroup": "/aws/lambda/test-function",
                "logStream": "2023/01/01/[$LATEST]abcd1234",
                "subscriptionFilters": ["test-filter"],
                "logEvents": [
                    {
                        "id": "eventId1",
                        "timestamp": 1672531200000,
                        "message": "INFO: Test log message"
                    }
                ]
            }).encode('utf-8'))).decode('utf-8')
        }
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))