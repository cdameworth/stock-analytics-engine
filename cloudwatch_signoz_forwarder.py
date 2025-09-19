"""
CloudWatch to SigNoz Log Forwarder
Forwards CloudWatch logs to SigNoz Cloud via HTTP API
"""

import json
import gzip
import base64
import urllib3
import os

def lambda_handler(event, context):
    """Forward CloudWatch logs to SigNoz via HTTP"""

    signoz_endpoint = os.environ.get('SIGNOZ_ENDPOINT', 'ingest.us.signoz.cloud:443')
    signoz_token = os.environ.get('SIGNOZ_TOKEN', '')

    if not signoz_token:
        print("ERROR: SIGNOZ_TOKEN not provided")
        return {"statusCode": 400, "body": "Missing SigNoz token"}

    try:
        # Parse CloudWatch Logs data
        cw_data = event['awslogs']['data']
        compressed_payload = base64.b64decode(cw_data)
        uncompressed_payload = gzip.decompress(compressed_payload)
        log_data = json.loads(uncompressed_payload)

        print(f"Processing {len(log_data['logEvents'])} log events from {log_data['logGroup']}")

        # Format logs for SigNoz OTLP format
        signoz_logs = []
        for log_event in log_data['logEvents']:
            # Convert timestamp to nanoseconds (SigNoz expects nanoseconds)
            timestamp_ns = log_event['timestamp'] * 1000000  # Convert ms to ns

            signoz_log = {
                "timeUnixNano": str(timestamp_ns),
                "body": {
                    "stringValue": log_event['message']
                },
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": extract_service_name(log_data['logGroup'])}},
                    {"key": "log.group", "value": {"stringValue": log_data['logGroup']}},
                    {"key": "log.stream", "value": {"stringValue": log_data['logStream']}},
                    {"key": "aws.region", "value": {"stringValue": context.invoked_function_arn.split(':')[3]}},
                    {"key": "aws.account.id", "value": {"stringValue": context.invoked_function_arn.split(':')[4]}},
                    {"key": "log.source", "value": {"stringValue": "cloudwatch"}}
                ],
                "severityText": "INFO"
            }
            signoz_logs.append(signoz_log)

        # Create OTLP logs payload
        payload = {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "service.namespace", "value": {"stringValue": "stock-analytics"}},
                        {"key": "deployment.environment", "value": {"stringValue": "development"}}
                    ]
                },
                "scopeLogs": [{
                    "scope": {
                        "name": "cloudwatch-forwarder",
                        "version": "1.0.0"
                    },
                    "logRecords": signoz_logs
                }]
            }]
        }

        # Send to SigNoz
        http = urllib3.PoolManager()

        # Ensure endpoint has https protocol
        if not signoz_endpoint.startswith('http'):
            signoz_endpoint = f"https://{signoz_endpoint}"

        headers = {
            'Content-Type': 'application/json',
            'signoz-access-token': signoz_token
        }

        response = http.request(
            'POST',
            f"{signoz_endpoint}/v1/logs",
            body=json.dumps(payload),
            headers=headers,
            timeout=urllib3.Timeout(total=30)
        )

        print(f"Sent {len(signoz_logs)} logs to SigNoz. Response: {response.status}")

        if response.status >= 400:
            print(f"SigNoz API Error: {response.data.decode('utf-8')}")
            return {"statusCode": response.status, "body": f"SigNoz API error: {response.status}"}

        return {
            "statusCode": 200,
            "body": f"Successfully forwarded {len(signoz_logs)} logs to SigNoz"
        }

    except Exception as e:
        print(f"Error forwarding to SigNoz: {str(e)}")
        return {"statusCode": 500, "body": f"Error: {str(e)}"}

def extract_service_name(log_group):
    """Extract service name from CloudWatch log group"""
    # Examples:
    # /aws/lambda/stock-recommendations-api -> stock-recommendations-api
    # /aws/rds/instance/aurora -> rds-aurora

    parts = log_group.strip('/').split('/')

    if len(parts) >= 3:
        if parts[1] == 'lambda':
            return parts[2]  # Lambda function name
        elif parts[1] == 'rds':
            return f"rds-{parts[-1]}"  # RDS instance/cluster
        elif parts[1] == 'sns':
            return "sns"
        elif parts[1] == 'apigateway':
            return "api-gateway"
        else:
            return f"{parts[1]}-{parts[-1]}"

    return parts[-1] if parts else "unknown"