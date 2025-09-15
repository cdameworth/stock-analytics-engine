"""
Model Tuning Reporter Lambda Function
Sends comprehensive reports after model tuning runs with findings and actions taken
"""

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
cloudwatch = boto3.client('cloudwatch')
lambda_client = boto3.client('lambda')

# Environment variables
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', 'arn:aws:sns:us-east-1:791060928878:stock-analytics-daily-reports')
TUNING_HISTORY_TABLE = os.environ.get('TUNING_HISTORY_TABLE', 'model-tuning-history')
ACCURACY_METRICS_TABLE = os.environ.get('ACCURACY_METRICS_TABLE', 'prediction-accuracy-metrics')
PRICE_PREDICTIONS_TABLE = os.environ.get('PRICE_PREDICTIONS_TABLE', 'price-predictions')
TIME_PREDICTIONS_TABLE = os.environ.get('TIME_PREDICTIONS_TABLE', 'time-to-hit-predictions')

def decimal_default(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def get_tuning_metrics(model_type: str, hours_back: int = 24) -> Dict:
    """
    Retrieve tuning metrics from CloudWatch and DynamoDB
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        # Get CloudWatch metrics
        metrics = {}

        # Get accuracy metrics
        response = cloudwatch.get_metric_statistics(
            Namespace='StockAnalytics/ModelTuning',
            MetricName=f'{model_type}_accuracy',
            Dimensions=[
                {'Name': 'ModelType', 'Value': model_type}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum', 'Minimum']
        )

        if response['Datapoints']:
            latest = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])[-1]
            metrics['accuracy'] = {
                'current': latest.get('Average', 0),
                'max': latest.get('Maximum', 0),
                'min': latest.get('Minimum', 0)
            }

        # Get prediction counts
        if model_type == 'price':
            table = dynamodb.Table(PRICE_PREDICTIONS_TABLE)
        else:
            table = dynamodb.Table(TIME_PREDICTIONS_TABLE)

        # Count recent predictions
        response = table.scan(
            FilterExpression='prediction_timestamp > :start_time',
            ExpressionAttributeValues={
                ':start_time': start_time.isoformat()
            },
            Select='COUNT'
        )

        metrics['prediction_count'] = response.get('Count', 0)

        return metrics

    except Exception as e:
        logger.error(f"Error getting tuning metrics: {str(e)}")
        return {}

def get_tuning_history(model_type: str, limit: int = 5) -> List[Dict]:
    """
    Get recent tuning history from DynamoDB
    """
    try:
        table = dynamodb.Table(TUNING_HISTORY_TABLE)

        response = table.query(
            KeyConditionExpression='model_type = :model_type',
            ExpressionAttributeValues={
                ':model_type': model_type
            },
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )

        return response.get('Items', [])

    except Exception as e:
        logger.error(f"Error getting tuning history: {str(e)}")
        return []

def analyze_tuning_results(tuning_data: Dict) -> Dict:
    """
    Analyze tuning results and determine actions taken
    """
    analysis = {
        'improvements': [],
        'concerns': [],
        'actions_taken': [],
        'recommendations': []
    }

    # Analyze accuracy changes
    if 'before_accuracy' in tuning_data and 'after_accuracy' in tuning_data:
        before = float(tuning_data['before_accuracy'])
        after = float(tuning_data['after_accuracy'])
        change = after - before

        if change > 0.02:  # 2% improvement
            analysis['improvements'].append(f"✅ Accuracy improved by {change:.1%} (from {before:.1%} to {after:.1%})")
            analysis['actions_taken'].append("Model parameters successfully optimized")
        elif change < -0.02:  # 2% degradation
            analysis['concerns'].append(f"⚠️ Accuracy decreased by {abs(change):.1%}")
            analysis['actions_taken'].append("Reverted to previous model version")
        else:
            analysis['improvements'].append(f"✓ Accuracy stable at {after:.1%}")
            analysis['actions_taken'].append("Model parameters maintained")

    # Analyze feature importance changes
    if 'feature_changes' in tuning_data:
        significant_changes = [f for f in tuning_data['feature_changes'] if abs(f.get('change', 0)) > 0.1]
        if significant_changes:
            analysis['improvements'].append(f"📊 {len(significant_changes)} features significantly reweighted")
            analysis['actions_taken'].append("Feature importance recalibrated")

    # Analyze hyperparameter tuning
    if 'hyperparameters' in tuning_data:
        hp = tuning_data['hyperparameters']
        if hp.get('learning_rate_adjusted'):
            analysis['actions_taken'].append(f"Learning rate adjusted to {hp.get('learning_rate', 'optimal')}")
        if hp.get('regularization_changed'):
            analysis['actions_taken'].append("Regularization parameters optimized")

    # Analyze prediction distribution
    if 'prediction_distribution' in tuning_data:
        dist = tuning_data['prediction_distribution']
        buy_ratio = dist.get('buy', 0) / max(dist.get('total', 1), 1)
        sell_ratio = dist.get('sell', 0) / max(dist.get('total', 1), 1)

        if buy_ratio > 0.6:
            analysis['concerns'].append("⚠️ Model showing bullish bias (>60% BUY signals)")
            analysis['recommendations'].append("Consider market regime adjustment")
        elif sell_ratio > 0.6:
            analysis['concerns'].append("⚠️ Model showing bearish bias (>60% SELL signals)")
            analysis['recommendations'].append("Consider market regime adjustment")
        else:
            analysis['improvements'].append("✅ Balanced prediction distribution")

    # Add performance metrics
    if 'performance_metrics' in tuning_data:
        perf = tuning_data['performance_metrics']
        if perf.get('sharpe_ratio', 0) > 1.0:
            analysis['improvements'].append(f"📈 Strong risk-adjusted returns (Sharpe: {perf.get('sharpe_ratio', 0):.2f})")
        if perf.get('max_drawdown', 1) < 0.15:
            analysis['improvements'].append(f"✅ Low maximum drawdown: {perf.get('max_drawdown', 0):.1%}")

    return analysis

def format_tuning_report(model_type: str, tuning_data: Dict, analysis: Dict) -> str:
    """
    Format the tuning report for email
    """
    model_name = "Price Prediction Model" if model_type == 'price' else "Time-to-Hit Prediction Model"

    report = f"""
📊 MODEL TUNING REPORT
═══════════════════════════════════════════════════════════════

🤖 Model: {model_name}
📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
🔧 Tuning Type: {tuning_data.get('tuning_type', 'Scheduled Weekly Optimization')}

═══════════════════════════════════════════════════════════════
📈 PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════════

Current Accuracy: {tuning_data.get('after_accuracy', 0):.1%}
Previous Accuracy: {tuning_data.get('before_accuracy', 0):.1%}
Change: {(tuning_data.get('after_accuracy', 0) - tuning_data.get('before_accuracy', 0)):.1%}

Predictions Analyzed: {tuning_data.get('predictions_analyzed', 0):,}
Training Samples: {tuning_data.get('training_samples', 0):,}
Validation Samples: {tuning_data.get('validation_samples', 0):,}

═══════════════════════════════════════════════════════════════
✅ IMPROVEMENTS
═══════════════════════════════════════════════════════════════
"""

    for improvement in analysis['improvements']:
        report += f"• {improvement}\n"

    if analysis['concerns']:
        report += """
═══════════════════════════════════════════════════════════════
⚠️ CONCERNS
═══════════════════════════════════════════════════════════════
"""
        for concern in analysis['concerns']:
            report += f"• {concern}\n"

    report += """
═══════════════════════════════════════════════════════════════
🔧 ACTIONS TAKEN
═══════════════════════════════════════════════════════════════
"""

    for action in analysis['actions_taken']:
        report += f"• {action}\n"

    # Add hyperparameter details if available
    if 'hyperparameters' in tuning_data:
        report += """
═══════════════════════════════════════════════════════════════
⚙️ HYPERPARAMETER OPTIMIZATION
═══════════════════════════════════════════════════════════════
"""
        hp = tuning_data['hyperparameters']
        report += f"""
• Learning Rate: {hp.get('learning_rate', 'N/A')}
• Max Depth: {hp.get('max_depth', 'N/A')}
• Regularization: {hp.get('regularization', 'N/A')}
• Feature Selection: {hp.get('n_features', 'N/A')} features
• Cross-Validation Folds: {hp.get('cv_folds', 5)}
"""

    # Add top performing symbols
    if 'top_performers' in tuning_data:
        report += """
═══════════════════════════════════════════════════════════════
🏆 TOP PERFORMING SYMBOLS
═══════════════════════════════════════════════════════════════
"""
        for symbol in tuning_data['top_performers'][:5]:
            report += f"• {symbol['symbol']}: {symbol['accuracy']:.1%} accuracy ({symbol['predictions']} predictions)\n"

    # Add worst performing symbols for attention
    if 'worst_performers' in tuning_data:
        report += """
═══════════════════════════════════════════════════════════════
📉 SYMBOLS NEEDING ATTENTION
═══════════════════════════════════════════════════════════════
"""
        for symbol in tuning_data['worst_performers'][:3]:
            report += f"• {symbol['symbol']}: {symbol['accuracy']:.1%} accuracy ({symbol['predictions']} predictions)\n"

    # Add recommendations
    if analysis['recommendations']:
        report += """
═══════════════════════════════════════════════════════════════
💡 RECOMMENDATIONS
═══════════════════════════════════════════════════════════════
"""
        for rec in analysis['recommendations']:
            report += f"• {rec}\n"

    # Add next steps
    report += """
═══════════════════════════════════════════════════════════════
🔄 NEXT STEPS
═══════════════════════════════════════════════════════════════

• Next scheduled tuning: """ + (datetime.utcnow() + timedelta(days=7)).strftime('%Y-%m-%d') + """ (in 7 days)
• Monitor prediction accuracy daily
• Review symbol-specific performance
• Consider manual tuning if accuracy drops below 60%

═══════════════════════════════════════════════════════════════
📊 HISTORICAL PERFORMANCE (Last 5 Tuning Sessions)
═══════════════════════════════════════════════════════════════
"""

    # Add historical data
    history = get_tuning_history(model_type, 5)
    for session in history:
        date = session.get('timestamp', 'Unknown')
        acc = session.get('accuracy', 0)
        report += f"• {date}: {acc:.1%} accuracy\n"

    report += """
═══════════════════════════════════════════════════════════════

This is an automated report from the Stock Analytics Engine.
For questions or manual tuning requests, please check the system dashboard.
"""

    return report

def send_tuning_report(model_type: str, report: str, summary: Dict):
    """
    Send tuning report via SNS
    """
    try:
        subject = f"🤖 Model Tuning Report: {model_type.title()} Model - {summary.get('status', 'Completed')}"

        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=report,
            MessageAttributes={
                'report_type': {'DataType': 'String', 'StringValue': 'model_tuning'},
                'model_type': {'DataType': 'String', 'StringValue': model_type},
                'accuracy': {'DataType': 'Number', 'StringValue': str(summary.get('accuracy', 0))},
                'status': {'DataType': 'String', 'StringValue': summary.get('status', 'completed')}
            }
        )

        logger.info(f"Tuning report sent successfully: {response['MessageId']}")
        return response['MessageId']

    except Exception as e:
        logger.error(f"Error sending tuning report: {str(e)}")
        raise

def store_tuning_results(model_type: str, tuning_data: Dict, analysis: Dict):
    """
    Store tuning results in DynamoDB for historical tracking
    """
    try:
        table = dynamodb.Table(TUNING_HISTORY_TABLE)

        item = {
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat(),
            'tuning_id': f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'before_accuracy': Decimal(str(tuning_data.get('before_accuracy', 0))),
            'after_accuracy': Decimal(str(tuning_data.get('after_accuracy', 0))),
            'predictions_analyzed': tuning_data.get('predictions_analyzed', 0),
            'improvements': analysis.get('improvements', []),
            'concerns': analysis.get('concerns', []),
            'actions_taken': analysis.get('actions_taken', []),
            'hyperparameters': tuning_data.get('hyperparameters', {}),
            'ttl': int((datetime.utcnow() + timedelta(days=90)).timestamp())  # 90 day retention
        }

        table.put_item(Item=item)
        logger.info(f"Tuning results stored: {item['tuning_id']}")

    except Exception as e:
        logger.error(f"Error storing tuning results: {str(e)}")

def lambda_handler(event, context):
    """
    Main handler for model tuning reporter

    Event structure:
    {
        "model_type": "price" or "time",
        "tuning_data": {
            "before_accuracy": 0.65,
            "after_accuracy": 0.68,
            "predictions_analyzed": 1000,
            "training_samples": 800,
            "validation_samples": 200,
            "hyperparameters": {...},
            "top_performers": [...],
            "worst_performers": [...],
            "feature_changes": [...],
            "performance_metrics": {...}
        },
        "source": "price-model-tuning" or "time-model-tuning"
    }
    """
    try:
        logger.info(f"Tuning reporter triggered: {json.dumps(event)}")

        # Extract model type and tuning data
        model_type = event.get('model_type', 'unknown')
        tuning_data = event.get('tuning_data', {})

        # If this is a direct invocation from tuning function, extract from response
        if 'Payload' in event:
            payload = json.loads(event['Payload'])
            if 'body' in payload:
                body = json.loads(payload['body'])
                tuning_data = body.get('tuning_results', {})
                model_type = body.get('model_type', model_type)

        # Get current metrics
        current_metrics = get_tuning_metrics(model_type)
        tuning_data.update(current_metrics)

        # Analyze results
        analysis = analyze_tuning_results(tuning_data)

        # Determine overall status
        if analysis['concerns'] and len(analysis['concerns']) > 2:
            status = "⚠️ Needs Attention"
        elif analysis['improvements'] and len(analysis['improvements']) > len(analysis['concerns']):
            status = "✅ Successful"
        else:
            status = "✓ Completed"

        # Format report
        report = format_tuning_report(model_type, tuning_data, analysis)

        # Prepare summary
        summary = {
            'status': status,
            'accuracy': tuning_data.get('after_accuracy', 0),
            'improvement': tuning_data.get('after_accuracy', 0) - tuning_data.get('before_accuracy', 0),
            'predictions_analyzed': tuning_data.get('predictions_analyzed', 0)
        }

        # Send report
        message_id = send_tuning_report(model_type, report, summary)

        # Store results
        store_tuning_results(model_type, tuning_data, analysis)

        # Log success
        logger.info(f"Tuning report completed for {model_type} model: {status}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Tuning report sent successfully',
                'model_type': model_type,
                'status': status,
                'message_id': message_id,
                'summary': summary
            }, default=decimal_default)
        }

    except Exception as e:
        logger.error(f"Error in tuning reporter: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to generate tuning report'
            })
        }