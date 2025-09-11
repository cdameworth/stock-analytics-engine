"""
Stock Analytics Report Sender
Fetches analytics from reporting API and sends formatted email reports
"""

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
sns = boto3.client('sns')

# Environment variables
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
REPORTING_API_URL = os.environ.get('REPORTING_API_URL')

def lambda_handler(event, context):
    """
    Fetch analytics and send formatted email report
    """
    try:
        report_type = event.get('report_type', 'morning')
        timeframe = event.get('timeframe', '24h')
        
        logger.info(f"Generating {report_type} report for timeframe: {timeframe}")
        
        # Fetch analytics from reporting API
        analytics_data = fetch_analytics_data()
        
        # Generate formatted report based on type
        if report_type == 'morning':
            subject, message = format_morning_report(analytics_data)
        else:
            subject, message = format_evening_report(analytics_data)
        
        # Send email via SNS
        send_email_report(subject, message)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'{report_type} report sent successfully',
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error sending report: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Report sending failed',
                'details': str(e)
            })
        }

def fetch_analytics_data():
    """Fetch analytics from the reporting API"""
    try:
        http = urllib3.PoolManager()
        
        # Fetch dashboard data
        response = http.request(
            'GET',
            f"{REPORTING_API_URL}dual-predictions/analytics?days=30",
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status == 200:
            return json.loads(response.data.decode('utf-8'))
        else:
            logger.error(f"API returned status {response.status}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        return None

def format_morning_report(data):
    """Format morning validation report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    if not data:
        return (
            "⚠️ Stock Analytics Morning Report - Data Unavailable",
            "Unable to fetch analytics data. Please check the system."
        )
    
    summary = data.get('executive_summary', {})
    price_analytics = data.get('detailed_analytics', {}).get('price_analytics', {})
    time_analytics = data.get('detailed_analytics', {}).get('time_analytics', {})
    key_metrics = data.get('key_metrics', {})
    
    subject = f"📊 Stock Analytics Morning Report - {datetime.utcnow().strftime('%B %d, %Y')}"
    
    message = f"""
═══════════════════════════════════════════
📊 STOCK ANALYTICS MORNING VALIDATION REPORT
═══════════════════════════════════════════
Generated: {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Predictions: {summary.get('total_predictions', 0):,}
• Price Model Accuracy: {summary.get('price_model_accuracy', 0):.1%}
• Time Model Accuracy: {summary.get('time_model_accuracy', 0):.1%}
• System Status: {summary.get('system_status', 'Unknown').replace('_', ' ').title()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 PRICE PREDICTION MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prediction Counts:
• Total Generated: {price_analytics.get('prediction_counts', {}).get('total_generated', 0):,}
• Total Validated: {price_analytics.get('prediction_counts', {}).get('total_validated', 0):,}
• Buy Signals: {price_analytics.get('prediction_counts', {}).get('buy_predictions', 0)}
• Sell Signals: {price_analytics.get('prediction_counts', {}).get('sell_predictions', 0)}
• Hold Signals: {price_analytics.get('prediction_counts', {}).get('hold_predictions', 0)}

Accuracy Metrics (±5% tolerance):
• Overall: {price_analytics.get('accuracy_metrics', {}).get('overall_accuracy', 0):.1%}
• Buy Accuracy: {price_analytics.get('accuracy_metrics', {}).get('buy_accuracy', 0):.1%}
• Sell Accuracy: {price_analytics.get('accuracy_metrics', {}).get('sell_accuracy', 0):.1%}
• Hold Accuracy: {price_analytics.get('accuracy_metrics', {}).get('hold_accuracy', 0):.1%}

Performance:
• Best Performing: {price_analytics.get('performance_summary', {}).get('best_performing_type', 'N/A').upper()}
• Avg Confidence: {price_analytics.get('performance_summary', {}).get('average_confidence', 0):.1%}
• Trend: {price_analytics.get('performance_summary', {}).get('accuracy_trend', 'unknown').replace('_', ' ').title()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ TIME PREDICTION MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prediction Counts:
• Total Generated: {time_analytics.get('prediction_counts', {}).get('total_generated', 0):,}
• Total Validated: {time_analytics.get('prediction_counts', {}).get('total_validated', 0):,}
• Short Term (≤7d): {time_analytics.get('prediction_counts', {}).get('short_term_predictions', 0)}
• Medium Term (8-30d): {time_analytics.get('prediction_counts', {}).get('medium_term_predictions', 0)}
• Long Term (>30d): {time_analytics.get('prediction_counts', {}).get('long_term_predictions', 0)}

Accuracy Metrics (±20% tolerance):
• Overall: {time_analytics.get('accuracy_metrics', {}).get('overall_accuracy', 0):.1%}
• Short Term: {time_analytics.get('accuracy_metrics', {}).get('short_term_accuracy', 0):.1%}
• Medium Term: {time_analytics.get('accuracy_metrics', {}).get('medium_term_accuracy', 0):.1%}
• Long Term: {time_analytics.get('accuracy_metrics', {}).get('long_term_accuracy', 0):.1%}

Timeline Analysis:
• Avg Predicted Days: {time_analytics.get('timeline_analysis', {}).get('average_predicted_days', 0):.1f}
• Avg Actual Days: {time_analytics.get('timeline_analysis', {}).get('average_actual_days', 0):.1f}
• Bias: {time_analytics.get('timeline_analysis', {}).get('timeline_bias', 'unknown').replace('_', ' ').title()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TODAY'S ACTIVITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Price Predictions Today: {key_metrics.get('price_predictions_today', 0)}
• Time Predictions Today: {key_metrics.get('time_predictions_today', 0)}
• Improvement Trend: {key_metrics.get('accuracy_improvement_trend', {}).get('overall_system_trend', 'unknown').replace('_', ' ').title()}
• Next Tuning: {format_next_tuning(key_metrics.get('next_scheduled_tuning', {}))}

═══════════════════════════════════════════
📧 Report generated by Stock Analytics Engine
View full dashboard at the reporting API endpoint
═══════════════════════════════════════════
"""
    
    return subject, message

def format_evening_report(data):
    """Format evening summary report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    if not data:
        return (
            "⚠️ Stock Analytics Evening Report - Data Unavailable",
            "Unable to fetch analytics data. Please check the system."
        )
    
    summary = data.get('executive_summary', {})
    key_metrics = data.get('key_metrics', {})
    tuning_info = data.get('detailed_analytics', {}).get('tuning_analytics', {}).get('tuning_summary', {})
    
    subject = f"🌙 Stock Analytics Evening Summary - {datetime.utcnow().strftime('%B %d, %Y')}"
    
    message = f"""
═══════════════════════════════════════════
🌙 STOCK ANALYTICS EVENING SUMMARY
═══════════════════════════════════════════
Generated: {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TODAY'S PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Price Predictions Generated: {key_metrics.get('price_predictions_today', 0)}
• Time Predictions Generated: {key_metrics.get('time_predictions_today', 0)}
• Total Daily Predictions: {key_metrics.get('price_predictions_today', 0) + key_metrics.get('time_predictions_today', 0)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 30-DAY PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Predictions: {summary.get('total_predictions', 0):,}
• Price Model Accuracy: {summary.get('price_model_accuracy', 0):.1%}
• Time Model Accuracy: {summary.get('time_model_accuracy', 0):.1%}
• Recent Tuning Sessions: {summary.get('recent_tuning_sessions', 0)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 MODEL TUNING STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Tuning Sessions: {tuning_info.get('total_tuning_sessions', 0)}
• Price Model Sessions: {tuning_info.get('price_model_sessions', 0)}
• Time Model Sessions: {tuning_info.get('time_model_sessions', 0)}

Last Price Model Tuning:
{format_tuning_session(tuning_info.get('last_price_tuning', {}))}

Last Time Model Tuning:
{format_tuning_session(tuning_info.get('last_time_tuning', {}))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 SYSTEM STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Status: {summary.get('system_status', 'Unknown').replace('_', ' ').title()}
• Overall Trend: {key_metrics.get('accuracy_improvement_trend', {}).get('overall_system_trend', 'unknown').replace('_', ' ').title()}
• Next Scheduled Tuning: {format_next_tuning(key_metrics.get('next_scheduled_tuning', {}))}

═══════════════════════════════════════════
🌙 End of day summary complete
Have a great evening!
═══════════════════════════════════════════
"""
    
    return subject, message

def format_tuning_session(session_data):
    """Format tuning session information"""
    if session_data.get('status') == 'no_sessions_found':
        return "  • No tuning sessions found"
    
    return f"""  • Session ID: {session_data.get('session_id', 'N/A')[:8]}...
  • Timestamp: {format_timestamp(session_data.get('timestamp', ''))}
  • Steps Completed: {session_data.get('steps_completed', 0)}"""

def format_next_tuning(tuning_data):
    """Format next tuning schedule"""
    if not tuning_data:
        return "Not scheduled"
    
    weekly = tuning_data.get('next_weekly_tuning', '')
    if weekly:
        try:
            dt = datetime.fromisoformat(weekly.replace('Z', '+00:00'))
            return dt.strftime('%B %d at %I:%M %p UTC')
        except:
            return "Schedule unavailable"
    return "Not scheduled"

def format_timestamp(iso_timestamp):
    """Format ISO timestamp to readable format"""
    if not iso_timestamp:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M UTC')
    except:
        return iso_timestamp

def send_email_report(subject, message):
    """Send email report via SNS"""
    try:
        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        
        logger.info(f"Report sent successfully. MessageId: {response['MessageId']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise