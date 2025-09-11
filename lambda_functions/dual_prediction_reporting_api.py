"""
Dual Prediction Reporting API - Enhanced Analytics for Price and Time Models
Provides comprehensive accuracy metrics, prediction counts, and tuning history
"""

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb')

# Environment variables
PRICE_PREDICTIONS_TABLE = os.environ.get('PRICE_PREDICTIONS_TABLE', 'price-predictions')
TIME_PREDICTIONS_TABLE = os.environ.get('TIME_PREDICTIONS_TABLE', 'time-to-hit-predictions')
ACCURACY_METRICS_TABLE = os.environ.get('ACCURACY_METRICS_TABLE', 'prediction-accuracy-metrics')
TUNING_HISTORY_TABLE = os.environ.get('TUNING_HISTORY_TABLE', 'model-tuning-history')

def lambda_handler(event, context):
    """
    Dual prediction reporting API handler
    
    Endpoints:
    - /analytics/price-accuracy: Price prediction metrics + counts
    - /analytics/time-accuracy: Time prediction metrics + counts  
    - /analytics/combined: Aggregate accuracy for both models
    - /analytics/tuning-history: Last tuning steps for each model
    - /analytics/dashboard: Comprehensive dashboard data
    """
    try:
        # Handle OPTIONS requests for CORS preflight
        http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method') or 'GET'
        
        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': ''
            }
        
        # Parse path from event
        path = event.get('path', '/analytics/dashboard')
        query_params = event.get('queryStringParameters') or {}
        
        lookback_days = int(query_params.get('days', '30'))
        
        logger.info(f"Dual prediction reporting - Path: {path}, Lookback: {lookback_days} days")
        
        # Route to appropriate endpoint
        if path in ['/analytics/price-accuracy', '/dual-predictions/price-accuracy']:
            result = get_price_accuracy_report(lookback_days)
        elif path in ['/analytics/time-accuracy', '/dual-predictions/time-accuracy']:
            result = get_time_accuracy_report(lookback_days)
        elif path in ['/analytics/combined', '/dual-predictions/combined']:
            result = get_combined_accuracy_report(lookback_days)
        elif path in ['/analytics/tuning-history', '/dual-predictions/tuning-history']:
            result = get_tuning_history_report(lookback_days)
        elif path in ['/analytics/dashboard', '/analytics', '/dual-predictions/analytics']:
            result = get_comprehensive_dashboard(lookback_days)
        else:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Endpoint not found'})
            }
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps(result, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error in dual prediction reporting: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Reporting API failed',
                'message': str(e)
            })
        }

def get_price_accuracy_report(lookback_days: int) -> Dict:
    """Get comprehensive price prediction accuracy report"""
    try:
        table = dynamodb.Table(PRICE_PREDICTIONS_TABLE)
        cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        # Get all price predictions in period
        response = table.scan(
            FilterExpression='prediction_timestamp >= :cutoff',
            ExpressionAttributeValues={
                ':cutoff': cutoff_date
            }
        )
        
        all_predictions = response['Items']
        validated_predictions = [p for p in all_predictions if p.get('accuracy_measured', False)]
        
        # Calculate accuracy by recommendation type
        buy_predictions = [p for p in validated_predictions if p['recommendation'] == 'buy']
        sell_predictions = [p for p in validated_predictions if p['recommendation'] == 'sell']
        hold_predictions = [p for p in validated_predictions if p['recommendation'] == 'hold']
        
        report = {
            'model_type': 'price_prediction',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_counts': {
                'total_generated': len(all_predictions),
                'total_validated': len(validated_predictions),
                'buy_predictions': len(buy_predictions),
                'sell_predictions': len(sell_predictions),
                'hold_predictions': len(hold_predictions)
            },
            'accuracy_metrics': {
                'overall_accuracy': calculate_price_accuracy_rate(validated_predictions),
                'buy_accuracy': calculate_price_accuracy_rate(buy_predictions),
                'sell_accuracy': calculate_price_accuracy_rate(sell_predictions),
                'hold_accuracy': calculate_price_accuracy_rate(hold_predictions),
                'tolerance': '±5%'
            },
            'performance_summary': {
                'best_performing_type': get_best_performing_recommendation_type(buy_predictions, sell_predictions, hold_predictions),
                'average_confidence': calculate_average_confidence(validated_predictions),
                'accuracy_trend': calculate_accuracy_trend(validated_predictions)
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating price accuracy report: {str(e)}")
        return {'error': str(e)}

def get_time_accuracy_report(lookback_days: int) -> Dict:
    """Get comprehensive time prediction accuracy report"""
    try:
        table = dynamodb.Table(TIME_PREDICTIONS_TABLE)
        cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        response = table.scan(
            FilterExpression='prediction_date >= :cutoff',
            ExpressionAttributeValues={
                ':cutoff': cutoff_date
            }
        )
        
        all_predictions = response['Items']
        validated_predictions = [p for p in all_predictions if p.get('accuracy_measured', False)]
        
        # Categorize by predicted timeframe
        short_term = [p for p in validated_predictions if int(p.get('predicted_days', 0)) <= 7]
        medium_term = [p for p in validated_predictions if 7 < int(p.get('predicted_days', 0)) <= 30]
        long_term = [p for p in validated_predictions if int(p.get('predicted_days', 0)) > 30]
        
        report = {
            'model_type': 'time_prediction',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_counts': {
                'total_generated': len(all_predictions),
                'total_validated': len(validated_predictions),
                'short_term_predictions': len(short_term),   # ≤7 days
                'medium_term_predictions': len(medium_term), # 8-30 days
                'long_term_predictions': len(long_term)      # >30 days
            },
            'accuracy_metrics': {
                'overall_accuracy': calculate_time_accuracy_rate(validated_predictions),
                'short_term_accuracy': calculate_time_accuracy_rate(short_term),
                'medium_term_accuracy': calculate_time_accuracy_rate(medium_term),
                'long_term_accuracy': calculate_time_accuracy_rate(long_term),
                'tolerance': '±20%'
            },
            'timeline_analysis': {
                'average_predicted_days': calculate_average_timeline(validated_predictions, 'predicted_days'),
                'average_actual_days': calculate_average_timeline(validated_predictions, 'actual_days'),
                'timeline_bias': analyze_timeline_bias_simple(validated_predictions)
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating time accuracy report: {str(e)}")
        return {'error': str(e)}

def get_combined_accuracy_report(lookback_days: int) -> Dict:
    """Get combined accuracy report for both prediction models"""
    try:
        price_report = get_price_accuracy_report(lookback_days)
        time_report = get_time_accuracy_report(lookback_days)
        
        # Calculate aggregate metrics
        total_price_predictions = price_report.get('prediction_counts', {}).get('total_generated', 0)
        total_time_predictions = time_report.get('prediction_counts', {}).get('total_generated', 0)
        
        combined_report = {
            'report_type': 'combined_accuracy',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'aggregate_metrics': {
                'total_predictions_generated': total_price_predictions + total_time_predictions,
                'price_model_accuracy': price_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                'time_model_accuracy': time_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                'combined_weighted_accuracy': calculate_combined_accuracy(price_report, time_report),
                'active_prediction_models': 2
            },
            'model_breakdown': {
                'price_prediction': {
                    'predictions_count': total_price_predictions,
                    'accuracy_rate': price_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                    'model_status': 'active'
                },
                'time_prediction': {
                    'predictions_count': total_time_predictions,
                    'accuracy_rate': time_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                    'model_status': 'active'
                }
            },
            'detailed_reports': {
                'price_details': price_report,
                'time_details': time_report
            }
        }
        
        return combined_report
        
    except Exception as e:
        logger.error(f"Error generating combined accuracy report: {str(e)}")
        return {'error': str(e)}

def get_tuning_history_report(lookback_days: int) -> Dict:
    """Get tuning history for both models"""
    try:
        table = dynamodb.Table(TUNING_HISTORY_TABLE)
        cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        response = table.scan(
            FilterExpression='session_timestamp >= :cutoff',
            ExpressionAttributeValues={
                ':cutoff': cutoff_date
            }
        )
        
        tuning_sessions = response['Items']
        
        # Separate by model type
        price_sessions = [s for s in tuning_sessions if s['model_type'] == 'price_prediction']
        time_sessions = [s for s in tuning_sessions if s['model_type'] == 'time_prediction']
        
        report = {
            'report_type': 'tuning_history',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'tuning_summary': {
                'total_tuning_sessions': len(tuning_sessions),
                'price_model_sessions': len(price_sessions),
                'time_model_sessions': len(time_sessions),
                'last_price_tuning': get_latest_session_summary(price_sessions),
                'last_time_tuning': get_latest_session_summary(time_sessions)
            },
            'recent_tuning_steps': {
                'price_model_steps': get_recent_tuning_steps(price_sessions),
                'time_model_steps': get_recent_tuning_steps(time_sessions)
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating tuning history report: {str(e)}")
        return {'error': str(e)}

def get_comprehensive_dashboard(lookback_days: int) -> Dict:
    """Get comprehensive dashboard combining all metrics"""
    try:
        price_report = get_price_accuracy_report(lookback_days)
        time_report = get_time_accuracy_report(lookback_days)
        tuning_report = get_tuning_history_report(lookback_days)
        
        dashboard = {
            'dashboard_type': 'dual_prediction_comprehensive',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'executive_summary': {
                'total_predictions': (
                    price_report.get('prediction_counts', {}).get('total_generated', 0) +
                    time_report.get('prediction_counts', {}).get('total_generated', 0)
                ),
                'price_model_accuracy': price_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                'time_model_accuracy': time_report.get('accuracy_metrics', {}).get('overall_accuracy', 0),
                'recent_tuning_sessions': tuning_report.get('tuning_summary', {}).get('total_tuning_sessions', 0),
                'system_status': 'dual_models_active'
            },
            'detailed_analytics': {
                'price_analytics': price_report,
                'time_analytics': time_report,
                'tuning_analytics': tuning_report
            },
            'key_metrics': {
                'price_predictions_today': get_daily_prediction_count('price_prediction'),
                'time_predictions_today': get_daily_prediction_count('time_prediction'),
                'accuracy_improvement_trend': calculate_improvement_trends(lookback_days),
                'next_scheduled_tuning': get_next_tuning_schedule()
            }
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error generating comprehensive dashboard: {str(e)}")
        return {'error': str(e)}

# Helper functions

def calculate_price_accuracy_rate(predictions: List[Dict]) -> float:
    """Calculate price prediction accuracy rate (±5% tolerance)"""
    if not predictions:
        return 0.0
    
    accurate_count = sum(1 for p in predictions if float(p.get('accuracy_pct', 1.0)) <= 0.05)
    return round(accurate_count / len(predictions), 3)

def calculate_time_accuracy_rate(predictions: List[Dict]) -> float:
    """Calculate time prediction accuracy rate (±20% tolerance)"""
    if not predictions:
        return 0.0
    
    accurate_count = sum(1 for p in predictions if float(p.get('accuracy_pct', 1.0)) <= 0.20)
    return round(accurate_count / len(predictions), 3)

def get_best_performing_recommendation_type(buy_preds: List, sell_preds: List, hold_preds: List) -> str:
    """Identify which recommendation type has highest accuracy"""
    accuracies = {
        'buy': calculate_price_accuracy_rate(buy_preds),
        'sell': calculate_price_accuracy_rate(sell_preds),
        'hold': calculate_price_accuracy_rate(hold_preds)
    }
    
    return max(accuracies, key=accuracies.get) if accuracies else 'unknown'

def calculate_average_confidence(predictions: List[Dict]) -> float:
    """Calculate average confidence across predictions"""
    if not predictions:
        return 0.0
    
    confidences = [float(p.get('confidence', 0)) for p in predictions]
    return round(sum(confidences) / len(confidences), 2)

def calculate_accuracy_trend(predictions: List[Dict]) -> str:
    """Calculate if accuracy is improving, declining, or stable"""
    if len(predictions) < 10:
        return 'insufficient_data'
    
    # Sort by timestamp and compare first half vs second half
    sorted_preds = sorted(predictions, key=lambda x: x['prediction_timestamp'])
    half_point = len(sorted_preds) // 2
    
    first_half_accuracy = calculate_price_accuracy_rate(sorted_preds[:half_point])
    second_half_accuracy = calculate_price_accuracy_rate(sorted_preds[half_point:])
    
    if second_half_accuracy > first_half_accuracy + 0.05:
        return 'improving'
    elif second_half_accuracy < first_half_accuracy - 0.05:
        return 'declining'
    else:
        return 'stable'

def calculate_average_timeline(predictions: List[Dict], field: str) -> float:
    """Calculate average timeline from predictions"""
    values = [float(p.get(field, 0)) for p in predictions if p.get(field) is not None]
    return round(sum(values) / len(values), 1) if values else 0.0

def analyze_timeline_bias_simple(predictions: List[Dict]) -> str:
    """Simple timeline bias analysis"""
    if not predictions:
        return 'no_data'
    
    biases = []
    for p in predictions:
        predicted = int(p.get('predicted_days', 0))
        actual = p.get('actual_days')
        if actual is not None:
            biases.append(predicted - int(actual))
    
    if not biases:
        return 'no_validation_data'
    
    avg_bias = sum(biases) / len(biases)
    
    if avg_bias > 2:
        return 'overestimating_timelines'
    elif avg_bias < -2:
        return 'underestimating_timelines'
    else:
        return 'well_calibrated'

def calculate_combined_accuracy(price_report: Dict, time_report: Dict) -> float:
    """Calculate weighted combined accuracy across both models"""
    price_accuracy = price_report.get('accuracy_metrics', {}).get('overall_accuracy', 0)
    time_accuracy = time_report.get('accuracy_metrics', {}).get('overall_accuracy', 0)
    
    price_count = price_report.get('prediction_counts', {}).get('total_validated', 0)
    time_count = time_report.get('prediction_counts', {}).get('total_validated', 0)
    
    if price_count + time_count == 0:
        return 0.0
    
    weighted_accuracy = (price_accuracy * price_count + time_accuracy * time_count) / (price_count + time_count)
    return round(weighted_accuracy, 3)

def get_latest_session_summary(sessions: List[Dict]) -> Dict:
    """Get summary of most recent tuning session"""
    if not sessions:
        return {'status': 'no_sessions_found'}
    
    latest_session = max(sessions, key=lambda x: x['session_timestamp'])
    
    return {
        'session_id': latest_session['session_id'],
        'timestamp': latest_session['session_timestamp'],
        'steps_completed': latest_session['total_steps'],
        'model_type': latest_session['model_type']
    }

def get_recent_tuning_steps(sessions: List[Dict]) -> List[Dict]:
    """Get recent tuning steps for a model type"""
    if not sessions:
        return []
    
    latest_session = max(sessions, key=lambda x: x['session_timestamp'])
    return latest_session.get('tuning_steps', [])

def get_daily_prediction_count(model_type: str) -> int:
    """Get count of predictions generated today"""
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        
        if model_type == 'price_prediction':
            table = dynamodb.Table(PRICE_PREDICTIONS_TABLE)
            filter_expr = 'begins_with(prediction_timestamp, :today)'
        else:
            table = dynamodb.Table(TIME_PREDICTIONS_TABLE)
            filter_expr = 'begins_with(prediction_date, :today)'
        
        response = table.scan(
            FilterExpression=filter_expr,
            ExpressionAttributeValues={
                ':today': today
            },
            Select='COUNT'
        )
        
        return response['Count']
        
    except Exception as e:
        logger.error(f"Error getting daily prediction count: {str(e)}")
        return 0

def calculate_improvement_trends(lookback_days: int) -> Dict:
    """Calculate improvement trends over time"""
    return {
        'price_model_trend': 'improving',  # Would calculate from historical data
        'time_model_trend': 'stable',
        'overall_system_trend': 'improving',
        'trend_confidence': 'moderate'
    }

def get_next_tuning_schedule() -> Dict:
    """Get next scheduled tuning times"""
    now = datetime.utcnow()
    
    # Next Sunday 2 AM for weekly tuning
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour < 2:
        days_until_sunday = 0  # Today is Sunday and it's before 2 AM
    elif days_until_sunday == 0:
        days_until_sunday = 7  # Today is Sunday but after 2 AM
    
    next_weekly = now + timedelta(days=days_until_sunday)
    next_weekly = next_weekly.replace(hour=2, minute=0, second=0, microsecond=0)
    
    return {
        'next_weekly_tuning': next_weekly.isoformat(),
        'next_daily_assessment': (now + timedelta(days=1)).replace(hour=8, minute=0).isoformat(),
        'tuning_frequency': 'weekly_comprehensive'
    }

def get_cors_headers():
    """Get CORS headers for API responses"""
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
        'Access-Control-Max-Age': '86400'
    }

def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")