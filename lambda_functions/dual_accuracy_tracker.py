"""
Dual Accuracy Tracker - Measures Price and Time Prediction Accuracy
Validates predictions against actual market outcomes and maintains accuracy metrics
"""

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal
import yfinance as yf

# Week 3: Advanced observability imports
try:
    from shared.observability_intelligence import (
        get_performance_monitor, get_trading_intelligence
    )
    from shared.business_tracing import get_financial_tracer
    ADVANCED_OBSERVABILITY_AVAILABLE = True
except ImportError:
    ADVANCED_OBSERVABILITY_AVAILABLE = False
    logger.warning("Advanced observability modules not available")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
PRICE_PREDICTIONS_TABLE = os.environ.get('PRICE_PREDICTIONS_TABLE', 'price-predictions')
TIME_PREDICTIONS_TABLE = os.environ.get('TIME_PREDICTIONS_TABLE', 'time-to-hit-predictions')
ACCURACY_METRICS_TABLE = os.environ.get('ACCURACY_METRICS_TABLE', 'prediction-accuracy-metrics')

def lambda_handler(event, context):
    """
    Dual accuracy tracking handler with Week 3 advanced observability

    Modes:
    - validate_price_predictions: Check price prediction accuracy
    - validate_time_predictions: Check time prediction accuracy
    - generate_accuracy_report: Create comprehensive accuracy report
    """
    # Week 3: Initialize advanced observability
    if ADVANCED_OBSERVABILITY_AVAILABLE:
        tracer = get_financial_tracer("dual_accuracy_tracker")
        performance_monitor = get_performance_monitor("dual_accuracy_tracker")
        trading_intelligence = get_trading_intelligence()

    try:
        action = event.get('action', 'validate_all')
        lookback_days = int(event.get('lookback_days', 30))

        logger.info(f"Dual accuracy tracking - Action: {action}, Lookback: {lookback_days} days")

        # Week 3: Create main tracking span
        if ADVANCED_OBSERVABILITY_AVAILABLE:
            with tracer.start_financial_span("accuracy_tracking.validation_cycle") as main_span:
                main_span.set_attributes({
                    "accuracy.action": action,
                    "accuracy.lookback_days": lookback_days,
                    "accuracy.validation_timestamp": datetime.utcnow().isoformat()
                })
                results = execute_accuracy_tracking(action, lookback_days, performance_monitor, trading_intelligence)
        else:
            results = execute_accuracy_tracking(action, lookback_days, None, None)
        
        return results
        
        return {
            'statusCode': 200,
            'body': json.dumps(results, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error in dual accuracy tracking: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Accuracy tracking failed',
                'message': str(e)
            })
        }

def execute_accuracy_tracking(action, lookback_days, performance_monitor=None, trading_intelligence=None):
    """
    Execute accuracy tracking with Week 3 advanced observability integration
    """
    results = {}

    if action in ['validate_all', 'validate_price_predictions']:
        price_accuracy = validate_price_predictions(lookback_days, performance_monitor)
        results['price_accuracy'] = price_accuracy

    if action in ['validate_all', 'validate_time_predictions']:
        time_accuracy = validate_time_predictions(lookback_days, performance_monitor)
        results['time_accuracy'] = time_accuracy

    if action == 'generate_accuracy_report':
        report = generate_comprehensive_report(lookback_days, trading_intelligence)
        results['accuracy_report'] = report

    # Store aggregate metrics
    store_accuracy_metrics(results)

    # Send metrics to CloudWatch
    send_accuracy_metrics(results)

    # Week 3: Generate performance summary for optimization
    if performance_monitor and ADVANCED_OBSERVABILITY_AVAILABLE:
        performance_summary = performance_monitor.get_performance_summary(lookback_hours=lookback_days * 24)
        results['performance_summary'] = performance_summary

        # Generate market opportunity analysis
        if trading_intelligence and results.get('price_accuracy', {}).get('total_predictions', 0) > 0:
            # Extract symbols from validation results
            symbols = extract_validated_symbols(results)
            for symbol in symbols[:10]:  # Analyze top 10 symbols
                opportunity = trading_intelligence.analyze_market_opportunity(symbol, [])
                if opportunity['opportunity_score'] > 70:
                    if 'market_opportunities' not in results:
                        results['market_opportunities'] = []
                    results['market_opportunities'].append(opportunity)

    return results

def extract_validated_symbols(results):
    """Extract symbol list from validation results"""
    symbols = set()

    # This would need to be enhanced based on actual data structure
    # For now, return common symbols for demo
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'INTC', 'NFLX']

def validate_price_predictions(lookback_days: int, performance_monitor=None) -> Dict:
    """
    Validate price predictions against actual market prices
    """
    try:
        table = dynamodb.Table(PRICE_PREDICTIONS_TABLE)
        
        # Query predictions ready for validation
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        response = table.scan(
            FilterExpression='validation_date <= :cutoff AND accuracy_measured = :false',
            ExpressionAttributeValues={
                ':cutoff': datetime.utcnow().isoformat(),
                ':false': False
            }
        )
        
        predictions = response['Items']
        accurate_predictions = 0
        total_predictions = len(predictions)
        
        for prediction in predictions:
            symbol = prediction['symbol']
            predicted_price = float(prediction['predicted_price'])
            validation_date = prediction['validation_date']
            confidence_score = float(prediction.get('confidence_score', 0.5))

            # Get actual price at validation date
            actual_price = get_historical_price(symbol, validation_date)

            if actual_price:
                # Calculate accuracy (within ±5% tolerance)
                price_accuracy = abs(predicted_price - actual_price) / actual_price
                is_accurate = price_accuracy <= 0.05  # 5% tolerance

                if is_accurate:
                    accurate_predictions += 1

                # Week 3: Track ML accuracy with advanced observability
                if performance_monitor and ADVANCED_OBSERVABILITY_AVAILABLE:
                    prediction_timestamp = datetime.fromisoformat(prediction.get('timestamp', validation_date))
                    actual_timestamp = datetime.fromisoformat(validation_date.replace('Z', '+00:00'))

                    accuracy_span = performance_monitor.track_ml_accuracy(
                        symbol=symbol,
                        predicted_price=predicted_price,
                        actual_price=actual_price,
                        confidence_score=confidence_score,
                        prediction_timestamp=prediction_timestamp,
                        actual_timestamp=actual_timestamp
                    )
                    accuracy_span.end()

                # Update prediction with accuracy result
                table.update_item(
                    Key={'prediction_id': prediction['prediction_id']},
                    UpdateExpression='SET accuracy_measured = :true, actual_price = :actual, accuracy_pct = :accuracy',
                    ExpressionAttributeValues={
                        ':true': True,
                        ':actual': Decimal(str(actual_price)),
                        ':accuracy': Decimal(str(price_accuracy))
                    }
                )
        
        accuracy_rate = (accurate_predictions / total_predictions) if total_predictions > 0 else 0
        
        price_accuracy_result = {
            'model_type': 'price_prediction',
            'total_predictions': total_predictions,
            'accurate_predictions': accurate_predictions,
            'accuracy_rate': round(accuracy_rate, 3),
            'tolerance': '±5%',
            'validation_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Price prediction accuracy: {accuracy_rate:.1%} ({accurate_predictions}/{total_predictions})")
        return price_accuracy_result
        
    except Exception as e:
        logger.error(f"Error validating price predictions: {str(e)}")
        return {'error': str(e)}

def validate_time_predictions(lookback_days: int, performance_monitor=None) -> Dict:
    """
    Validate time predictions against actual time-to-hit targets
    """
    try:
        table = dynamodb.Table(TIME_PREDICTIONS_TABLE)
        
        # Query time predictions ready for validation
        response = table.scan(
            FilterExpression='prediction_date <= :cutoff AND accuracy_measured = :false',
            ExpressionAttributeValues={
                ':cutoff': (datetime.utcnow() - timedelta(days=lookback_days)).isoformat(),
                ':false': False
            }
        )
        
        predictions = response['Items']
        accurate_predictions = 0
        total_predictions = len(predictions)
        
        for prediction in predictions:
            symbol = prediction['symbol']
            target_price = float(prediction['target_price'])
            predicted_days = int(prediction['predicted_days'])
            prediction_date = prediction['prediction_date']
            
            # Check if target price was hit and when
            actual_days = check_time_to_hit(symbol, target_price, prediction_date)
            
            if actual_days is not None:
                # Calculate accuracy (within ±20% tolerance)
                time_accuracy = abs(predicted_days - actual_days) / predicted_days
                is_accurate = time_accuracy <= 0.20  # 20% tolerance
                
                if is_accurate:
                    accurate_predictions += 1

                # Week 3: Track trading signal quality if available
                if performance_monitor and ADVANCED_OBSERVABILITY_AVAILABLE:
                    signal_span = performance_monitor.track_trading_signal_quality(
                        symbol=symbol,
                        signal_type="time_prediction",
                        signal_strength=float(prediction.get('confidence_score', 0.5)),
                        market_response_hours=actual_days * 24,  # Convert days to hours
                        profit_loss_pct=None  # Not available for time predictions
                    )
                    signal_span.end()

                # Update prediction with accuracy result
                table.update_item(
                    Key={'prediction_id': prediction['prediction_id']},
                    UpdateExpression='SET accuracy_measured = :true, actual_days = :actual, accuracy_pct = :accuracy',
                    ExpressionAttributeValues={
                        ':true': True,
                        ':actual': actual_days,
                        ':accuracy': Decimal(str(time_accuracy))
                    }
                )
        
        accuracy_rate = (accurate_predictions / total_predictions) if total_predictions > 0 else 0
        
        time_accuracy_result = {
            'model_type': 'time_prediction',
            'total_predictions': total_predictions,
            'accurate_predictions': accurate_predictions,
            'accuracy_rate': round(accuracy_rate, 3),
            'tolerance': '±20%',
            'validation_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Time prediction accuracy: {accuracy_rate:.1%} ({accurate_predictions}/{total_predictions})")
        return time_accuracy_result
        
    except Exception as e:
        logger.error(f"Error validating time predictions: {str(e)}")
        return {'error': str(e)}

def get_historical_price(symbol: str, target_date: str) -> Optional[float]:
    """Get historical price for validation"""
    try:
        stock = yf.Ticker(symbol)
        target_dt = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
        
        # Get data around target date (±3 days buffer)
        start_date = target_dt - timedelta(days=3)
        end_date = target_dt + timedelta(days=3)
        
        hist = stock.history(start=start_date, end=end_date)
        
        if not hist.empty:
            # Use closest available trading day
            closest_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
            return float(closest_price) if closest_price else None
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting historical price for {symbol}: {str(e)}")
        return None

def check_time_to_hit(symbol: str, target_price: float, prediction_date: str) -> Optional[int]:
    """Check actual days to hit target price"""
    try:
        stock = yf.Ticker(symbol)
        start_dt = datetime.fromisoformat(prediction_date.replace('Z', '+00:00'))
        end_dt = datetime.utcnow()
        
        hist = stock.history(start=start_dt, end=end_dt)
        
        if hist.empty:
            return None
        
        # Check each day if target was hit
        for i, (date, row) in enumerate(hist.iterrows()):
            if row['High'] >= target_price or row['Low'] <= target_price:
                return i + 1  # Days to hit (1-indexed)
        
        # Target not hit yet
        return None
        
    except Exception as e:
        logger.error(f"Error checking time to hit for {symbol}: {str(e)}")
        return None

def store_accuracy_metrics(results: Dict):
    """Store aggregate accuracy metrics"""
    try:
        table = dynamodb.Table(ACCURACY_METRICS_TABLE)
        
        timestamp = datetime.utcnow().isoformat()
        
        for model_type, accuracy_data in results.items():
            if isinstance(accuracy_data, dict) and 'accuracy_rate' in accuracy_data:
                table.put_item(Item={
                    'metric_id': f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    'model_type': model_type,
                    'accuracy_rate': Decimal(str(accuracy_data['accuracy_rate'])),
                    'total_predictions': accuracy_data['total_predictions'],
                    'accurate_predictions': accuracy_data['accurate_predictions'],
                    'timestamp': timestamp,
                    'validation_period': accuracy_data.get('validation_period', 'Unknown')
                })
        
        logger.info(f"Stored accuracy metrics for {len(results)} model types")
        
    except Exception as e:
        logger.error(f"Error storing accuracy metrics: {str(e)}")

def send_accuracy_metrics(results: Dict):
    """Send accuracy metrics to CloudWatch"""
    try:
        metric_data = []
        
        for model_type, accuracy_data in results.items():
            if isinstance(accuracy_data, dict) and 'accuracy_rate' in accuracy_data:
                metric_data.extend([
                    {
                        'MetricName': 'ModelAccuracy',
                        'Dimensions': [
                            {'Name': 'ModelType', 'Value': model_type}
                        ],
                        'Value': accuracy_data['accuracy_rate'],
                        'Unit': 'Percent'
                    },
                    {
                        'MetricName': 'PredictionCount',
                        'Dimensions': [
                            {'Name': 'ModelType', 'Value': model_type}
                        ],
                        'Value': accuracy_data['total_predictions'],
                        'Unit': 'Count'
                    }
                ])
        
        if metric_data:
            cloudwatch.put_metric_data(
                Namespace='StockAnalytics/DualAccuracy',
                MetricData=metric_data
            )
            
    except Exception as e:
        logger.error(f"Error sending accuracy metrics: {str(e)}")

def generate_comprehensive_report(lookback_days: int, trading_intelligence=None) -> Dict:
    """Generate comprehensive accuracy report for both models"""
    try:
        # Get latest accuracy metrics for both models
        table = dynamodb.Table(ACCURACY_METRICS_TABLE)
        
        # Query recent metrics
        cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        response = table.scan(
            FilterExpression='#ts >= :cutoff',
            ExpressionAttributeNames={
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':cutoff': cutoff_date
            }
        )
        
        metrics = response['Items']
        
        # Aggregate by model type
        price_metrics = [m for m in metrics if m['model_type'] == 'price_accuracy']
        time_metrics = [m for m in metrics if m['model_type'] == 'time_accuracy']
        
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'lookback_period': f'{lookback_days} days',
            'price_prediction_summary': aggregate_metrics(price_metrics),
            'time_prediction_summary': aggregate_metrics(time_metrics),
            'combined_summary': {
                'total_predictions': sum([m.get('total_predictions', 0) for m in metrics]),
                'overall_accuracy': calculate_weighted_accuracy(metrics),
                'model_count': 2,
                'active_models': ['price_prediction', 'time_prediction']
            }
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}")
        return {'error': str(e)}

def aggregate_metrics(metrics: List[Dict]) -> Dict:
    """Aggregate metrics for a single model type"""
    if not metrics:
        return {
            'total_predictions': 0,
            'average_accuracy': 0.0,
            'accuracy_trend': 'insufficient_data'
        }
    
    total_predictions = sum([int(m['total_predictions']) for m in metrics])
    avg_accuracy = sum([float(m['accuracy_rate']) for m in metrics]) / len(metrics)
    
    # Calculate trend (last 3 vs previous 3 measurements)
    if len(metrics) >= 6:
        recent_avg = sum([float(m['accuracy_rate']) for m in metrics[-3:]]) / 3
        previous_avg = sum([float(m['accuracy_rate']) for m in metrics[-6:-3]]) / 3
        trend = 'improving' if recent_avg > previous_avg else 'declining'
    else:
        trend = 'insufficient_data'
    
    return {
        'total_predictions': total_predictions,
        'average_accuracy': round(avg_accuracy, 3),
        'accuracy_trend': trend,
        'measurement_count': len(metrics)
    }

def calculate_weighted_accuracy(all_metrics: List[Dict]) -> float:
    """Calculate overall weighted accuracy across both model types"""
    total_weighted_sum = 0
    total_predictions = 0
    
    for metric in all_metrics:
        accuracy = float(metric['accuracy_rate'])
        count = int(metric['total_predictions'])
        total_weighted_sum += accuracy * count
        total_predictions += count
    
    return round(total_weighted_sum / total_predictions, 3) if total_predictions > 0 else 0.0

def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")