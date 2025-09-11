"""
Price Prediction Model - Dedicated Price Target Predictions
Predicts target prices for buy/sell/hold recommendations with confidence scoring
"""

import json
import logging
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal

# Import shared utilities
from shared.lambda_utils import (
    setup_logger, LambdaResponse, handle_lambda_errors,
    AWSClients, MetricsHelper, DynamoDBHelper, InputValidator
)
from shared.config import get_config, FeatureFlags

# Configure logging
logger = setup_logger(__name__)
config = get_config()

# AWS clients (using centralized client management)
dynamodb = AWSClients.get_resource('dynamodb')
s3 = AWSClients.get_client('s3')
cloudwatch = AWSClients.get_client('cloudwatch')

# Initialize helpers
metrics_helper = MetricsHelper("StockAnalytics/PricePrediction")
db_helper = DynamoDBHelper(config.database.price_predictions_table)

@handle_lambda_errors(logger)
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Price prediction handler - generates target price predictions

    Input:
    {
        "symbol": "AAPL",
        "current_price": 150.00,
        "timeframe_days": 30,
        "technical_indicators": {...}
    }

    Output:
    {
        "target_price": 165.50,
        "recommendation": "buy",
        "confidence": 0.78,
        "price_range": {"low": 160.00, "high": 170.00},
        "factors": ["strong_earnings", "sector_rotation"]
    }
    """
    logger.info(f"Price prediction request: {json.dumps(event, default=str)}")

    # Check if this is a batch request from EventBridge or data ingestion
    if event.get('trigger_type') in ['scheduled', 'data_ingestion'] and 'symbols' in event:
        return handle_batch_predictions(event, context)

    # Parse and validate input for single symbol
    prediction_input = parse_prediction_input(event)

    # Generate price prediction
    prediction_result = generate_price_prediction(
        prediction_input['symbol'],
        prediction_input['current_price'],
        prediction_input['timeframe_days'],
        prediction_input['technical_indicators']
    )

    # Store prediction for accuracy tracking
    store_price_prediction(
        prediction_input['symbol'],
        prediction_input['current_price'],
        prediction_result,
        prediction_input['timeframe_days']
    )

    # Send metrics to CloudWatch
    if FeatureFlags.is_metrics_enabled():
        send_prediction_metrics(prediction_input['symbol'], prediction_result)

    return LambdaResponse.success(prediction_result)


def parse_prediction_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate prediction input from event."""
    # Parse input for single symbol
    if 'body' in event:
        body = json.loads(event['body']) if event.get('body') else {}
        symbol = body.get('symbol')
        current_price = body.get('current_price', 0)
        timeframe_days = body.get('timeframe_days', 30)
        technical_indicators = body.get('technical_indicators', {})
    else:
        symbol = event.get('symbol')
        current_price = event.get('current_price', 0)
        timeframe_days = event.get('timeframe_days', 30)
        technical_indicators = event.get('technical_indicators', {})

    # Validate inputs
    if not symbol:
        raise ValueError("Symbol is required")

    symbol = InputValidator.validate_symbol(symbol)
    current_price = InputValidator.validate_positive_number(current_price, "current_price")

    if not isinstance(timeframe_days, int) or timeframe_days <= 0:
        raise ValueError("timeframe_days must be a positive integer")

    return {
        'symbol': symbol,
        'current_price': current_price,
        'timeframe_days': timeframe_days,
        'technical_indicators': technical_indicators or {}
    }

def generate_price_prediction(symbol: str, current_price: float, timeframe_days: int, 
                            technical_indicators: Dict) -> Dict:
    """
    Generate price target prediction using technical and fundamental analysis
    """
    try:
        # Extract technical indicators
        rsi = technical_indicators.get('rsi', 50)
        macd = technical_indicators.get('macd', 0)
        bollinger_position = technical_indicators.get('bollinger_position', 0.5)
        volume_ratio = technical_indicators.get('volume_ratio', 1.0)
        
        # Fundamental factors (simplified)
        sector_strength = get_sector_strength(symbol)
        market_trend = get_market_trend()
        volatility = calculate_volatility(symbol)
        
        # Price prediction algorithm
        base_trend = calculate_base_trend(rsi, macd, bollinger_position)
        sector_adjustment = sector_strength * 0.02  # ±2% sector impact
        market_adjustment = market_trend * 0.015    # ±1.5% market impact
        volume_impact = min((volume_ratio - 1.0) * 0.01, 0.03)  # Volume boost up to 3%
        
        # Calculate total expected return
        total_expected_return = base_trend + sector_adjustment + market_adjustment + volume_impact
        
        # Apply timeframe scaling
        timeframe_multiplier = math.sqrt(timeframe_days / 30.0)  # Scale for timeframe
        adjusted_return = total_expected_return * timeframe_multiplier
        
        # Calculate target price
        target_price = current_price * (1 + adjusted_return)
        
        # Determine recommendation
        if adjusted_return > 0.05:  # >5% expected return
            recommendation = "buy"
        elif adjusted_return < -0.05:  # <-5% expected return
            recommendation = "sell"
        else:
            recommendation = "hold"
        
        # Calculate confidence based on signal strength
        signal_strength = abs(adjusted_return)
        confidence = min(0.5 + (signal_strength * 2), 0.95)  # 50-95% confidence range
        
        # Price range estimation (±volatility)
        price_range_pct = volatility * 0.5  # Half of volatility as range
        price_low = target_price * (1 - price_range_pct)
        price_high = target_price * (1 + price_range_pct)
        
        # Key factors driving prediction
        factors = []
        if abs(rsi - 50) > 20:
            factors.append("rsi_signal" if rsi < 30 else "rsi_overbought" if rsi > 70 else "rsi_neutral")
        if abs(macd) > 0.5:
            factors.append("macd_bullish" if macd > 0 else "macd_bearish")
        if sector_strength > 0.02:
            factors.append("sector_strength")
        elif sector_strength < -0.02:
            factors.append("sector_weakness")
        if volume_ratio > 1.5:
            factors.append("high_volume")
        
        prediction_result = {
            'symbol': symbol,
            'target_price': round(target_price, 2),
            'recommendation': recommendation,
            'confidence': round(confidence, 2),
            'expected_return': round(adjusted_return * 100, 2),  # As percentage
            'price_range': {
                'low': round(price_low, 2),
                'high': round(price_high, 2)
            },
            'factors': factors,
            'timeframe_days': timeframe_days,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_version': 'price_v1.0'
        }
        
        logger.info(f"Price prediction generated for {symbol}: {prediction_result}")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error generating price prediction: {str(e)}")
        raise e

def calculate_base_trend(rsi: float, macd: float, bollinger_position: float) -> float:
    """Calculate base trend from technical indicators"""
    # RSI contribution (-10% to +10%)
    if rsi < 30:
        rsi_signal = 0.08  # Oversold - bullish
    elif rsi > 70:
        rsi_signal = -0.06  # Overbought - bearish
    else:
        rsi_signal = (50 - rsi) / 1000  # Linear scaling around 50
    
    # MACD contribution (-5% to +5%)
    macd_signal = max(min(macd * 0.02, 0.05), -0.05)
    
    # Bollinger Band position (-3% to +3%)
    bollinger_signal = (bollinger_position - 0.5) * 0.06
    
    return rsi_signal + macd_signal + bollinger_signal

def get_sector_strength(symbol: str) -> float:
    """Get sector relative strength (simplified)"""
    # Sector mapping (simplified)
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA']
    healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK']
    
    # Simplified sector strength (would be enhanced with real data)
    if symbol in tech_stocks:
        return 0.01  # Tech slightly positive
    elif symbol in finance_stocks:
        return -0.005  # Finance slightly negative
    elif symbol in healthcare_stocks:
        return 0.015  # Healthcare positive
    else:
        return 0.0  # Neutral for others

def get_market_trend() -> float:
    """Get overall market trend (simplified)"""
    # Simplified market sentiment (would be enhanced with SPY/QQQ data)
    return 0.005  # Slightly positive market

def calculate_volatility(symbol: str) -> float:
    """Calculate stock volatility (simplified)"""
    # Simplified volatility by market cap (would be enhanced with real data)
    large_cap = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    if symbol in large_cap:
        return 0.15  # 15% volatility for large cap
    else:
        return 0.25  # 25% volatility for others

def store_price_prediction(symbol: str, current_price: float, prediction: Dict, 
                          timeframe_days: int):
    """Store prediction for accuracy tracking"""
    try:
        table = dynamodb.Table(PRICE_PREDICTIONS_TABLE)
        
        prediction_item = {
            'prediction_id': f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'current_price': Decimal(str(current_price)),
            'predicted_price': Decimal(str(prediction['target_price'])),
            'recommendation': prediction['recommendation'],
            'confidence': Decimal(str(prediction['confidence'])),
            'expected_return': Decimal(str(prediction['expected_return'])),
            'timeframe_days': timeframe_days,
            'factors': prediction['factors'],
            'model_version': prediction['model_version'],
            'validation_date': (datetime.utcnow() + timedelta(days=timeframe_days)).isoformat(),
            'accuracy_measured': False
        }
        
        table.put_item(Item=prediction_item)
        logger.info(f"Stored price prediction for {symbol}")
        
    except Exception as e:
        logger.error(f"Error storing price prediction: {str(e)}")

def send_prediction_metrics(symbol: str, prediction: Dict):
    """Send custom metrics to CloudWatch"""
    try:
        cloudwatch.put_metric_data(
            Namespace='StockAnalytics/PricePrediction',
            MetricData=[
                {
                    'MetricName': 'PredictionGenerated',
                    'Dimensions': [
                        {'Name': 'Symbol', 'Value': symbol},
                        {'Name': 'Recommendation', 'Value': prediction['recommendation']}
                    ],
                    'Value': 1,
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'PredictionConfidence', 
                    'Dimensions': [
                        {'Name': 'Symbol', 'Value': symbol}
                    ],
                    'Value': prediction['confidence'],
                    'Unit': 'None'
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error sending metrics: {str(e)}")

def handle_batch_predictions(event, context):
    """Handle batch predictions from scheduled events or data ingestion"""
    try:
        symbols = event.get('symbols', [])
        stock_data = event.get('stock_data', {})
        trigger_type = event.get('trigger_type', 'unknown')
        market_session = event.get('market_session', 'unknown')
        timeframe_days = 30
        
        logger.info(f"Processing batch predictions for {len(symbols)} symbols from {trigger_type} trigger")
        
        predictions = []
        success_count = 0
        
        for symbol in symbols:
            try:
                # Get current price from stock data or fallback to 0 if not available
                current_price = extract_current_price(symbol, stock_data)
                if current_price <= 0:
                    logger.warning(f"No valid price data for {symbol}, skipping")
                    continue
                
                # Generate prediction
                prediction_result = generate_price_prediction(
                    symbol, current_price, timeframe_days, {}
                )
                
                # Store prediction
                store_price_prediction(symbol, current_price, prediction_result, timeframe_days)
                
                # Add to results
                predictions.append({
                    'symbol': symbol,
                    'prediction': prediction_result,
                    'status': 'success'
                })
                
                success_count += 1
                logger.info(f"Generated prediction for {symbol} at ${current_price}")
                
            except Exception as e:
                logger.error(f"Failed to generate prediction for {symbol}: {str(e)}")
                predictions.append({
                    'symbol': symbol,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Send batch metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='StockAnalytics/PricePrediction',
            MetricData=[
                {
                    'MetricName': 'BatchPredictionsProcessed',
                    'Value': len(symbols),
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'TriggerType',
                            'Value': trigger_type
                        }
                    ]
                },
                {
                    'MetricName': 'BatchPredictionsSuccess',
                    'Value': success_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'TriggerType',
                            'Value': trigger_type
                        }
                    ]
                }
            ]
        )
        
        result = {
            'trigger_type': trigger_type,
            'market_session': market_session,
            'processed_count': len(symbols),
            'success_count': success_count,
            'failed_count': len(symbols) - success_count,
            'timestamp': datetime.utcnow().isoformat(),
            'predictions': predictions
        }
        
        logger.info(f"Batch processing complete: {success_count}/{len(symbols)} successful")
        
        return {
            'statusCode': 200,
            'body': json.dumps(result, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error in batch predictions: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Batch prediction failed',
                'message': str(e)
            })
        }

def extract_current_price(symbol: str, stock_data: Dict) -> float:
    """Extract current price from stock data"""
    try:
        if symbol in stock_data:
            # Stock data structure from Alpha Vantage
            symbol_data = stock_data[symbol]
            if 'close' in symbol_data:
                return float(symbol_data['close'])
            elif 'price' in symbol_data:
                return float(symbol_data['price'])
            elif '4. close' in symbol_data:  # Alpha Vantage format
                return float(symbol_data['4. close'])
        return 0.0
    except Exception as e:
        logger.error(f"Error extracting price for {symbol}: {str(e)}")
        return 0.0

def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")