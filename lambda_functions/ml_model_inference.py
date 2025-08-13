"""
ML Model Inference Lambda Function
Uses SageMaker endpoint to generate stock recommendations
"""

import json
import boto3
import logging
from datetime import datetime, timedelta
from decimal import Decimal
#import numpy as np
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Configuration
SAGEMAKER_ENDPOINT = os.environ['SAGEMAKER_ENDPOINT']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
S3_BUCKET = os.environ['S3_BUCKET']

def lambda_handler(event, context):
    """
    Main Lambda handler function
    """
    try:
        logger.info("Starting ML model inference process")
        
        # Extract data from event
        index_data = event.get('index_data', [])
        stock_data = event.get('stock_data', [])
        
        if not index_data or not stock_data:
            logger.warning("Insufficient data for ML inference")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Insufficient market data'})
            }
        
        # Prepare features for ML model
        features = prepare_ml_features(index_data, stock_data)
        
        # Get predictions from SageMaker endpoint
        predictions = get_ml_predictions(features)
        
        # Generate stock recommendations
        recommendations = generate_recommendations(predictions, stock_data)
        
        # Store recommendations in DynamoDB
        stored_count = store_recommendations(recommendations)
        
        # Store analysis results in S3
        store_analysis_results(recommendations, features)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'ML inference completed successfully',
                'recommendations_generated': len(recommendations),
                'recommendations_stored': stored_count,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def prepare_ml_features(index_data, stock_data):
    """
    Prepare features for ML model input
    """
    try:
        features = []
        
        # Market sentiment features from indexes
        market_features = calculate_market_sentiment(index_data)
        
        # Individual stock features
        for stock in stock_data:
            stock_features = {
                'symbol': stock['symbol'],
                'price': float(stock['close']),
                'volume': stock['volume'],
                'moving_avg_5': float(stock['moving_avg_5']),
                'moving_avg_20': float(stock['moving_avg_20']),
                'volatility': float(stock['volatility']),
                'price_to_ma5_ratio': float(stock['close'] / stock['moving_avg_5']),
                'price_to_ma20_ratio': float(stock['close'] / stock['moving_avg_20']),
                'volume_normalized': normalize_volume(stock['volume'], stock_data),
                'market_correlation': calculate_market_correlation(stock, index_data)
            }
            
            # Combine with market features
            stock_features.update(market_features)
            features.append(stock_features)
        
        logger.info(f"Prepared features for {len(features)} stocks")
        return features
        
    except Exception as e:
        logger.error(f"Error preparing ML features: {str(e)}")
        return []

def calculate_market_sentiment(index_data):
    """
    Calculate market sentiment from index data
    """
    try:
        if not index_data:
            return {'market_sentiment': 0.5}
        
        # Calculate average change across indexes
        changes = []
        for index in index_data:
            change = (float(index['close']) - float(index['open'])) / float(index['open'])
            changes.append(change)
        
        avg_change = sum(changes) / len(changes)
        
        # Calculate volatility across indexes
        volatilities = [float(index['volatility']) for index in index_data]
        avg_volatility = sum(volatilities) / len(volatilities)
        
        # Normalize sentiment (0 = bearish, 1 = bullish)
        sentiment = 0.5 + (avg_change * 10)  # Scale and center
        sentiment = max(0, min(1, sentiment))  # Clamp between 0 and 1
        
        return {
            'market_sentiment': sentiment,
            'market_volatility': avg_volatility,
            'market_change': avg_change,
            'index_count': len(index_data)
        }
        
    except Exception as e:
        logger.error(f"Error calculating market sentiment: {str(e)}")
        return {'market_sentiment': 0.5}

def normalize_volume(volume, all_stocks):
    """
    Normalize volume relative to other stocks
    """
    try:
        volumes = [stock['volume'] for stock in all_stocks]
        max_volume = max(volumes)
        min_volume = min(volumes)
        
        if max_volume == min_volume:
            return 0.5
        
        return (volume - min_volume) / (max_volume - min_volume)
        
    except Exception as e:
        logger.error(f"Error normalizing volume: {str(e)}")
        return 0.5

def calculate_market_correlation(stock, index_data):
    """
    Calculate correlation between stock and market indexes
    """
    try:
        if not index_data:
            return 0.5
        
        stock_change = (float(stock['close']) - float(stock['open'])) / float(stock['open'])
        
        correlations = []
        for index in index_data:
            index_change = (float(index['close']) - float(index['open'])) / float(index['open'])
            # Simple correlation approximation
            correlation = 1 if (stock_change * index_change) > 0 else 0
            correlations.append(correlation)
        
        return sum(correlations) / len(correlations)
        
    except Exception as e:
        logger.error(f"Error calculating market correlation: {str(e)}")
        return 0.5

def get_ml_predictions(features):
    """
    Get predictions from SageMaker endpoint
    """
    try:
        predictions = []
        
        for feature_set in features:
            # Prepare input for SageMaker model
            model_input = [
                feature_set['price_to_ma5_ratio'],
                feature_set['price_to_ma20_ratio'],
                feature_set['volatility'],
                feature_set['volume_normalized'],
                feature_set['market_sentiment'],
                feature_set['market_volatility'],
                feature_set['market_correlation']
            ]
            
            try:
                # Call SageMaker endpoint
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=SAGEMAKER_ENDPOINT,
                    ContentType='application/json',
                    Body=json.dumps({'instances': [model_input]})
                )
                
                # Parse response
                result = json.loads(response['Body'].read().decode())
                prediction_score = result.get('predictions', [0.5])[0]
                
                predictions.append({
                    'symbol': feature_set['symbol'],
                    'prediction_score': prediction_score,
                    'confidence': calculate_confidence(feature_set, prediction_score),
                    'features': feature_set
                })
                
            except Exception as e:
                logger.error(f"Error getting prediction for {feature_set['symbol']}: {str(e)}")
                # Fallback to rule-based prediction
                fallback_score = generate_fallback_prediction(feature_set)
                predictions.append({
                    'symbol': feature_set['symbol'],
                    'prediction_score': fallback_score,
                    'confidence': 0.6,
                    'features': feature_set,
                    'fallback': True
                })
        
        logger.info(f"Generated predictions for {len(predictions)} stocks")
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting ML predictions: {str(e)}")
        return []

def generate_fallback_prediction(features):
    """
    Generate rule-based prediction when ML model is unavailable
    """
    try:
        score = 0.5  # Neutral baseline
        
        # Technical analysis rules
        if features['price_to_ma5_ratio'] > 1.02:  # Price above 5-day MA
            score += 0.1
        if features['price_to_ma20_ratio'] > 1.05:  # Price above 20-day MA
            score += 0.15
        if features['market_sentiment'] > 0.6:  # Bullish market
            score += 0.1
        if features['volatility'] < 0.02:  # Low volatility
            score += 0.05
        if features['volume_normalized'] > 0.7:  # High volume
            score += 0.1
        
        return max(0, min(1, score))
        
    except Exception as e:
        logger.error(f"Error generating fallback prediction: {str(e)}")
        return 0.5

def calculate_confidence(features, prediction_score):
    """
    Calculate confidence level for prediction
    """
    try:
        confidence = 0.7  # Base confidence
        
        # Adjust based on market conditions
        if features['market_volatility'] < 0.02:
            confidence += 0.1
        elif features['market_volatility'] > 0.05:
            confidence -= 0.1
        
        # Adjust based on technical indicators alignment
        if abs(features['price_to_ma5_ratio'] - 1) < 0.05:  # Price close to MA
            confidence -= 0.05
        
        return max(0.5, min(0.95, confidence))
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return 0.7

def generate_recommendations(predictions, stock_data):
    """
    Generate final stock recommendations
    """
    try:
        recommendations = []
        
        # Sort by prediction score
        sorted_predictions = sorted(predictions, key=lambda x: x['prediction_score'], reverse=True)
        
        for i, pred in enumerate(sorted_predictions):
            # Find corresponding stock data
            stock_info = next((s for s in stock_data if s['symbol'] == pred['symbol']), None)
            if not stock_info:
                continue
            
            # Determine recommendation type
            score = pred['prediction_score']
            if score >= 0.7:
                recommendation_type = 'BUY'
                risk_level = 'LOW' if pred['confidence'] > 0.8 else 'MEDIUM'
            elif score >= 0.6:
                recommendation_type = 'HOLD'
                risk_level = 'MEDIUM'
            else:
                recommendation_type = 'SELL'
                risk_level = 'HIGH'
            
            # Calculate target price (simple model)
            current_price = float(stock_info['close'])
            price_change_prediction = (score - 0.5) * 0.2  # Max 10% change
            target_price = current_price * (1 + price_change_prediction)
            
            recommendation = {
                'recommendation_id': f"{pred['symbol']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'symbol': pred['symbol'],
                'timestamp': datetime.utcnow().isoformat(),
                'recommendation_type': recommendation_type,
                'prediction_score': Decimal(str(pred['prediction_score'])),
                'confidence': Decimal(str(pred['confidence'])),
                'current_price': Decimal(str(current_price)),
                'target_price': Decimal(str(target_price)),
                'risk_level': risk_level,
                'ranking': i + 1,
                'rationale': generate_rationale(pred, stock_info),
                'metadata': {
                    'model_version': '1.0',
                    'features_used': list(pred['features'].keys()),
                    'market_conditions': pred['features'].get('market_sentiment', 0.5),
                    'fallback_used': pred.get('fallback', False)
                }
            }
            
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} final recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []

def generate_rationale(prediction, stock_info):
    """
    Generate human-readable rationale for recommendation
    """
    try:
        features = prediction['features']
        score = prediction['prediction_score']
        
        rationale_parts = []
        
        # Technical indicators
        if features['price_to_ma5_ratio'] > 1.02:
            rationale_parts.append("trading above short-term moving average")
        elif features['price_to_ma5_ratio'] < 0.98:
            rationale_parts.append("trading below short-term moving average")
        
        # Market sentiment
        if features['market_sentiment'] > 0.6:
            rationale_parts.append("positive market sentiment")
        elif features['market_sentiment'] < 0.4:
            rationale_parts.append("negative market sentiment")
        
        # Volatility
        if features['volatility'] < 0.02:
            rationale_parts.append("low volatility environment")
        elif features['volatility'] > 0.05:
            rationale_parts.append("high volatility environment")
        
        # Volume
        if features['volume_normalized'] > 0.7:
            rationale_parts.append("above-average trading volume")
        
        if rationale_parts:
            return f"Recommendation based on {', '.join(rationale_parts)}."
        else:
            return "Recommendation based on technical analysis and market conditions."
        
    except Exception as e:
        logger.error(f"Error generating rationale: {str(e)}")
        return "Recommendation based on quantitative analysis."

def store_recommendations(recommendations):
    """
    Store recommendations in DynamoDB
    """
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        stored_count = 0
        
        for rec in recommendations:
            try:
                table.put_item(Item=rec)
                stored_count += 1
            except Exception as e:
                logger.error(f"Error storing recommendation for {rec['symbol']}: {str(e)}")
                continue
        
        logger.info(f"Stored {stored_count} recommendations in DynamoDB")
        return stored_count
        
    except Exception as e:
        logger.error(f"Error storing recommendations: {str(e)}")
        return 0

def store_analysis_results(recommendations, features):
    """
    Store analysis results in S3 for audit and debugging
    """
    try:
        analysis_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations_count': len(recommendations),
            'features_analyzed': len(features),
            'recommendations': json.loads(json.dumps(recommendations, default=decimal_default)),
            'features': features,
            'analysis_metadata': {
                'model_version': '1.0',
                'lambda_function': 'ml-model-inference',
                'analysis_type': 'daily_stock_recommendations'
            }
        }
        
        key = f"ml-analysis/{datetime.utcnow().strftime('%Y/%m/%d')}/recommendations_{datetime.utcnow().strftime('%H%M%S')}.json"
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(analysis_data, indent=2),
            ContentType='application/json',
            Metadata={
                'analysis_type': 'stock_recommendations',
                'timestamp': datetime.utcnow().isoformat(),
                'recommendations_count': str(len(recommendations))
            }
        )
        
        logger.info(f"Stored analysis results in S3: {key}")
        
    except Exception as e:
        logger.error(f"Error storing analysis results: {str(e)}")

def decimal_default(obj):
    """
    JSON serializer for Decimal objects
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")