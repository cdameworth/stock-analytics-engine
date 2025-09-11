"""
Machine Learning utilities for Stock Analytics Engine.
Provides common ML operations, model management, and prediction utilities.
"""

import json
import logging
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from decimal import Decimal

from .lambda_utils import AWSClients, MetricsHelper, DynamoDBHelper
from .config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class PredictionResult:
    """Standardized prediction result container."""
    
    def __init__(self, symbol: str, prediction_type: str):
        self.symbol = symbol
        self.prediction_type = prediction_type
        self.timestamp = datetime.utcnow().isoformat()
        self.confidence = 0.0
        self.factors = []
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'prediction_type': self.prediction_type,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'factors': self.factors,
            'metadata': self.metadata
        }


class PricePredictionResult(PredictionResult):
    """Price prediction specific result."""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'price')
        self.target_price = 0.0
        self.current_price = 0.0
        self.recommendation = 'hold'
        self.price_range = {'low': 0.0, 'high': 0.0}
        self.timeframe_days = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'target_price': self.target_price,
            'current_price': self.current_price,
            'recommendation': self.recommendation,
            'price_range': self.price_range,
            'timeframe_days': self.timeframe_days
        })
        return base_dict


class TimePredictionResult(PredictionResult):
    """Time prediction specific result."""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'time')
        self.expected_timeline = 'unknown'
        self.probability_ranges = {}
        self.target_price = 0.0
        self.current_price = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'expected_timeline': self.expected_timeline,
            'probability_ranges': self.probability_ranges,
            'target_price': self.target_price,
            'current_price': self.current_price
        })
        return base_dict


class TechnicalIndicators:
    """Technical analysis indicators calculator."""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    @staticmethod
    def calculate_moving_average(prices: List[float], period: int) -> float:
        """Calculate simple moving average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> float:
        """Calculate price volatility (standard deviation)."""
        if len(prices) < 2:
            return 0.0
        
        recent_prices = prices[-period:] if len(prices) >= period else prices
        
        if len(recent_prices) < 2:
            return 0.0
        
        return statistics.stdev(recent_prices) / statistics.mean(recent_prices)
    
    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 10) -> float:
        """Calculate price momentum."""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        return (current_price - past_price) / past_price


class MarketDataProcessor:
    """Process and analyze market data for predictions."""
    
    def __init__(self):
        self.s3_client = AWSClients.get_client('s3')
        self.data_bucket = config.s3.data_lake_bucket
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve historical data for a symbol."""
        try:
            # Try to get data from S3
            key = f"stock_data/{symbol.upper()}/daily_data.json"
            response = self.s3_client.get_object(Bucket=self.data_bucket, Key=key)
            data = json.loads(response['Body'].read())
            
            # Sort by date and return recent data
            sorted_data = sorted(data, key=lambda x: x.get('date', ''), reverse=True)
            return sorted_data[:days]
            
        except Exception as e:
            logger.warning(f"Could not retrieve historical data for {symbol}: {str(e)}")
            return []
    
    def extract_price_series(self, historical_data: List[Dict[str, Any]]) -> List[float]:
        """Extract closing prices from historical data."""
        prices = []
        for data_point in historical_data:
            try:
                close_price = float(data_point.get('close', 0))
                if close_price > 0:
                    prices.append(close_price)
            except (ValueError, TypeError):
                continue
        
        return list(reversed(prices))  # Return in chronological order
    
    def calculate_technical_indicators(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Calculate technical indicators for a symbol."""
        historical_data = self.get_historical_data(symbol, 60)  # Get 60 days of data
        prices = self.extract_price_series(historical_data)
        
        if not prices:
            # Return default indicators if no historical data
            return {
                'rsi': 50.0,
                'ma_5': current_price,
                'ma_20': current_price,
                'volatility': 0.02,
                'momentum': 0.0,
                'price_to_ma5_ratio': 1.0,
                'price_to_ma20_ratio': 1.0
            }
        
        # Add current price to the series
        prices.append(current_price)
        
        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(prices)
        ma_5 = TechnicalIndicators.calculate_moving_average(prices, 5)
        ma_20 = TechnicalIndicators.calculate_moving_average(prices, 20)
        volatility = TechnicalIndicators.calculate_volatility(prices)
        momentum = TechnicalIndicators.calculate_momentum(prices)
        
        return {
            'rsi': rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'volatility': volatility,
            'momentum': momentum,
            'price_to_ma5_ratio': current_price / ma_5 if ma_5 > 0 else 1.0,
            'price_to_ma20_ratio': current_price / ma_20 if ma_20 > 0 else 1.0
        }


class PredictionEngine:
    """Base class for prediction engines."""
    
    def __init__(self, prediction_type: str):
        self.prediction_type = prediction_type
        self.market_processor = MarketDataProcessor()
        self.metrics_helper = MetricsHelper(f"StockAnalytics/{prediction_type.title()}Prediction")
    
    def generate_prediction(self, symbol: str, current_price: float, **kwargs) -> PredictionResult:
        """Generate prediction - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_prediction")
    
    def calculate_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate prediction confidence based on various factors."""
        # Base confidence calculation
        confidence_factors = []
        
        # Technical strength (RSI, momentum, etc.)
        rsi = factors.get('rsi', 50)
        if 30 <= rsi <= 70:  # Neutral RSI is more reliable
            confidence_factors.append(0.7)
        elif rsi < 30 or rsi > 70:  # Extreme RSI can be reliable for reversals
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Volatility factor (lower volatility = higher confidence)
        volatility = factors.get('volatility', 0.02)
        if volatility < 0.02:
            confidence_factors.append(0.8)
        elif volatility < 0.05:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Trend consistency
        momentum = factors.get('momentum', 0)
        price_to_ma5 = factors.get('price_to_ma5_ratio', 1.0)
        price_to_ma20 = factors.get('price_to_ma20_ratio', 1.0)
        
        trend_consistency = abs(momentum) * (abs(price_to_ma5 - 1) + abs(price_to_ma20 - 1))
        if trend_consistency > 0.1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average
        base_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Add some randomness to simulate model uncertainty
        import random
        random.seed(hash(symbol) % 1000)  # Deterministic randomness based on symbol
        uncertainty = random.uniform(-0.1, 0.1)
        
        final_confidence = max(0.1, min(0.95, base_confidence + uncertainty))
        return round(final_confidence, 3)
    
    def send_prediction_metrics(self, symbol: str, prediction: PredictionResult) -> None:
        """Send prediction metrics to CloudWatch."""
        try:
            self.metrics_helper.put_metric(
                'PredictionGenerated',
                1,
                'Count',
                {'Symbol': symbol, 'PredictionType': self.prediction_type}
            )
            
            self.metrics_helper.put_metric(
                'PredictionConfidence',
                prediction.confidence,
                'None',
                {'Symbol': symbol, 'PredictionType': self.prediction_type}
            )
            
        except Exception as e:
            logger.warning(f"Failed to send prediction metrics: {str(e)}")


def decimal_to_float(obj: Any) -> Any:
    """Convert Decimal objects to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
