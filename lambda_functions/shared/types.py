"""
Type definitions for Stock Analytics Engine.
Provides comprehensive type hints and data structures for better code safety and documentation.
"""

from typing import Dict, List, Optional, Union, Any, Literal, TypedDict, Protocol
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum


# Basic type aliases
Price = Union[float, Decimal]
Timestamp = Union[str, datetime]
Symbol = str
Confidence = float  # 0.0 to 1.0


# Enums for standardized values
class RecommendationType(Enum):
    """Stock recommendation types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionType(Enum):
    """Types of predictions."""
    PRICE = "price"
    TIME = "time"
    DIRECTION = "direction"
    VOLATILITY = "volatility"


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class TriggerType(Enum):
    """Event trigger types."""
    SCHEDULED = "scheduled"
    DATA_INGESTION = "data_ingestion"
    API_REQUEST = "api_request"
    MANUAL = "manual"


# TypedDict definitions for structured data
class TechnicalIndicators(TypedDict, total=False):
    """Technical analysis indicators."""
    rsi: float
    ma_5: float
    ma_20: float
    ma_50: float
    ma_200: float
    volatility: float
    momentum: float
    price_to_ma5_ratio: float
    price_to_ma20_ratio: float
    volume_ratio: float
    bollinger_upper: float
    bollinger_lower: float
    macd: float
    macd_signal: float


class PriceRange(TypedDict):
    """Price range definition."""
    low: Price
    high: Price


class MarketData(TypedDict):
    """Market data structure."""
    symbol: Symbol
    price: Price
    volume: int
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    change: float
    change_percent: float


class PredictionInput(TypedDict):
    """Input for prediction functions."""
    symbol: Symbol
    current_price: Price
    timeframe_days: int
    technical_indicators: TechnicalIndicators


class PricePredictionOutput(TypedDict):
    """Price prediction output structure."""
    symbol: Symbol
    target_price: Price
    current_price: Price
    recommendation: str
    confidence: Confidence
    price_range: PriceRange
    factors: List[str]
    timeframe_days: int
    timestamp: Timestamp


class TimePredictionOutput(TypedDict):
    """Time prediction output structure."""
    symbol: Symbol
    expected_timeline: str
    probability_ranges: Dict[str, float]
    target_price: Price
    current_price: Price
    confidence: Confidence
    factors: List[str]
    timestamp: Timestamp


class RecommendationData(TypedDict):
    """Stock recommendation data structure."""
    symbol: Symbol
    recommendation: str
    confidence: Confidence
    target_price: Price
    current_price: Price
    risk_level: str
    factors: List[str]
    technical_indicators: TechnicalIndicators
    timestamp: Timestamp
    expires_at: Timestamp


class PerformanceMetrics(TypedDict):
    """Performance metrics structure."""
    hit_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    total_return: float
    market_outperformance: float
    max_drawdown: float
    win_rate: float


class BacktestResult(TypedDict):
    """Backtest result structure."""
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    benchmark_return: float
    outperformance: float
    sharpe_ratio: float
    max_drawdown: float
    performance_metrics: PerformanceMetrics


class LambdaEvent(TypedDict, total=False):
    """Standard Lambda event structure."""
    body: Optional[str]
    headers: Optional[Dict[str, str]]
    queryStringParameters: Optional[Dict[str, str]]
    pathParameters: Optional[Dict[str, str]]
    requestContext: Optional[Dict[str, Any]]
    trigger_type: Optional[str]
    symbols: Optional[List[Symbol]]
    action: Optional[str]


class LambdaResponse(TypedDict):
    """Standard Lambda response structure."""
    statusCode: int
    headers: Dict[str, str]
    body: str


class ErrorResponse(TypedDict):
    """Error response structure."""
    success: bool
    error: Dict[str, Any]


class SuccessResponse(TypedDict):
    """Success response structure."""
    success: bool
    data: Any
    timestamp: str


# Dataclasses for complex objects
@dataclass
class StockRecommendation:
    """Stock recommendation data class."""
    symbol: Symbol
    recommendation: RecommendationType
    confidence: Confidence
    target_price: Price
    current_price: Price
    risk_level: RiskLevel
    factors: List[str]
    technical_indicators: TechnicalIndicators
    timestamp: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'recommendation': self.recommendation.value,
            'confidence': self.confidence,
            'target_price': float(self.target_price),
            'current_price': float(self.current_price),
            'risk_level': self.risk_level.value,
            'factors': self.factors,
            'technical_indicators': dict(self.technical_indicators),
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }


@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_name: str
    prediction_type: PredictionType
    hit_rate: float
    accuracy: float
    confidence_calibration: float
    last_updated: datetime
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'prediction_type': self.prediction_type.value,
            'hit_rate': self.hit_rate,
            'accuracy': self.accuracy,
            'confidence_calibration': self.confidence_calibration,
            'last_updated': self.last_updated.isoformat(),
            'sample_size': self.sample_size
        }


@dataclass
class TradingSignal:
    """Trading signal data class."""
    symbol: Symbol
    signal_type: RecommendationType
    strength: float  # 0.0 to 1.0
    entry_price: Price
    target_price: Price
    stop_loss: Price
    risk_reward_ratio: float
    timeframe: str
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'entry_price': float(self.entry_price),
            'target_price': float(self.target_price),
            'stop_loss': float(self.stop_loss),
            'risk_reward_ratio': self.risk_reward_ratio,
            'timeframe': self.timeframe,
            'generated_at': self.generated_at.isoformat()
        }


# Protocol definitions for interfaces
class PredictionModel(Protocol):
    """Protocol for prediction models."""
    
    def predict(self, input_data: PredictionInput) -> Union[PricePredictionOutput, TimePredictionOutput]:
        """Generate prediction from input data."""
        ...
    
    def get_confidence(self, input_data: PredictionInput) -> Confidence:
        """Calculate prediction confidence."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        ...


class DataProcessor(Protocol):
    """Protocol for data processors."""
    
    def process_market_data(self, raw_data: Dict[str, Any]) -> MarketData:
        """Process raw market data into standardized format."""
        ...
    
    def calculate_indicators(self, market_data: MarketData) -> TechnicalIndicators:
        """Calculate technical indicators from market data."""
        ...
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data integrity and completeness."""
        ...


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    def collect_prediction_metrics(self, prediction: Union[PricePredictionOutput, TimePredictionOutput]) -> None:
        """Collect metrics from predictions."""
        ...
    
    def collect_performance_metrics(self, performance: PerformanceMetrics) -> None:
        """Collect performance metrics."""
        ...
    
    def send_metrics(self) -> None:
        """Send collected metrics to monitoring system."""
        ...


# Utility type guards
def is_price_prediction(prediction: Dict[str, Any]) -> bool:
    """Type guard for price predictions."""
    return 'target_price' in prediction and 'recommendation' in prediction


def is_time_prediction(prediction: Dict[str, Any]) -> bool:
    """Type guard for time predictions."""
    return 'expected_timeline' in prediction and 'probability_ranges' in prediction


def is_valid_symbol(symbol: str) -> bool:
    """Validate stock symbol format."""
    return isinstance(symbol, str) and symbol.isalpha() and 1 <= len(symbol) <= 10


def is_valid_confidence(confidence: float) -> bool:
    """Validate confidence value."""
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0


def is_valid_price(price: Union[int, float, Decimal]) -> bool:
    """Validate price value."""
    try:
        return float(price) > 0
    except (ValueError, TypeError):
        return False


# Type conversion utilities
def ensure_float(value: Union[int, float, Decimal, str]) -> float:
    """Ensure value is converted to float."""
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def ensure_symbol(value: str) -> Symbol:
    """Ensure value is a valid symbol."""
    if not isinstance(value, str):
        raise ValueError("Symbol must be a string")
    
    symbol = value.upper().strip()
    if not is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    return symbol


def ensure_confidence(value: Union[int, float]) -> Confidence:
    """Ensure value is a valid confidence score."""
    conf = float(value)
    if not is_valid_confidence(conf):
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {conf}")
    
    return conf
