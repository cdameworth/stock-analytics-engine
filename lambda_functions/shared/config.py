"""
Configuration management for Stock Analytics Engine Lambda functions.
Centralizes environment variables, AWS resource names, and application settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from .lambda_utils import EnvConfig


@dataclass
class DatabaseConfig:
    """Database table configuration."""
    recommendations_table: str
    analytics_table: str
    competitive_table: str
    price_predictions_table: str
    time_predictions_table: str
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        return cls(
            recommendations_table=EnvConfig.get_optional('RECOMMENDATIONS_TABLE', 'stock-recommendations'),
            analytics_table=EnvConfig.get_optional('ANALYTICS_TABLE', 'ai-performance-analytics'),
            competitive_table=EnvConfig.get_optional('COMPETITIVE_TABLE', 'competitive-analysis'),
            price_predictions_table=EnvConfig.get_optional('PRICE_PREDICTIONS_TABLE', 'price-predictions'),
            time_predictions_table=EnvConfig.get_optional('TIME_PREDICTIONS_TABLE', 'time-to-hit-predictions')
        )


@dataclass
class S3Config:
    """S3 bucket configuration."""
    data_lake_bucket: str
    model_artifacts_bucket: str
    performance_bucket: str
    
    @classmethod
    def from_env(cls) -> 'S3Config':
        """Create S3 config from environment variables."""
        return cls(
            data_lake_bucket=EnvConfig.get_optional('S3_DATA_BUCKET', 'stock-analytics-data-lake'),
            model_artifacts_bucket=EnvConfig.get_optional('S3_MODEL_BUCKET', 'stock-analytics-ml-models'),
            performance_bucket=EnvConfig.get_optional('S3_PERFORMANCE_BUCKET', 'stock-analytics-model-performance')
        )


@dataclass
class APIConfig:
    """External API configuration."""
    alpha_vantage_secret_arn: str
    use_premium_api: bool
    premium_calls_per_minute: int
    connect_timeout: int
    per_call_timeout: int
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        return cls(
            alpha_vantage_secret_arn=EnvConfig.get_optional('ALPHA_VANTAGE_API_KEY_SECRET_ARN'),
            use_premium_api=EnvConfig.get_bool('USE_PREMIUM_API_KEY', False),
            premium_calls_per_minute=EnvConfig.get_int('PREMIUM_API_CALLS_PER_MINUTE', 5),
            connect_timeout=EnvConfig.get_int('CONNECT_TEST_TIMEOUT', 2),
            per_call_timeout=EnvConfig.get_int('PER_CALL_TIMEOUT', 6)
        )


@dataclass
class LambdaConfig:
    """Lambda function configuration."""
    ml_inference_function: str
    price_prediction_function: str
    time_prediction_function: str
    
    @classmethod
    def from_env(cls) -> 'LambdaConfig':
        """Create Lambda config from environment variables."""
        return cls(
            ml_inference_function=EnvConfig.get_optional('ML_INFERENCE_FUNCTION_NAME', 'ml-model-inference-tier'),
            price_prediction_function=EnvConfig.get_optional('PRICE_PREDICTION_FUNCTION', 'price-prediction-model'),
            time_prediction_function=EnvConfig.get_optional('TIME_PREDICTION_FUNCTION', 'time-to-hit-predictor')
        )


@dataclass
class CacheConfig:
    """Cache configuration."""
    valkey_endpoint: Optional[str]
    default_ttl: int
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create cache config from environment variables."""
        return cls(
            valkey_endpoint=EnvConfig.get_optional('VALKEY_ENDPOINT') or EnvConfig.get_optional('REDIS_ENDPOINT'),
            default_ttl=EnvConfig.get_int('CACHE_DEFAULT_TTL', 300)
        )


@dataclass
class NotificationConfig:
    """Notification configuration."""
    sns_topic_arn: str
    
    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Create notification config from environment variables."""
        return cls(
            sns_topic_arn=EnvConfig.get_optional('SNS_TOPIC_ARN', 
                'arn:aws:sns:us-east-1:791060928878:stock-analytics-ai-performance-reports')
        )


@dataclass
class TradingConfig:
    """Trading and market data configuration."""
    max_symbols_per_run: int
    abort_after_seconds: int
    major_indexes: List[str]
    popular_stocks: List[str]
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Create trading config from environment variables."""
        # Default major indexes
        default_indexes = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'SCHD', 'VYM', 'VUG', 'VTV', 'VEA']
        
        # Default popular stocks (subset for brevity)
        default_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'INTU', 'CSCO', 'IBM', 'UBER',
            'LYFT', 'SNAP', 'TWTR', 'PINS', 'SQ', 'PYPL', 'V', 'MA', 'JPM', 'BAC',
            'WFC', 'GS', 'MS', 'C', 'AXP', 'BRK.B', 'JNJ', 'PFE', 'UNH', 'ABBV',
            'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX'
        ]
        
        return cls(
            max_symbols_per_run=EnvConfig.get_int('MAX_SYMBOLS_PER_RUN', 12),
            abort_after_seconds=EnvConfig.get_int('ABORT_AFTER_SEC', 40),
            major_indexes=default_indexes,
            popular_stocks=default_stocks
        )


@dataclass
class PerformanceConfig:
    """Performance and benchmarking configuration."""
    target_hit_rate: float
    target_sharpe_ratio: float
    target_market_outperformance: float
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create performance config from environment variables."""
        return cls(
            target_hit_rate=float(EnvConfig.get_optional('TARGET_HIT_RATE', '0.65')),
            target_sharpe_ratio=float(EnvConfig.get_optional('TARGET_SHARPE_RATIO', '1.0')),
            target_market_outperformance=float(EnvConfig.get_optional('TARGET_MARKET_OUTPERFORMANCE', '0.05'))
        )


class AppConfig:
    """Main application configuration container."""
    
    def __init__(self):
        self.database = DatabaseConfig.from_env()
        self.s3 = S3Config.from_env()
        self.api = APIConfig.from_env()
        self.lambda_functions = LambdaConfig.from_env()
        self.cache = CacheConfig.from_env()
        self.notifications = NotificationConfig.from_env()
        self.trading = TradingConfig.from_env()
        self.performance = PerformanceConfig.from_env()
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate required S3 buckets
        if not self.s3.data_lake_bucket:
            errors.append("S3_DATA_BUCKET is required")
        
        # Validate performance targets
        if not (0 < self.performance.target_hit_rate <= 1):
            errors.append("TARGET_HIT_RATE must be between 0 and 1")
        
        if self.performance.target_sharpe_ratio <= 0:
            errors.append("TARGET_SHARPE_RATIO must be positive")
        
        # Validate trading limits
        if self.trading.max_symbols_per_run <= 0:
            errors.append("MAX_SYMBOLS_PER_RUN must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_aws_region(self) -> str:
        """Get AWS region from environment or default."""
        return EnvConfig.get_optional('AWS_REGION', 'us-east-1')
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return EnvConfig.get_optional('ENVIRONMENT', 'production').lower() in ('dev', 'development', 'local')
    
    def get_log_level(self) -> str:
        """Get logging level from environment."""
        return EnvConfig.get_optional('LOG_LEVEL', 'INFO' if not self.is_development() else 'DEBUG')


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def validate_config() -> None:
    """Validate the global configuration."""
    config.validate()


# Environment-specific configurations
def get_tier_config(tier: str) -> Dict[str, any]:
    """Get tier-specific configuration overrides."""
    tier_configs = {
        'tier1': {
            'max_symbols_per_run': 5,
            'premium_calls_per_minute': 5,
            'cache_ttl': 600
        },
        'tier2': {
            'max_symbols_per_run': 8,
            'premium_calls_per_minute': 10,
            'cache_ttl': 300
        },
        'tier3': {
            'max_symbols_per_run': 12,
            'premium_calls_per_minute': 15,
            'cache_ttl': 180
        }
    }
    
    return tier_configs.get(tier.lower(), {})


# Feature flags
class FeatureFlags:
    """Feature flag management."""
    
    @staticmethod
    def is_dual_prediction_enabled() -> bool:
        """Check if dual prediction system is enabled."""
        return EnvConfig.get_bool('ENABLE_DUAL_PREDICTIONS', True)
    
    @staticmethod
    def is_advanced_tuning_enabled() -> bool:
        """Check if advanced model tuning is enabled."""
        return EnvConfig.get_bool('ENABLE_ADVANCED_TUNING', True)
    
    @staticmethod
    def is_caching_enabled() -> bool:
        """Check if caching is enabled."""
        return EnvConfig.get_bool('ENABLE_CACHING', True) and config.cache.valkey_endpoint is not None
    
    @staticmethod
    def is_metrics_enabled() -> bool:
        """Check if custom metrics are enabled."""
        return EnvConfig.get_bool('ENABLE_CUSTOM_METRICS', True)
    
    @staticmethod
    def is_debug_mode() -> bool:
        """Check if debug mode is enabled."""
        return EnvConfig.get_bool('DEBUG_MODE', False) or config.is_development()
