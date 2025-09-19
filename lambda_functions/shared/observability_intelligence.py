"""
Advanced Observability Intelligence for Stock Analytics
Provides performance monitoring, alerting, and trace-based optimization
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

from .market_utils import (
    get_market_session, classify_symbol, MarketSession, SymbolTier,
    get_financial_attributes
)
from .business_tracing import FinancialTracer, get_financial_tracer

logger = logging.getLogger(__name__)

class PerformanceThreshold(Enum):
    """Performance monitoring thresholds"""
    EXCELLENT = "excellent"      # >85% accuracy
    GOOD = "good"               # 70-85% accuracy
    ACCEPTABLE = "acceptable"   # 60-70% accuracy
    POOR = "poor"              # 45-60% accuracy
    CRITICAL = "critical"      # <45% accuracy

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_name: str
    value: float
    threshold: float
    timestamp: datetime
    symbol: Optional[str] = None
    market_session: Optional[MarketSession] = None

class PerformanceMonitor:
    """
    Advanced performance monitoring with business intelligence
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = get_financial_tracer(service_name)
        self.performance_history: List[PerformanceMetric] = []

    def track_ml_accuracy(
        self,
        symbol: str,
        predicted_price: float,
        actual_price: float,
        confidence_score: float,
        prediction_timestamp: datetime,
        actual_timestamp: Optional[datetime] = None
    ) -> Span:
        """
        Track ML model accuracy with comprehensive business context
        """
        if actual_timestamp is None:
            actual_timestamp = datetime.utcnow()

        # Calculate accuracy metrics
        price_error = abs(predicted_price - actual_price)
        price_error_pct = (price_error / actual_price) * 100
        prediction_age_hours = (actual_timestamp - prediction_timestamp).total_seconds() / 3600

        # Determine accuracy tier
        if price_error_pct <= 2.0:
            accuracy_tier = PerformanceThreshold.EXCELLENT
        elif price_error_pct <= 5.0:
            accuracy_tier = PerformanceThreshold.GOOD
        elif price_error_pct <= 10.0:
            accuracy_tier = PerformanceThreshold.ACCEPTABLE
        elif price_error_pct <= 20.0:
            accuracy_tier = PerformanceThreshold.POOR
        else:
            accuracy_tier = PerformanceThreshold.CRITICAL

        # Create comprehensive tracking span
        span = self.tracer.start_financial_span(
            "ml.accuracy_tracking",
            symbol=symbol,
            current_price=actual_price,
            target_price=predicted_price,
            confidence_score=confidence_score
        )

        # Add accuracy-specific attributes
        accuracy_attrs = {
            "ml.accuracy.price_error": price_error,
            "ml.accuracy.price_error_pct": price_error_pct,
            "ml.accuracy.prediction_age_hours": prediction_age_hours,
            "ml.accuracy.tier": accuracy_tier.value,
            "ml.accuracy.is_accurate": price_error_pct <= 10.0,
            "ml.accuracy.confidence_calibrated": self._is_confidence_calibrated(
                confidence_score, price_error_pct
            ),
            "ml.prediction_timestamp": prediction_timestamp.isoformat(),
            "ml.actual_timestamp": actual_timestamp.isoformat()
        }

        span.set_attributes(accuracy_attrs)

        # Store performance metric
        metric = PerformanceMetric(
            metric_name="ml_accuracy",
            value=100 - price_error_pct,  # Convert to accuracy percentage
            threshold=90.0,  # 90% accuracy threshold
            timestamp=actual_timestamp,
            symbol=symbol,
            market_session=get_market_session(actual_timestamp)
        )
        self.performance_history.append(metric)

        # Check for performance alerts
        self._check_performance_alerts(metric, accuracy_tier)

        return span

    def _is_confidence_calibrated(self, confidence: float, error_pct: float) -> bool:
        """
        Check if model confidence is well-calibrated to actual performance
        """
        # High confidence should correlate with low error
        if confidence >= 0.8 and error_pct <= 5.0:
            return True
        # Medium confidence should correlate with medium error
        elif 0.6 <= confidence < 0.8 and error_pct <= 10.0:
            return True
        # Low confidence can have any error (model is appropriately uncertain)
        elif confidence < 0.6:
            return True
        else:
            return False

    def track_trading_signal_quality(
        self,
        symbol: str,
        signal_type: str,  # "buy", "sell", "hold"
        signal_strength: float,
        market_response_hours: float,
        profit_loss_pct: Optional[float] = None
    ) -> Span:
        """
        Track trading signal quality and market response
        """
        span = self.tracer.start_financial_span(
            "trading.signal_quality",
            symbol=symbol
        )

        # Determine signal quality
        if profit_loss_pct is not None:
            if signal_type == "buy" and profit_loss_pct > 2.0:
                signal_quality = PerformanceThreshold.EXCELLENT
            elif signal_type == "sell" and profit_loss_pct < -2.0:
                signal_quality = PerformanceThreshold.EXCELLENT
            elif abs(profit_loss_pct) > 1.0:
                signal_quality = PerformanceThreshold.GOOD
            elif abs(profit_loss_pct) > 0.5:
                signal_quality = PerformanceThreshold.ACCEPTABLE
            else:
                signal_quality = PerformanceThreshold.POOR
        else:
            # Without P&L, assess based on market response time
            if market_response_hours <= 1.0:
                signal_quality = PerformanceThreshold.EXCELLENT
            elif market_response_hours <= 4.0:
                signal_quality = PerformanceThreshold.GOOD
            else:
                signal_quality = PerformanceThreshold.ACCEPTABLE

        signal_attrs = {
            "trading.signal_type": signal_type,
            "trading.signal_strength": signal_strength,
            "trading.market_response_hours": market_response_hours,
            "trading.signal_quality": signal_quality.value,
            "trading.is_profitable": profit_loss_pct > 0 if profit_loss_pct else None,
            "trading.profit_loss_pct": profit_loss_pct
        }

        span.set_attributes(signal_attrs)
        return span

    def _check_performance_alerts(
        self,
        metric: PerformanceMetric,
        performance_tier: PerformanceThreshold
    ) -> None:
        """
        Check performance metrics against thresholds and generate alerts
        """
        # Critical performance degradation
        if performance_tier == PerformanceThreshold.CRITICAL:
            self._create_alert(
                AlertSeverity.CRITICAL,
                f"Critical ML accuracy degradation for {metric.symbol}: {metric.value:.1f}%",
                {
                    "metric": metric.metric_name,
                    "value": metric.value,
                    "symbol": metric.symbol,
                    "threshold": metric.threshold,
                    "severity": "critical"
                }
            )

        # Poor performance warning
        elif performance_tier == PerformanceThreshold.POOR:
            self._create_alert(
                AlertSeverity.WARNING,
                f"Poor ML accuracy for {metric.symbol}: {metric.value:.1f}%",
                {
                    "metric": metric.metric_name,
                    "value": metric.value,
                    "symbol": metric.symbol,
                    "threshold": metric.threshold,
                    "severity": "warning"
                }
            )

    def _create_alert(
        self,
        severity: AlertSeverity,
        message: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Create and log performance alerts
        """
        alert_span = self.tracer.start_financial_span("system.performance_alert")

        alert_attrs = {
            "alert.severity": severity.value,
            "alert.message": message,
            "alert.timestamp": datetime.utcnow().isoformat(),
            "alert.service": self.service_name
        }

        # Add context attributes
        for key, value in context.items():
            alert_attrs[f"alert.context.{key}"] = value

        alert_span.set_attributes(alert_attrs)

        # Log alert based on severity
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"PERFORMANCE ALERT: {message}", extra=context)
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"PERFORMANCE ALERT: {message}", extra=context)
        else:
            logger.info(f"PERFORMANCE ALERT: {message}", extra=context)

        alert_span.end()

    def get_performance_summary(
        self,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_metrics = [
            m for m in self.performance_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"status": "no_data", "lookback_hours": lookback_hours}

        # Calculate aggregate statistics
        accuracy_values = [m.value for m in recent_metrics if m.metric_name == "ml_accuracy"]

        summary = {
            "lookback_hours": lookback_hours,
            "total_predictions": len(accuracy_values),
            "timestamp": datetime.utcnow().isoformat()
        }

        if accuracy_values:
            summary.update({
                "accuracy_mean": statistics.mean(accuracy_values),
                "accuracy_median": statistics.median(accuracy_values),
                "accuracy_stdev": statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0,
                "accuracy_min": min(accuracy_values),
                "accuracy_max": max(accuracy_values)
            })

            # Performance tier distribution
            tier_counts = {}
            for metric in recent_metrics:
                if metric.value >= 85:
                    tier = PerformanceThreshold.EXCELLENT.value
                elif metric.value >= 70:
                    tier = PerformanceThreshold.GOOD.value
                elif metric.value >= 60:
                    tier = PerformanceThreshold.ACCEPTABLE.value
                elif metric.value >= 45:
                    tier = PerformanceThreshold.POOR.value
                else:
                    tier = PerformanceThreshold.CRITICAL.value

                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            summary["performance_distribution"] = tier_counts

        return summary

class DynamicSamplingOptimizer:
    """
    Optimize sampling rates based on performance and cost metrics
    """

    def __init__(self):
        self.performance_monitor = PerformanceMonitor("sampling_optimizer")

    def optimize_sampling_rates(
        self,
        performance_summary: Dict[str, Any],
        cost_budget_daily: float = 50.0
    ) -> Dict[str, float]:
        """
        Dynamically adjust sampling rates based on performance and cost
        """
        span = self.performance_monitor.tracer.start_financial_span(
            "system.sampling_optimization"
        )

        # Base sampling rates
        base_rates = {
            SymbolTier.MEGA_CAP.value: 1.0,
            SymbolTier.LARGE_CAP.value: 0.75,
            SymbolTier.ETF.value: 0.9,
            SymbolTier.MID_CAP.value: 0.5,
            SymbolTier.SMALL_CAP.value: 0.25,
            SymbolTier.UNKNOWN.value: 0.25
        }

        # Adjust based on performance
        if "accuracy_mean" in performance_summary:
            accuracy = performance_summary["accuracy_mean"]

            if accuracy < 60:  # Poor performance - increase sampling for debugging
                performance_multiplier = 1.5
            elif accuracy < 75:  # Acceptable - slight increase
                performance_multiplier = 1.2
            elif accuracy > 90:  # Excellent - can reduce sampling
                performance_multiplier = 0.8
            else:  # Good performance - maintain current levels
                performance_multiplier = 1.0
        else:
            performance_multiplier = 1.0

        # Calculate optimized rates
        optimized_rates = {}
        for tier, base_rate in base_rates.items():
            optimized_rate = min(base_rate * performance_multiplier, 1.0)
            optimized_rates[tier] = optimized_rate

        span.set_attributes({
            "optimization.performance_multiplier": performance_multiplier,
            "optimization.cost_budget_daily": cost_budget_daily,
            "optimization.accuracy_input": performance_summary.get("accuracy_mean", 0),
            "optimization.total_predictions": performance_summary.get("total_predictions", 0)
        })

        span.end()
        return optimized_rates

class TradingIntelligence:
    """
    Advanced trading intelligence from observability data
    """

    def __init__(self):
        self.tracer = get_financial_tracer("trading_intelligence")

    def analyze_market_opportunity(
        self,
        symbol: str,
        current_traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze market opportunities based on tracing data patterns
        """
        span = self.tracer.start_financial_span(
            "trading.opportunity_analysis",
            symbol=symbol
        )

        # Extract confidence and accuracy patterns
        confidence_scores = []
        accuracy_scores = []
        recent_predictions = []

        for trace in current_traces:
            if trace.get("finance.symbol") == symbol:
                if "finance.confidence_score" in trace:
                    confidence_scores.append(trace["finance.confidence_score"])
                if "ml.accuracy.price_error_pct" in trace:
                    accuracy_scores.append(100 - trace["ml.accuracy.price_error_pct"])
                if "finance.target_price" in trace:
                    recent_predictions.append({
                        "target_price": trace["finance.target_price"],
                        "confidence": trace.get("finance.confidence_score", 0),
                        "timestamp": trace.get("timestamp", "")
                    })

        # Calculate opportunity signals
        opportunity_score = 0.0
        signals = []

        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence >= 0.8:
                opportunity_score += 30
                signals.append("high_confidence")

        if accuracy_scores:
            avg_accuracy = statistics.mean(accuracy_scores)
            if avg_accuracy >= 85:
                opportunity_score += 25
                signals.append("high_accuracy")

        # Market session bonus
        current_session = get_market_session()
        if current_session == MarketSession.MARKET_HOURS:
            opportunity_score += 20
            signals.append("market_hours")
        elif current_session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]:
            opportunity_score += 10
            signals.append("extended_hours")

        # Symbol tier bonus
        symbol_tier = classify_symbol(symbol)
        if symbol_tier in [SymbolTier.MEGA_CAP, SymbolTier.LARGE_CAP]:
            opportunity_score += 15
            signals.append("major_symbol")

        opportunity_analysis = {
            "symbol": symbol,
            "opportunity_score": min(opportunity_score, 100),  # Cap at 100
            "signals": signals,
            "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "avg_accuracy": statistics.mean(accuracy_scores) if accuracy_scores else 0,
            "recent_predictions_count": len(recent_predictions),
            "market_session": current_session.value,
            "symbol_tier": symbol_tier.value,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

        span.set_attributes({
            f"opportunity.{k}": v for k, v in opportunity_analysis.items()
            if isinstance(v, (str, int, float, bool))
        })

        span.end()
        return opportunity_analysis

# Global instances for easy access
_performance_monitor: Optional[PerformanceMonitor] = None
_sampling_optimizer: Optional[DynamicSamplingOptimizer] = None
_trading_intelligence: Optional[TradingIntelligence] = None

def get_performance_monitor(service_name: str = "stock_analytics") -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(service_name)
    return _performance_monitor

def get_sampling_optimizer() -> DynamicSamplingOptimizer:
    """Get or create global sampling optimizer"""
    global _sampling_optimizer
    if _sampling_optimizer is None:
        _sampling_optimizer = DynamicSamplingOptimizer()
    return _sampling_optimizer

def get_trading_intelligence() -> TradingIntelligence:
    """Get or create global trading intelligence"""
    global _trading_intelligence
    if _trading_intelligence is None:
        _trading_intelligence = TradingIntelligence()
    return _trading_intelligence