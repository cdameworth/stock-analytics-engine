"""
SigNoz Dashboard Integration for Stock Analytics
Provides custom dashboards, queries, and business intelligence for financial tracing data
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of SigNoz dashboards"""
    ML_PERFORMANCE = "ml_performance"
    TRADING_SIGNALS = "trading_signals"
    MARKET_OVERVIEW = "market_overview"
    COST_OPTIMIZATION = "cost_optimization"
    SYSTEM_HEALTH = "system_health"

@dataclass
class QueryTemplate:
    """SigNoz query template for dashboard widgets"""
    query_type: str  # "metrics" or "traces"
    query: str
    legend: str
    aggregation: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

@dataclass
class DashboardWidget:
    """SigNoz dashboard widget configuration"""
    title: str
    widget_type: str  # "line", "bar", "table", "stat", "heatmap"
    queries: List[QueryTemplate]
    description: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None

class SigNozDashboardBuilder:
    """
    Build custom SigNoz dashboards for financial analytics
    """

    def __init__(self):
        self.base_filters = {
            "service_name": "stock_analytics",
            "time_range": "1h"
        }

    def create_ml_performance_dashboard(self) -> Dict[str, Any]:
        """
        Create ML model performance dashboard
        """
        widgets = [
            # Real-time accuracy tracking
            DashboardWidget(
                title="ML Model Accuracy by Symbol Tier",
                widget_type="line",
                description="Real-time accuracy tracking across different symbol tiers",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            quantile(0.5)(toFloat64(attribute_ml_accuracy_price_error_pct)) as median_error,
                            attribute_finance_symbol_tier as symbol_tier,
                            toStartOfInterval(timestamp, INTERVAL 5 minute) as time_bucket
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_ml_accuracy_price_error_pct != ''
                        AND timestamp >= now() - INTERVAL 1 HOUR
                        GROUP BY symbol_tier, time_bucket
                        ORDER BY time_bucket
                        """,
                        legend="{{symbol_tier}} Error %",
                        aggregation="median"
                    )
                ],
                thresholds={"warning": 10.0, "critical": 20.0}
            ),

            # Confidence calibration
            DashboardWidget(
                title="Model Confidence Calibration",
                widget_type="heatmap",
                description="Confidence vs actual accuracy correlation",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            round(toFloat64(attribute_finance_confidence_score), 1) as confidence_bucket,
                            round(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as accuracy_bucket,
                            count() as prediction_count
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_confidence_score != ''
                        AND attribute_ml_accuracy_price_error_pct != ''
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        GROUP BY confidence_bucket, accuracy_bucket
                        """,
                        legend="Confidence vs Accuracy"
                    )
                ]
            ),

            # Symbol performance ranking
            DashboardWidget(
                title="Top Performing Symbols (24h)",
                widget_type="table",
                description="Best and worst performing symbols by accuracy",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_symbol as symbol,
                            count() as prediction_count,
                            avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as avg_accuracy,
                            max(toFloat64(attribute_finance_confidence_score)) as max_confidence
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_symbol != ''
                        AND attribute_ml_accuracy_price_error_pct != ''
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        GROUP BY symbol
                        HAVING prediction_count >= 5
                        ORDER BY avg_accuracy DESC
                        LIMIT 20
                        """,
                        legend="Symbol Performance"
                    )
                ]
            ),

            # Market session impact
            DashboardWidget(
                title="Performance by Market Session",
                widget_type="bar",
                description="Model accuracy across different market sessions",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_market_session as market_session,
                            avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as avg_accuracy,
                            count() as prediction_count
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_market_session != ''
                        AND attribute_ml_accuracy_price_error_pct != ''
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        GROUP BY market_session
                        ORDER BY avg_accuracy DESC
                        """,
                        legend="{{market_session}}"
                    )
                ]
            )
        ]

        return self._build_dashboard("ML Model Performance", widgets)

    def create_trading_signals_dashboard(self) -> Dict[str, Any]:
        """
        Create trading signals quality dashboard
        """
        widgets = [
            # Signal quality distribution
            DashboardWidget(
                title="Trading Signal Quality Distribution",
                widget_type="bar",
                description="Distribution of trading signal quality tiers",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_trading_signal_quality as quality_tier,
                            count() as signal_count,
                            avg(toFloat64(attribute_trading_profit_loss_pct)) as avg_profit_loss
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE spanName = 'trading.signal_quality'
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        GROUP BY quality_tier
                        ORDER BY signal_count DESC
                        """,
                        legend="{{quality_tier}}"
                    )
                ]
            ),

            # Profitable signals by symbol tier
            DashboardWidget(
                title="Profitable Signals by Symbol Tier",
                widget_type="line",
                description="Percentage of profitable signals by symbol importance",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_symbol_tier as symbol_tier,
                            toStartOfInterval(timestamp, INTERVAL 1 hour) as time_bucket,
                            countIf(toFloat64(attribute_trading_profit_loss_pct) > 0) * 100.0 / count() as profitable_pct
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE spanName = 'trading.signal_quality'
                        AND attribute_trading_profit_loss_pct != ''
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        GROUP BY symbol_tier, time_bucket
                        ORDER BY time_bucket
                        """,
                        legend="{{symbol_tier}}"
                    )
                ]
            ),

            # Market response time analysis
            DashboardWidget(
                title="Average Market Response Time",
                widget_type="stat",
                description="How quickly the market responds to our signals",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            avg(toFloat64(attribute_trading_market_response_hours)) as avg_response_hours
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE spanName = 'trading.signal_quality'
                        AND attribute_trading_market_response_hours != ''
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        """,
                        legend="Hours"
                    )
                ],
                thresholds={"good": 2.0, "warning": 6.0}
            )
        ]

        return self._build_dashboard("Trading Signals Quality", widgets)

    def create_market_overview_dashboard(self) -> Dict[str, Any]:
        """
        Create market overview dashboard
        """
        widgets = [
            # Active predictions by market session
            DashboardWidget(
                title="Active Predictions by Market Session",
                widget_type="line",
                description="Real-time prediction volume across market sessions",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_market_session as market_session,
                            toStartOfInterval(timestamp, INTERVAL 15 minute) as time_bucket,
                            count() as prediction_count
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE spanName LIKE 'ml.%prediction%'
                        AND timestamp >= now() - INTERVAL 4 HOUR
                        GROUP BY market_session, time_bucket
                        ORDER BY time_bucket
                        """,
                        legend="{{market_session}}"
                    )
                ]
            ),

            # Symbol tier distribution
            DashboardWidget(
                title="Prediction Volume by Symbol Tier",
                widget_type="bar",
                description="Distribution of predictions across symbol importance tiers",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_symbol_tier as symbol_tier,
                            count() as prediction_count,
                            uniq(attribute_finance_symbol) as unique_symbols
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_symbol_tier != ''
                        AND timestamp >= now() - INTERVAL 1 HOUR
                        GROUP BY symbol_tier
                        ORDER BY prediction_count DESC
                        """,
                        legend="{{symbol_tier}}"
                    )
                ]
            ),

            # High confidence opportunities
            DashboardWidget(
                title="High Confidence Opportunities",
                widget_type="table",
                description="Current high-confidence trading opportunities",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_symbol as symbol,
                            max(toFloat64(attribute_finance_confidence_score)) as confidence,
                            max(toFloat64(attribute_finance_target_price)) as target_price,
                            max(toFloat64(attribute_finance_current_price)) as current_price,
                            max(toFloat64(attribute_finance_price_change_pct)) as price_change_pct
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_confidence_score != ''
                        AND toFloat64(attribute_finance_confidence_score) >= 0.8
                        AND timestamp >= now() - INTERVAL 1 HOUR
                        GROUP BY symbol
                        ORDER BY confidence DESC
                        LIMIT 10
                        """,
                        legend="High Confidence Signals"
                    )
                ]
            )
        ]

        return self._build_dashboard("Market Overview", widgets)

    def create_cost_optimization_dashboard(self) -> Dict[str, Any]:
        """
        Create cost optimization dashboard
        """
        widgets = [
            # Trace volume and sampling rates
            DashboardWidget(
                title="Trace Volume by Sampling Strategy",
                widget_type="line",
                description="Trace volume and cost impact by sampling strategy",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            attribute_finance_symbol_tier as symbol_tier,
                            toStartOfInterval(timestamp, INTERVAL 30 minute) as time_bucket,
                            count() as trace_count
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE attribute_finance_symbol_tier != ''
                        AND timestamp >= now() - INTERVAL 6 HOUR
                        GROUP BY symbol_tier, time_bucket
                        ORDER BY time_bucket
                        """,
                        legend="{{symbol_tier}}"
                    )
                ]
            ),

            # Performance alerts frequency
            DashboardWidget(
                title="Performance Alerts",
                widget_type="stat",
                description="Recent performance alerts requiring attention",
                queries=[
                    QueryTemplate(
                        query_type="traces",
                        query="""
                        SELECT
                            countIf(attribute_alert_severity = 'critical') as critical_alerts,
                            countIf(attribute_alert_severity = 'warning') as warning_alerts
                        FROM signoz_traces.distributed_signoz_index_v2
                        WHERE spanName = 'system.performance_alert'
                        AND timestamp >= now() - INTERVAL 24 HOUR
                        """,
                        legend="Alerts"
                    )
                ],
                thresholds={"warning": 10, "critical": 5}
            )
        ]

        return self._build_dashboard("Cost Optimization", widgets)

    def _build_dashboard(self, title: str, widgets: List[DashboardWidget]) -> Dict[str, Any]:
        """
        Build complete dashboard configuration
        """
        dashboard_config = {
            "title": title,
            "description": f"Auto-generated {title} dashboard for stock analytics",
            "tags": ["stock_analytics", "financial", "ml"],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "30s",
            "widgets": []
        }

        for i, widget in enumerate(widgets):
            widget_config = {
                "id": f"widget_{i}",
                "title": widget.title,
                "type": widget.widget_type,
                "description": widget.description or "",
                "queries": []
            }

            for j, query in enumerate(widget.queries):
                query_config = {
                    "id": f"query_{i}_{j}",
                    "query": query.query,
                    "legend": query.legend,
                    "queryType": query.query_type
                }

                if query.aggregation:
                    query_config["aggregation"] = query.aggregation

                if query.filters:
                    query_config["filters"] = query.filters

                widget_config["queries"].append(query_config)

            if widget.thresholds:
                widget_config["thresholds"] = widget.thresholds

            dashboard_config["widgets"].append(widget_config)

        return dashboard_config

    def export_all_dashboards(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all dashboard configurations
        """
        dashboards = {
            "ml_performance": self.create_ml_performance_dashboard(),
            "trading_signals": self.create_trading_signals_dashboard(),
            "market_overview": self.create_market_overview_dashboard(),
            "cost_optimization": self.create_cost_optimization_dashboard()
        }

        return dashboards

class SigNozQueryBuilder:
    """
    Build SigNoz queries for specific financial analytics use cases
    """

    @staticmethod
    def get_symbol_performance_query(symbol: str, hours: int = 24) -> str:
        """Get performance metrics for a specific symbol"""
        return f"""
        SELECT
            toStartOfInterval(timestamp, INTERVAL 1 hour) as time_bucket,
            avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as accuracy,
            avg(toFloat64(attribute_finance_confidence_score)) as confidence,
            count() as prediction_count
        FROM signoz_traces.distributed_signoz_index_v2
        WHERE attribute_finance_symbol = '{symbol}'
        AND attribute_ml_accuracy_price_error_pct != ''
        AND timestamp >= now() - INTERVAL {hours} HOUR
        GROUP BY time_bucket
        ORDER BY time_bucket
        """

    @staticmethod
    def get_market_session_analysis_query(session: str) -> str:
        """Analyze performance during specific market session"""
        return f"""
        SELECT
            attribute_finance_symbol as symbol,
            count() as prediction_count,
            avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as avg_accuracy,
            max(toFloat64(attribute_finance_confidence_score)) as max_confidence
        FROM signoz_traces.distributed_signoz_index_v2
        WHERE attribute_finance_market_session = '{session}'
        AND attribute_ml_accuracy_price_error_pct != ''
        AND timestamp >= now() - INTERVAL 24 HOUR
        GROUP BY symbol
        HAVING prediction_count >= 3
        ORDER BY avg_accuracy DESC
        """

    @staticmethod
    def get_low_confidence_symbols_query(threshold: float = 0.6) -> str:
        """Find symbols with consistently low confidence predictions"""
        return f"""
        SELECT
            attribute_finance_symbol as symbol,
            count() as prediction_count,
            avg(toFloat64(attribute_finance_confidence_score)) as avg_confidence,
            avg(100 - toFloat64(attribute_ml_accuracy_price_error_pct)) as avg_accuracy
        FROM signoz_traces.distributed_signoz_index_v2
        WHERE toFloat64(attribute_finance_confidence_score) < {threshold}
        AND attribute_finance_confidence_score != ''
        AND timestamp >= now() - INTERVAL 24 HOUR
        GROUP BY symbol
        HAVING prediction_count >= 5
        ORDER BY avg_confidence ASC
        """

def generate_signoz_dashboard_config() -> str:
    """
    Generate complete SigNoz dashboard configuration as JSON
    """
    builder = SigNozDashboardBuilder()
    dashboards = builder.export_all_dashboards()

    return json.dumps(dashboards, indent=2, default=str)