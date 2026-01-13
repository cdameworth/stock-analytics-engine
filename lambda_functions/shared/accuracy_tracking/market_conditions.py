"""
Market Condition Tracker - PostgreSQL Implementation.

Tracks market conditions and correlates with prediction accuracy.
Understanding how models perform in different market regimes helps
identify when predictions are more/less reliable.

Market Regimes:
    - bull: Strong uptrend (SPY +5% in 20 days, above 50-day MA)
    - bear: Strong downtrend (SPY -5% in 20 days, below 50-day MA)
    - sideways: Consolidation (low volatility, mixed signals)
    - volatile: High volatility (VIX > 30)

Uses yfinance for market data when available.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..database import db, DatabaseError
from .audit_logger import AccuracyAuditLogger

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class MarketConditionTracker:
    """
    Track market conditions and correlate with model accuracy.

    Classifies market regimes and analyzes how model performance
    varies across different conditions.
    """

    REGIMES = ['bull', 'bear', 'sideways', 'volatile']

    # VIX thresholds
    VIX_LOW = 15
    VIX_MODERATE = 20
    VIX_HIGH = 30
    VIX_EXTREME = 40

    # Return thresholds for regime classification
    BULL_THRESHOLD = 0.05  # +5% return
    BEAR_THRESHOLD = -0.05  # -5% return

    def __init__(self):
        """Initialize market condition tracker."""
        self.audit_logger = AccuracyAuditLogger()

    def classify_current_regime(
        self,
        lookback_days: int = 20
    ) -> str:
        """
        Classify current market as bull/bear/sideways/volatile.

        Uses SPY price action and VIX level to determine regime.

        Args:
            lookback_days: Days to analyze for regime classification

        Returns:
            Market regime string
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available, using database cache")
            return self._get_cached_regime()

        try:
            # Get SPY data
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period=f"{lookback_days + 10}d")

            if spy_hist.empty or len(spy_hist) < lookback_days:
                logger.warning("Insufficient SPY data")
                return self._get_cached_regime()

            # Get VIX data
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            vix_level = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 20.0

            # Calculate SPY metrics
            spy_close = spy_hist['Close']
            current_price = float(spy_close.iloc[-1])
            start_price = float(spy_close.iloc[-lookback_days])
            spy_return = (current_price / start_price) - 1

            # Calculate 50-day MA if enough data
            if len(spy_close) >= 50:
                ma50 = float(spy_close.rolling(50).mean().iloc[-1])
            else:
                ma50 = float(spy_close.mean())

            # Classify regime
            if vix_level > self.VIX_HIGH:
                regime = 'volatile'
            elif spy_return > self.BULL_THRESHOLD and current_price > ma50:
                regime = 'bull'
            elif spy_return < self.BEAR_THRESHOLD and current_price < ma50:
                regime = 'bear'
            else:
                regime = 'sideways'

            # Categorize VIX
            if vix_level < self.VIX_LOW:
                vix_category = 'low'
            elif vix_level < self.VIX_MODERATE:
                vix_category = 'moderate'
            elif vix_level < self.VIX_HIGH:
                vix_category = 'high'
            else:
                vix_category = 'extreme'

            # Store condition
            self._store_condition({
                'date': datetime.utcnow().date(),
                'market_regime': regime,
                'spy_daily_return': round(spy_return / lookback_days, 4),
                'spy_weekly_return': round(spy_return, 4),
                'spy_vs_ma50': round((current_price / ma50) - 1, 4),
                'vix_level': round(vix_level, 2),
                'vix_category': vix_category
            })

            logger.info(f"Market regime classified: {regime} (VIX: {vix_level:.1f})")
            return regime

        except Exception as e:
            logger.error(f"Failed to classify market regime: {e}")
            return self._get_cached_regime()

    def _get_cached_regime(self) -> str:
        """Get most recent cached regime from database."""
        try:
            result = db.execute_one("""
                SELECT market_regime
                FROM market_conditions
                ORDER BY date DESC
                LIMIT 1
            """)
            return result['market_regime'] if result else 'sideways'
        except DatabaseError:
            return 'sideways'

    def _store_condition(self, data: Dict):
        """Store market condition to database."""
        query = """
            INSERT INTO market_conditions
            (date, market_regime, spy_daily_return, spy_weekly_return,
             spy_vs_ma50, vix_level, vix_category)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                market_regime = EXCLUDED.market_regime,
                spy_daily_return = EXCLUDED.spy_daily_return,
                spy_weekly_return = EXCLUDED.spy_weekly_return,
                spy_vs_ma50 = EXCLUDED.spy_vs_ma50,
                vix_level = EXCLUDED.vix_level,
                vix_category = EXCLUDED.vix_category
        """

        try:
            db.execute(query, (
                data['date'],
                data['market_regime'],
                data.get('spy_daily_return'),
                data.get('spy_weekly_return'),
                data.get('spy_vs_ma50'),
                data.get('vix_level'),
                data.get('vix_category')
            ))

            # Audit log
            self.audit_logger.log_event('market_condition', data)

        except DatabaseError as e:
            logger.warning(f"Failed to store market condition: {e}")

    def get_current_conditions(self) -> Dict:
        """
        Get current market conditions from database.

        Uses v_current_market view.
        """
        query = """
            SELECT *
            FROM v_current_market
        """

        try:
            result = db.execute_one(query)

            if result:
                return {
                    'date': result['date'].isoformat() if result['date'] else None,
                    'market_regime': result['market_regime'],
                    'spy_daily_return': (
                        float(result['spy_daily_return'])
                        if result['spy_daily_return'] else None
                    ),
                    'spy_weekly_return': (
                        float(result['spy_weekly_return'])
                        if result['spy_weekly_return'] else None
                    ),
                    'vix_level': (
                        float(result['vix_level'])
                        if result['vix_level'] else None
                    ),
                    'vix_category': result['vix_category']
                }

            return {'error': 'No market condition data available'}

        except DatabaseError as e:
            logger.error(f"Failed to get current conditions: {e}")
            raise

    def correlate_accuracy_with_regime(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 90
    ) -> Dict:
        """
        Analyze accuracy by market regime.

        Args:
            model_type: Type of model to analyze
            lookback_days: Days to include in analysis

        Returns:
            Dictionary with accuracy by regime
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Select appropriate table
        if model_type == 'time_prediction':
            table = 'time_predictions'
        else:
            table = 'price_predictions'

        # Join predictions with market conditions
        query = f"""
            SELECT
                mc.market_regime,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN p.accuracy_pct >= 95 THEN 1 ELSE 0 END) as accurate,
                AVG(p.accuracy_pct) as avg_accuracy_pct
            FROM {table} p
            JOIN market_conditions mc ON DATE(p.validated_at) = mc.date
            WHERE p.validation_status = 'validated'
              AND p.validated_at >= %s
            GROUP BY mc.market_regime
            ORDER BY total_predictions DESC
        """

        try:
            results = db.execute(query, (cutoff_date,))

            if not results:
                return {
                    'error': 'No data available for correlation',
                    'model_type': model_type
                }

            # Calculate overall accuracy for comparison
            total_all = sum(r['total_predictions'] for r in results)
            accurate_all = sum(r['accurate'] for r in results)
            baseline_accuracy = accurate_all / total_all if total_all > 0 else 0

            regime_stats = {}
            for r in results:
                regime = r['market_regime']
                accuracy = r['accurate'] / r['total_predictions'] if r['total_predictions'] > 0 else 0

                regime_stats[regime] = {
                    'total_predictions': r['total_predictions'],
                    'accurate_predictions': r['accurate'],
                    'accuracy_rate': round(accuracy, 4),
                    'outperformance': round(accuracy - baseline_accuracy, 4),
                    'avg_accuracy_pct': (
                        round(float(r['avg_accuracy_pct']), 2)
                        if r['avg_accuracy_pct'] else None
                    )
                }

                # Store correlation
                self._store_correlation(model_type, regime, accuracy, baseline_accuracy, r)

            # Find best and worst regimes
            sorted_regimes = sorted(
                regime_stats.items(),
                key=lambda x: x[1]['accuracy_rate'],
                reverse=True
            )

            return {
                'model_type': model_type,
                'baseline_accuracy': round(baseline_accuracy, 4),
                'regimes': regime_stats,
                'best_regime': sorted_regimes[0][0] if sorted_regimes else None,
                'worst_regime': sorted_regimes[-1][0] if sorted_regimes else None,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to correlate accuracy with regime: {e}")
            raise

    def _store_correlation(
        self,
        model_type: str,
        regime: str,
        accuracy: float,
        baseline: float,
        data: Dict
    ):
        """Store accuracy-regime correlation to database."""
        today = datetime.utcnow().date()
        week_ago = today - timedelta(days=7)

        query = """
            INSERT INTO accuracy_correlation
            (model_type, market_regime, period_start, period_end,
             sample_size, accuracy_rate, baseline_accuracy, outperformance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_type, market_regime, period_end)
            DO UPDATE SET
                sample_size = EXCLUDED.sample_size,
                accuracy_rate = EXCLUDED.accuracy_rate,
                baseline_accuracy = EXCLUDED.baseline_accuracy,
                outperformance = EXCLUDED.outperformance
        """

        try:
            db.execute(query, (
                model_type,
                regime,
                week_ago,
                today,
                data['total_predictions'],
                accuracy,
                baseline,
                accuracy - baseline
            ))
        except DatabaseError as e:
            logger.warning(f"Failed to store correlation: {e}")

    def get_correlation_report(
        self,
        model_type: str = 'price_prediction'
    ) -> Dict:
        """
        Get accuracy-market correlation report.

        Main entry point for /analytics/market-correlation endpoint.
        """
        # Get current conditions
        conditions = self.get_current_conditions()

        # Get correlation analysis
        correlation = self.correlate_accuracy_with_regime(model_type)

        return {
            'current_conditions': conditions,
            'correlation': correlation,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_regime_adjusted_confidence(
        self,
        base_confidence: float,
        model_type: str = 'price_prediction'
    ) -> float:
        """
        Adjust confidence based on historical regime performance.

        If the model historically underperforms in the current regime,
        reduce confidence. If it overperforms, increase confidence.

        Args:
            base_confidence: Original confidence score (0-1)
            model_type: Type of model

        Returns:
            Adjusted confidence score (0-1)
        """
        try:
            # Get current regime
            current = self.get_current_conditions()
            if 'error' in current:
                return base_confidence

            current_regime = current.get('market_regime', 'sideways')

            # Get correlation for current regime
            query = """
                SELECT accuracy_rate, baseline_accuracy, outperformance
                FROM accuracy_correlation
                WHERE model_type = %s
                  AND market_regime = %s
                ORDER BY period_end DESC
                LIMIT 1
            """

            result = db.execute_one(query, (model_type, current_regime))

            if not result or result['outperformance'] is None:
                return base_confidence

            # Adjust confidence based on regime performance
            # If model outperforms in this regime, boost confidence
            # If model underperforms, reduce confidence
            outperformance = float(result['outperformance'])

            # Adjustment factor: -0.05 outperformance -> 0.95x, +0.05 -> 1.05x
            adjustment = 1.0 + (outperformance * 2)  # Scale factor

            # Apply adjustment with bounds
            adjusted = base_confidence * adjustment
            adjusted = max(0.0, min(1.0, adjusted))

            return round(adjusted, 4)

        except Exception as e:
            logger.warning(f"Failed to adjust confidence: {e}")
            return base_confidence

    def get_regime_history(
        self,
        days: int = 30
    ) -> List[Dict]:
        """
        Get recent market regime history.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily regime records
        """
        query = """
            SELECT date, market_regime, vix_level, vix_category,
                   spy_weekly_return
            FROM market_conditions
            ORDER BY date DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, (days,))
            return [
                {
                    'date': r['date'].isoformat(),
                    'regime': r['market_regime'],
                    'vix_level': float(r['vix_level']) if r['vix_level'] else None,
                    'vix_category': r['vix_category'],
                    'spy_return': (
                        float(r['spy_weekly_return'])
                        if r['spy_weekly_return'] else None
                    )
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get regime history: {e}")
            raise

    def get_regime_distribution(
        self,
        days: int = 90
    ) -> Dict:
        """
        Get distribution of market regimes over period.

        Args:
            days: Days to analyze

        Returns:
            Dictionary with regime counts and percentages
        """
        query = """
            SELECT market_regime, COUNT(*) as days
            FROM market_conditions
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY market_regime
            ORDER BY days DESC
        """

        try:
            results = db.execute(query, (days,))

            total = sum(r['days'] for r in results)
            distribution = {
                r['market_regime']: {
                    'days': r['days'],
                    'percentage': round(r['days'] / total * 100, 1) if total > 0 else 0
                }
                for r in results
            }

            return {
                'distribution': distribution,
                'total_days': total,
                'period_days': days,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to get regime distribution: {e}")
            raise
