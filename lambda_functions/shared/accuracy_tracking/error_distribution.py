"""
Error Distribution Analyzer - PostgreSQL Implementation.

Analyzes error magnitude distribution beyond simple pass/fail metrics.
Understanding the distribution of errors helps identify:
- Whether model tends to over/underestimate
- What the typical error range is
- How many extreme outliers occur

Metrics:
    - Percentiles (p10, p25, p50, p75, p90)
    - Mean and standard deviation
    - Skewness (positive = overestimates, negative = underestimates)
    - Outlier count (errors > 3 std)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..database import db, DatabaseError
from .audit_logger import AccuracyAuditLogger

logger = logging.getLogger(__name__)

# Try to import numpy for statistics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class ErrorDistributionAnalyzer:
    """
    Analyze error magnitude distribution using PostgreSQL.

    Goes beyond binary pass/fail to understand the full distribution
    of prediction errors.
    """

    def __init__(self):
        """Initialize error distribution analyzer."""
        self.audit_logger = AccuracyAuditLogger()

    def calculate_distribution(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Calculate error percentile distribution.

        Args:
            model_type: Type of model to analyze
            lookback_days: Days to include in analysis

        Returns:
            Dictionary with distribution statistics

        Example:
            result = analyzer.calculate_distribution(lookback_days=30)
            print(f"Median error: {result['p50']:.2%}")
            print(f"90th percentile: {result['p90']:.2%}")
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Select appropriate table
        if model_type == 'time_prediction':
            table = 'time_predictions'
            error_field = 'days_error'  # Use days_error for time predictions
        else:
            table = 'price_predictions'
            error_field = 'error_magnitude'

        # Get all errors for validated predictions
        query = f"""
            SELECT {error_field} as error_value, error_direction
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
              AND {error_field} IS NOT NULL
        """

        try:
            results = db.execute(query, (cutoff_date,))

            if not results:
                return {
                    'error': 'No error data available',
                    'model_type': model_type,
                    'lookback_days': lookback_days
                }

            # Extract error values
            errors = [
                abs(float(r['error_value']))
                for r in results
                if r['error_value'] is not None
            ]

            if not errors:
                return {
                    'error': 'No valid error data',
                    'model_type': model_type
                }

            # Calculate statistics
            if NUMPY_AVAILABLE:
                stats = self._calculate_with_numpy(errors)
            else:
                stats = self._calculate_without_numpy(errors)

            # Count directional bias
            overestimates = sum(
                1 for r in results
                if r.get('error_direction') == 'overestimated'
            )
            underestimates = sum(
                1 for r in results
                if r.get('error_direction') == 'underestimated'
            )

            # Prepare result
            result = {
                'model_type': model_type,
                'sample_size': len(errors),
                'p10': stats['p10'],
                'p25': stats['p25'],
                'p50': stats['p50'],
                'p75': stats['p75'],
                'p90': stats['p90'],
                'mean': stats['mean'],
                'std': stats['std'],
                'skewness': stats['skewness'],
                'outlier_count': stats['outliers'],
                'direction_bias': {
                    'overestimates': overestimates,
                    'underestimates': underestimates,
                    'bias': 'overestimate' if overestimates > underestimates else 'underestimate'
                },
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Store to database
            self._store_distribution(result)

            # Audit log
            self.audit_logger.log_event('error_distribution', result)

            return result

        except DatabaseError as e:
            logger.error(f"Failed to calculate distribution: {e}")
            raise

    def _calculate_with_numpy(self, errors: List[float]) -> Dict:
        """Calculate statistics using numpy."""
        arr = np.array(errors)

        mean = float(np.mean(arr))
        std = float(np.std(arr))

        # Calculate skewness
        if std > 0:
            skewness = float(np.mean(((arr - mean) / std) ** 3))
        else:
            skewness = 0.0

        # Count outliers (> 3 std from mean)
        outliers = int(np.sum(np.abs(arr - mean) > 3 * std))

        return {
            'p10': float(np.percentile(arr, 10)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'mean': round(mean, 4),
            'std': round(std, 4),
            'skewness': round(skewness, 4),
            'outliers': outliers
        }

    def _calculate_without_numpy(self, errors: List[float]) -> Dict:
        """Calculate statistics without numpy (fallback)."""
        sorted_errors = sorted(errors)
        n = len(sorted_errors)

        def percentile(p):
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            return sorted_errors[idx]

        mean = sum(errors) / n
        variance = sum((e - mean) ** 2 for e in errors) / n
        std = variance ** 0.5

        # Simple skewness calculation
        if std > 0:
            skewness = sum(((e - mean) / std) ** 3 for e in errors) / n
        else:
            skewness = 0.0

        # Count outliers
        outliers = sum(1 for e in errors if abs(e - mean) > 3 * std)

        return {
            'p10': round(percentile(10), 4),
            'p25': round(percentile(25), 4),
            'p50': round(percentile(50), 4),
            'p75': round(percentile(75), 4),
            'p90': round(percentile(90), 4),
            'mean': round(mean, 4),
            'std': round(std, 4),
            'skewness': round(skewness, 4),
            'outliers': outliers
        }

    def _store_distribution(self, data: Dict):
        """Store distribution statistics to database."""
        period_end = datetime.utcnow().date()
        period_start = period_end - timedelta(days=data.get('lookback_days', 30))

        query = """
            INSERT INTO error_distribution
            (model_type, period, sample_size, p10_error, p25_error,
             p50_error, p75_error, p90_error, mean_error, std_error,
             skewness, outlier_count, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            db.execute(query, (
                data['model_type'],
                'daily',
                data['sample_size'],
                data['p10'],
                data['p25'],
                data['p50'],
                data['p75'],
                data['p90'],
                data['mean'],
                data['std'],
                data['skewness'],
                data['outlier_count'],
                period_start,
                period_end
            ))
        except DatabaseError as e:
            logger.warning(f"Failed to store distribution: {e}")

    def get_distribution(
        self,
        model_type: str = 'price_prediction'
    ) -> Dict:
        """
        Get latest error distribution for API.

        Main entry point for /analytics/error-distribution endpoint.
        """
        # First try to get from database cache
        query = """
            SELECT *
            FROM error_distribution
            WHERE model_type = %s
            ORDER BY period_end DESC
            LIMIT 1
        """

        try:
            result = db.execute_one(query, (model_type,))

            if result:
                return {
                    'model_type': result['model_type'],
                    'sample_size': result['sample_size'],
                    'p10': float(result['p10_error']),
                    'p25': float(result['p25_error']),
                    'p50': float(result['p50_error']),
                    'p75': float(result['p75_error']),
                    'p90': float(result['p90_error']),
                    'mean': float(result['mean_error']),
                    'std': float(result['std_error']),
                    'skewness': float(result['skewness']) if result['skewness'] else None,
                    'outlier_count': result['outlier_count'],
                    'period_start': result['period_start'].isoformat(),
                    'period_end': result['period_end'].isoformat(),
                    'cached': True,
                    'timestamp': datetime.utcnow().isoformat()
                }

            # If no cached data, calculate fresh
            return self.calculate_distribution(model_type)

        except DatabaseError as e:
            logger.error(f"Failed to get distribution: {e}")
            # Try fresh calculation
            return self.calculate_distribution(model_type)

    def get_distribution_history(
        self,
        model_type: str = 'price_prediction',
        limit: int = 30
    ) -> List[Dict]:
        """
        Get historical distribution statistics.

        Args:
            model_type: Type of model
            limit: Maximum records to return

        Returns:
            List of historical distribution records
        """
        query = """
            SELECT period_end, sample_size, p50_error as median,
                   mean_error, std_error, skewness, outlier_count
            FROM error_distribution
            WHERE model_type = %s
            ORDER BY period_end DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, (model_type, limit))
            return [
                {
                    'date': r['period_end'].isoformat(),
                    'sample_size': r['sample_size'],
                    'median_error': float(r['median']),
                    'mean_error': float(r['mean_error']),
                    'std_error': float(r['std_error']),
                    'skewness': float(r['skewness']) if r['skewness'] else None,
                    'outlier_count': r['outlier_count']
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get distribution history: {e}")
            raise

    def get_error_trend(
        self,
        model_type: str = 'price_prediction',
        days: int = 14
    ) -> Dict:
        """
        Analyze error trend over time.

        Compares recent errors to older errors to detect improvement
        or degradation.
        """
        history = self.get_distribution_history(model_type, limit=days)

        if len(history) < 2:
            return {
                'trend': 'insufficient_data',
                'data_points': len(history)
            }

        # Split into recent and older
        mid = len(history) // 2
        recent = history[:mid]
        older = history[mid:]

        recent_median = sum(h['median_error'] for h in recent) / len(recent)
        older_median = sum(h['median_error'] for h in older) / len(older)

        diff = recent_median - older_median

        if diff < -0.005:  # 0.5% improvement
            trend = 'improving'
        elif diff > 0.005:  # 0.5% degradation
            trend = 'degrading'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_median_error': round(recent_median, 4),
            'older_median_error': round(older_median, 4),
            'change': round(diff, 4),
            'change_pct': round(diff / older_median * 100, 2) if older_median > 0 else 0,
            'data_points': len(history)
        }

    def get_outlier_analysis(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Analyze outlier predictions in detail.

        Args:
            model_type: Type of model
            lookback_days: Days to analyze

        Returns:
            Dictionary with outlier analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        if model_type == 'time_prediction':
            table = 'time_predictions'
            error_field = 'days_error'
        else:
            table = 'price_predictions'
            error_field = 'error_magnitude'

        # Get error statistics
        stats_query = f"""
            SELECT
                AVG({error_field}) as mean_error,
                STDDEV({error_field}) as std_error
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
              AND {error_field} IS NOT NULL
        """

        try:
            stats = db.execute_one(stats_query, (cutoff_date,))

            if not stats or stats['mean_error'] is None:
                return {'error': 'No data for outlier analysis'}

            mean = float(stats['mean_error'])
            std = float(stats['std_error']) if stats['std_error'] else 0

            if std == 0:
                return {
                    'outliers': [],
                    'outlier_count': 0,
                    'message': 'No variance in errors'
                }

            # Find outliers (> 3 std from mean)
            outlier_threshold = mean + 3 * std

            outlier_query = f"""
                SELECT symbol, {error_field} as error, error_direction,
                       predicted_price, actual_price, validated_at
                FROM {table}
                WHERE validation_status = 'validated'
                  AND validated_at >= %s
                  AND ABS({error_field}) > %s
                ORDER BY ABS({error_field}) DESC
                LIMIT 20
            """

            outliers = db.execute(outlier_query, (cutoff_date, outlier_threshold))

            return {
                'mean_error': round(mean, 4),
                'std_error': round(std, 4),
                'outlier_threshold': round(outlier_threshold, 4),
                'outlier_count': len(outliers),
                'outliers': [
                    {
                        'symbol': o['symbol'],
                        'error': float(o['error']),
                        'direction': o['error_direction'],
                        'predicted': float(o['predicted_price']) if o.get('predicted_price') else None,
                        'actual': float(o['actual_price']) if o.get('actual_price') else None,
                        'date': o['validated_at'].isoformat()
                    }
                    for o in outliers
                ],
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to analyze outliers: {e}")
            raise

    def get_error_by_confidence(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Analyze error distribution by confidence level.

        Higher confidence predictions should have lower errors.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        if model_type == 'time_prediction':
            table = 'time_predictions'
        else:
            table = 'price_predictions'

        query = f"""
            SELECT
                CASE
                    WHEN confidence_score < 0.5 THEN 'low'
                    WHEN confidence_score < 0.7 THEN 'medium'
                    WHEN confidence_score < 0.9 THEN 'high'
                    ELSE 'very_high'
                END as confidence_level,
                COUNT(*) as count,
                AVG(error_magnitude) as avg_error,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY error_magnitude) as median_error
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
              AND error_magnitude IS NOT NULL
            GROUP BY
                CASE
                    WHEN confidence_score < 0.5 THEN 'low'
                    WHEN confidence_score < 0.7 THEN 'medium'
                    WHEN confidence_score < 0.9 THEN 'high'
                    ELSE 'very_high'
                END
            ORDER BY avg_error
        """

        try:
            results = db.execute(query, (cutoff_date,))

            return {
                'by_confidence': {
                    r['confidence_level']: {
                        'count': r['count'],
                        'avg_error': round(float(r['avg_error']), 4) if r['avg_error'] else None,
                        'median_error': round(float(r['median_error']), 4) if r['median_error'] else None
                    }
                    for r in results
                },
                'model_type': model_type,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to analyze error by confidence: {e}")
            raise
