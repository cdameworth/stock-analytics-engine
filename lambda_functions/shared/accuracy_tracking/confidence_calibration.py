"""
Confidence Calibration Tracker - PostgreSQL Implementation.

Tracks and analyzes confidence calibration using Expected Calibration Error (ECE).
A well-calibrated model should have predictions where confidence matches accuracy:
- 70% confidence predictions should be correct ~70% of the time
- 90% confidence predictions should be correct ~90% of the time

ECE measures the gap between predicted confidence and actual accuracy.
Lower ECE = better calibration.

Calibration Quality:
    - Good: ECE < 0.05 (predictions match confidence well)
    - Moderate: 0.05 <= ECE < 0.10 (some calibration drift)
    - Poor: ECE >= 0.10 (significant over/under-confidence)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..database import db, DatabaseError
from .audit_logger import AccuracyAuditLogger

logger = logging.getLogger(__name__)


class ConfidenceCalibrationTracker:
    """
    Track and analyze confidence calibration using PostgreSQL.

    Calculates Expected Calibration Error (ECE) by grouping predictions
    into confidence buckets and comparing expected vs actual accuracy.
    """

    # Confidence buckets: 0-10%, 10-20%, ..., 90-100%
    CONFIDENCE_BUCKETS = [(i, i + 10) for i in range(0, 100, 10)]

    # Accuracy threshold for "correct" prediction (within 5% of target)
    ACCURACY_THRESHOLD = 95.0

    def __init__(self):
        """Initialize calibration tracker."""
        self.audit_logger = AccuracyAuditLogger()

    def calculate_ece(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Calculate Expected Calibration Error.

        ECE = sum(|accuracy_bucket - confidence_bucket| * n_bucket) / n_total

        Args:
            model_type: Type of model ('price_prediction' or 'time_prediction')
            lookback_days: Number of days to include in calculation

        Returns:
            Dictionary with ECE, bucket breakdown, and quality assessment

        Example:
            result = tracker.calculate_ece(lookback_days=30)
            print(f"ECE: {result['ece']:.4f}")
            print(f"Quality: {result['calibration_quality']}")
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Select the right table based on model type
        if model_type == 'time_prediction':
            table = 'time_predictions'
        else:
            table = 'price_predictions'

        # Get validated predictions with confidence scores
        query = f"""
            SELECT
                confidence_score,
                CASE WHEN accuracy_pct >= %s THEN 1 ELSE 0 END as is_accurate
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
              AND confidence_score IS NOT NULL
        """

        try:
            predictions = db.execute(query, (self.ACCURACY_THRESHOLD, cutoff_date))

            if not predictions:
                result = {
                    'ece': None,
                    'total_predictions': 0,
                    'calibration_quality': 'insufficient_data',
                    'buckets': {},
                    'model_type': model_type,
                    'lookback_days': lookback_days,
                    'timestamp': datetime.utcnow().isoformat()
                }
                return result

            # Calculate per-bucket metrics
            bucket_stats = {}
            for low, high in self.CONFIDENCE_BUCKETS:
                bucket_preds = [
                    p for p in predictions
                    if p['confidence_score'] is not None
                    and low <= float(p['confidence_score']) * 100 < high
                ]

                if bucket_preds:
                    accurate_count = sum(p['is_accurate'] for p in bucket_preds)
                    expected = (low + high) / 200  # Midpoint as decimal
                    actual = accurate_count / len(bucket_preds)

                    bucket_stats[f"{low}-{high}"] = {
                        'count': len(bucket_preds),
                        'accurate_count': accurate_count,
                        'expected_accuracy': round(expected, 4),
                        'actual_accuracy': round(actual, 4),
                        'calibration_error': round(abs(expected - actual), 4)
                    }

                    # Store bucket metrics to database
                    self._store_bucket_metrics(
                        model_type, low, high, len(bucket_preds),
                        accurate_count, expected, actual, lookback_days
                    )

            # Calculate weighted ECE
            total_count = sum(b['count'] for b in bucket_stats.values())
            if total_count > 0:
                ece = sum(
                    b['calibration_error'] * b['count']
                    for b in bucket_stats.values()
                ) / total_count
            else:
                ece = None

            # Determine quality
            if ece is None:
                quality = 'insufficient_data'
            elif ece < 0.05:
                quality = 'good'
            elif ece < 0.10:
                quality = 'moderate'
            else:
                quality = 'poor'

            # Store summary
            if ece is not None:
                self._store_summary(model_type, ece, total_count, quality, lookback_days)

            result = {
                'ece': round(ece, 4) if ece is not None else None,
                'total_predictions': total_count,
                'calibration_quality': quality,
                'buckets': bucket_stats,
                'model_type': model_type,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Audit log
            self.audit_logger.log_event('calibration_run', result)

            return result

        except DatabaseError as e:
            logger.error(f"Failed to calculate ECE: {e}")
            raise

    def _store_bucket_metrics(
        self,
        model_type: str,
        low: int,
        high: int,
        count: int,
        accurate: int,
        expected: float,
        actual: float,
        lookback_days: int
    ):
        """Store bucket metrics to database."""
        period_end = datetime.utcnow().date()
        period_start = period_end - timedelta(days=lookback_days)

        query = """
            INSERT INTO calibration_metrics
            (model_type, confidence_bucket, bucket_low, bucket_high,
             prediction_count, accurate_count, expected_accuracy,
             actual_accuracy, calibration_error, period, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            db.execute(query, (
                model_type, f"{low}-{high}", low, high, count, accurate,
                expected, actual, abs(expected - actual),
                'daily', period_start, period_end
            ))
        except DatabaseError as e:
            logger.warning(f"Failed to store bucket metrics: {e}")

    def _store_summary(
        self,
        model_type: str,
        ece: float,
        total: int,
        quality: str,
        lookback_days: int
    ):
        """Store calibration summary."""
        query = """
            INSERT INTO calibration_summary
            (model_type, ece, total_predictions, calibration_quality, lookback_days)
            VALUES (%s, %s, %s, %s, %s)
        """

        try:
            db.execute(query, (model_type, ece, total, quality, lookback_days))
        except DatabaseError as e:
            logger.warning(f"Failed to store calibration summary: {e}")

    def get_calibration_report(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Get calibration report for API response.

        This is the main entry point for the /analytics/calibration endpoint.
        """
        return self.calculate_ece(model_type=model_type, lookback_days=lookback_days)

    def get_calibration_history(
        self,
        model_type: str = 'price_prediction',
        limit: int = 30
    ) -> List[Dict]:
        """
        Get historical calibration summaries.

        Args:
            model_type: Type of model
            limit: Maximum number of records

        Returns:
            List of historical calibration records
        """
        query = """
            SELECT model_type, ece, total_predictions,
                   calibration_quality, lookback_days, calculated_at
            FROM calibration_summary
            WHERE model_type = %s
            ORDER BY calculated_at DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, (model_type, limit))
            return [
                {
                    'model_type': r['model_type'],
                    'ece': float(r['ece']) if r['ece'] else None,
                    'total_predictions': r['total_predictions'],
                    'calibration_quality': r['calibration_quality'],
                    'lookback_days': r['lookback_days'],
                    'calculated_at': r['calculated_at'].isoformat()
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get calibration history: {e}")
            raise

    def get_current_calibration(
        self,
        model_type: str = 'price_prediction'
    ) -> Optional[Dict]:
        """
        Get most recent calibration status.

        Uses the v_current_calibration view for efficiency.
        """
        query = """
            SELECT model_type, ece, total_predictions,
                   calibration_quality, calculated_at
            FROM v_current_calibration
            WHERE model_type = %s
        """

        try:
            result = db.execute_one(query, (model_type,))
            if result:
                return {
                    'model_type': result['model_type'],
                    'ece': float(result['ece']) if result['ece'] else None,
                    'total_predictions': result['total_predictions'],
                    'calibration_quality': result['calibration_quality'],
                    'calculated_at': result['calculated_at'].isoformat()
                }
            return None

        except DatabaseError as e:
            logger.error(f"Failed to get current calibration: {e}")
            raise

    def get_bucket_details(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 7
    ) -> List[Dict]:
        """
        Get detailed bucket breakdown for recent period.

        Args:
            model_type: Type of model
            lookback_days: Days to look back

        Returns:
            List of bucket details sorted by confidence range
        """
        query = """
            SELECT confidence_bucket, bucket_low, bucket_high,
                   SUM(prediction_count) as total_count,
                   SUM(accurate_count) as total_accurate,
                   AVG(expected_accuracy) as avg_expected,
                   AVG(actual_accuracy) as avg_actual,
                   AVG(calibration_error) as avg_error
            FROM calibration_metrics
            WHERE model_type = %s
              AND created_at >= NOW() - INTERVAL '%s days'
            GROUP BY confidence_bucket, bucket_low, bucket_high
            ORDER BY bucket_low
        """

        try:
            results = db.execute(query, (model_type, lookback_days))
            return [
                {
                    'bucket': r['confidence_bucket'],
                    'range_low': r['bucket_low'],
                    'range_high': r['bucket_high'],
                    'prediction_count': r['total_count'],
                    'accurate_count': r['total_accurate'],
                    'expected_accuracy': round(float(r['avg_expected']), 4),
                    'actual_accuracy': round(float(r['avg_actual']), 4),
                    'calibration_error': round(float(r['avg_error']), 4)
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get bucket details: {e}")
            raise

    def get_calibration_trend(
        self,
        model_type: str = 'price_prediction',
        days: int = 30
    ) -> Dict:
        """
        Analyze calibration trend over time.

        Returns trend direction (improving, stable, declining) based on
        recent ECE values.
        """
        history = self.get_calibration_history(model_type, limit=days)

        if len(history) < 2:
            return {
                'trend': 'insufficient_data',
                'data_points': len(history)
            }

        # Compare recent vs older ECE values
        recent = history[:len(history)//2]
        older = history[len(history)//2:]

        recent_ece = sum(h['ece'] for h in recent if h['ece']) / len(recent)
        older_ece = sum(h['ece'] for h in older if h['ece']) / len(older)

        diff = recent_ece - older_ece

        if diff < -0.01:
            trend = 'improving'
        elif diff > 0.01:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_ece': round(recent_ece, 4),
            'older_ece': round(older_ece, 4),
            'change': round(diff, 4),
            'data_points': len(history)
        }
