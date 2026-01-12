"""
Railway-native Stock Predictions Service.
Provides price and time-to-hit predictions from PostgreSQL database.
No AWS dependencies - designed for Railway deployment.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PostgreSQL database connection
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lambda_functions.shared.database import db, is_database_available


class PredictionsService:
    """Railway-native predictions service using PostgreSQL."""

    def get_price_predictions(
        self,
        symbol: str = None,
        status: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get price predictions, optionally filtered by symbol and status."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'predictions': []
            }

        try:
            query = """
                SELECT
                    id,
                    symbol,
                    predicted_price,
                    confidence,
                    validation_status,
                    timestamp,
                    validation_date,
                    actual_price,
                    accuracy_pct,
                    validated_at
                FROM price_predictions
                WHERE 1=1
            """
            params = []

            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())

            if status:
                query += " AND validation_status = %s"
                params.append(status)

            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            rows = db.execute(query, tuple(params) if params else None)

            predictions = []
            for row in rows:
                predictions.append({
                    'id': str(row[0]) if row[0] else None,
                    'symbol': row[1],
                    'predicted_price': float(row[2]) if row[2] else None,
                    'confidence_score': float(row[3]) if row[3] else None,
                    'validation_status': row[4],
                    'prediction_date': row[5].isoformat() if row[5] else None,
                    'validation_date': row[6].isoformat() if row[6] else None,
                    'actual_price': float(row[7]) if row[7] else None,
                    'accuracy_pct': float(row[8]) if row[8] else None,
                    'validated_at': row[9].isoformat() if row[9] else None
                })

            return {
                'success': True,
                'count': len(predictions),
                'predictions': predictions,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching price predictions: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

    def get_time_predictions(
        self,
        symbol: str = None,
        status: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get time-to-hit predictions, optionally filtered."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'predictions': []
            }

        try:
            query = """
                SELECT
                    id,
                    symbol,
                    target_price,
                    predicted_days,
                    confidence_score,
                    prediction_date,
                    expected_hit_date,
                    validation_status,
                    actual_days,
                    accuracy_pct,
                    validated_at
                FROM time_predictions
                WHERE 1=1
            """
            params = []

            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())

            if status:
                query += " AND validation_status = %s"
                params.append(status)

            query += " ORDER BY prediction_date DESC LIMIT %s"
            params.append(limit)

            rows = db.execute(query, tuple(params) if params else None)

            predictions = []
            for row in rows:
                predictions.append({
                    'id': str(row[0]) if row[0] else None,
                    'symbol': row[1],
                    'target_price': float(row[2]) if row[2] else None,
                    'predicted_days': row[3],
                    'confidence_score': float(row[4]) if row[4] else None,
                    'prediction_date': row[5].isoformat() if row[5] else None,
                    'expected_hit_date': row[6].isoformat() if row[6] else None,
                    'validation_status': row[7],
                    'actual_days': row[8],
                    'accuracy_pct': float(row[9]) if row[9] else None,
                    'validated_at': row[10].isoformat() if row[10] else None
                })

            return {
                'success': True,
                'count': len(predictions),
                'predictions': predictions,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching time predictions: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

    def create_price_prediction(
        self,
        symbol: str,
        predicted_price: float,
        confidence_score: float,
        recommendation: str = 'HOLD',
        validation_days: int = 30
    ) -> Dict[str, Any]:
        """Create a new price prediction."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available'
            }

        try:
            validation_date = datetime.utcnow() + timedelta(days=validation_days)

            db.execute("""
                INSERT INTO price_predictions (
                    symbol, predicted_price, confidence_score, recommendation,
                    prediction_date, validation_date, validation_status
                ) VALUES (%s, %s, %s, %s, NOW(), %s, 'pending')
            """, (
                symbol.upper(),
                predicted_price,
                confidence_score,
                recommendation,
                validation_date
            ))

            logger.info(f"Created price prediction for {symbol}: ${predicted_price:.2f}")
            return {
                'success': True,
                'symbol': symbol.upper(),
                'predicted_price': predicted_price,
                'confidence_score': confidence_score,
                'validation_date': validation_date.isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating price prediction for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_time_prediction(
        self,
        symbol: str,
        target_price: float,
        predicted_days: int,
        confidence_score: float
    ) -> Dict[str, Any]:
        """Create a new time-to-hit prediction."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available'
            }

        try:
            expected_hit_date = datetime.utcnow() + timedelta(days=predicted_days)

            db.execute("""
                INSERT INTO time_predictions (
                    symbol, target_price, predicted_days, confidence_score,
                    prediction_date, expected_hit_date, validation_status
                ) VALUES (%s, %s, %s, %s, NOW(), %s, 'pending')
            """, (
                symbol.upper(),
                target_price,
                predicted_days,
                confidence_score,
                expected_hit_date
            ))

            logger.info(f"Created time prediction for {symbol}: {predicted_days} days to ${target_price:.2f}")
            return {
                'success': True,
                'symbol': symbol.upper(),
                'target_price': target_price,
                'predicted_days': predicted_days,
                'confidence_score': confidence_score,
                'expected_hit_date': expected_hit_date.isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating time prediction for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_pending_validations(self) -> Dict[str, Any]:
        """Get predictions that are due for validation."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'price_predictions': [],
                'time_predictions': []
            }

        try:
            # Price predictions due for validation
            price_rows = db.execute("""
                SELECT
                    id, symbol, predicted_price, confidence_score,
                    prediction_date, validation_date
                FROM price_predictions
                WHERE validation_status = 'pending'
                  AND validation_date <= NOW()
                ORDER BY validation_date ASC
                LIMIT 100
            """)

            price_predictions = []
            for row in price_rows:
                price_predictions.append({
                    'id': str(row[0]),
                    'symbol': row[1],
                    'predicted_price': float(row[2]) if row[2] else None,
                    'confidence_score': float(row[3]) if row[3] else None,
                    'prediction_date': row[4].isoformat() if row[4] else None,
                    'validation_date': row[5].isoformat() if row[5] else None
                })

            # Time predictions due for validation
            time_rows = db.execute("""
                SELECT
                    id, symbol, target_price, predicted_days,
                    confidence_score, prediction_date, expected_hit_date
                FROM time_predictions
                WHERE validation_status = 'pending'
                  AND expected_hit_date <= NOW()
                ORDER BY expected_hit_date ASC
                LIMIT 100
            """)

            time_predictions = []
            for row in time_rows:
                time_predictions.append({
                    'id': str(row[0]),
                    'symbol': row[1],
                    'target_price': float(row[2]) if row[2] else None,
                    'predicted_days': row[3],
                    'confidence_score': float(row[4]) if row[4] else None,
                    'prediction_date': row[5].isoformat() if row[5] else None,
                    'expected_hit_date': row[6].isoformat() if row[6] else None
                })

            return {
                'success': True,
                'price_predictions_count': len(price_predictions),
                'time_predictions_count': len(time_predictions),
                'price_predictions': price_predictions,
                'time_predictions': time_predictions,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching pending validations: {e}")
            return {
                'success': False,
                'error': str(e),
                'price_predictions': [],
                'time_predictions': []
            }

    def validate_price_prediction(
        self,
        prediction_id: str,
        actual_price: float
    ) -> Dict[str, Any]:
        """Validate a price prediction with actual price."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available'
            }

        try:
            # Get the prediction
            row = db.execute_one("""
                SELECT predicted_price, confidence_score
                FROM price_predictions
                WHERE id = %s
            """, (prediction_id,))

            if not row:
                return {
                    'success': False,
                    'error': f'Prediction {prediction_id} not found'
                }

            predicted_price = float(row[0])

            # Calculate accuracy
            error = abs(actual_price - predicted_price)
            error_pct = (error / predicted_price) * 100
            accuracy_pct = max(0, 100 - error_pct)
            error_magnitude = error / predicted_price

            # Determine direction
            if actual_price > predicted_price:
                error_direction = 'underestimated'
            elif actual_price < predicted_price:
                error_direction = 'overestimated'
            else:
                error_direction = 'exact'

            # Update the prediction
            db.execute("""
                UPDATE price_predictions
                SET validation_status = 'validated',
                    actual_price = %s,
                    accuracy_pct = %s,
                    error_magnitude = %s,
                    validated_at = NOW()
                WHERE id = %s
            """, (actual_price, accuracy_pct, error_magnitude, prediction_id))

            logger.info(f"Validated prediction {prediction_id}: {accuracy_pct:.1f}% accurate")
            return {
                'success': True,
                'prediction_id': prediction_id,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'accuracy_pct': accuracy_pct,
                'error_direction': error_direction,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error validating prediction {prediction_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available'
            }

        try:
            # Overall stats
            stats = db.execute_one("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE validation_status = 'validated') as validated,
                    COUNT(*) FILTER (WHERE validation_status = 'pending') as pending,
                    AVG(accuracy_pct) FILTER (WHERE validation_status = 'validated') as avg_accuracy,
                    AVG(confidence) as avg_confidence
                FROM price_predictions
                WHERE timestamp > NOW() - INTERVAL '30 days'
            """)

            # Accuracy by confidence bucket
            buckets = db.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 0.9 THEN '90-100%'
                        WHEN confidence >= 0.8 THEN '80-90%'
                        WHEN confidence >= 0.7 THEN '70-80%'
                        WHEN confidence >= 0.6 THEN '60-70%'
                        ELSE '< 60%'
                    END as bucket,
                    COUNT(*) as count,
                    AVG(accuracy_pct) as avg_accuracy
                FROM price_predictions
                WHERE validation_status = 'validated'
                  AND timestamp > NOW() - INTERVAL '30 days'
                GROUP BY bucket
                ORDER BY bucket DESC
            """)

            bucket_stats = {}
            for row in buckets:
                bucket_stats[row[0]] = {
                    'count': row[1],
                    'avg_accuracy': float(row[2]) if row[2] else None
                }

            return {
                'success': True,
                'period': '30 days',
                'total_predictions': stats[0] if stats else 0,
                'validated_predictions': stats[1] if stats else 0,
                'pending_predictions': stats[2] if stats else 0,
                'avg_accuracy': float(stats[3]) if stats and stats[3] else None,
                'avg_confidence': float(stats[4]) if stats and stats[4] else None,
                'accuracy_by_confidence': bucket_stats,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching dashboard: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Module-level instance
_service = None

def get_service() -> PredictionsService:
    """Get or create the predictions service instance."""
    global _service
    if _service is None:
        _service = PredictionsService()
    return _service


def get_price_predictions(symbol: str = None, limit: int = 100) -> Dict[str, Any]:
    """Convenience function to get price predictions."""
    return get_service().get_price_predictions(symbol=symbol, limit=limit)


def get_time_predictions(symbol: str = None, limit: int = 100) -> Dict[str, Any]:
    """Convenience function to get time predictions."""
    return get_service().get_time_predictions(symbol=symbol, limit=limit)


def get_dashboard() -> Dict[str, Any]:
    """Convenience function to get analytics dashboard."""
    return get_service().get_analytics_dashboard()
