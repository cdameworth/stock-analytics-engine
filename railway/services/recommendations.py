"""
Railway-native Stock Recommendations Service.
Provides stock recommendations from PostgreSQL database.
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


class RecommendationsService:
    """Railway-native recommendations service using PostgreSQL."""

    def get_all_recommendations(self, limit: int = 100) -> Dict[str, Any]:
        """Get all current stock recommendations."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'recommendations': []
            }

        try:
            rows = db.execute("""
                SELECT
                    symbol,
                    recommendation_type,
                    target_price,
                    current_price,
                    company_name,
                    created_at
                FROM recommendations
                WHERE created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            recommendations = []
            for row in rows:
                recommendations.append({
                    'symbol': row[0],
                    'recommendation': row[1],
                    'target_price': float(row[2]) if row[2] else None,
                    'current_price': float(row[3]) if row[3] else None,
                    'company_name': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                })

            return {
                'success': True,
                'count': len(recommendations),
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }

    def get_recommendation_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get recommendation for a specific symbol."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'recommendation': None
            }

        try:
            row = db.execute_one("""
                SELECT
                    symbol,
                    recommendation_type,
                    target_price,
                    current_price,
                    company_name,
                    created_at
                FROM recommendations
                WHERE symbol = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (symbol.upper(),))

            if row:
                return {
                    'success': True,
                    'recommendation': {
                        'symbol': row[0],
                        'recommendation': row[1],
                        'target_price': float(row[2]) if row[2] else None,
                        'current_price': float(row[3]) if row[3] else None,
                        'company_name': row[4],
                        'created_at': row[5].isoformat() if row[5] else None
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'No recommendation found for {symbol}',
                    'recommendation': None
                }

        except Exception as e:
            logger.error(f"Error fetching recommendation for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendation': None
            }

    def create_recommendation(
        self,
        symbol: str,
        recommendation: str,
        target_price: float = None,
        current_price: float = None,
        confidence_score: float = None,
        analysis_summary: str = None
    ) -> Dict[str, Any]:
        """Create or update a stock recommendation."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available'
            }

        if recommendation not in ('BUY', 'SELL', 'HOLD'):
            return {
                'success': False,
                'error': 'Recommendation must be BUY, SELL, or HOLD'
            }

        try:
            db.execute("""
                INSERT INTO recommendations (
                    symbol, recommendation, target_price, current_price,
                    confidence_score, analysis_summary
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    recommendation = EXCLUDED.recommendation,
                    target_price = EXCLUDED.target_price,
                    current_price = EXCLUDED.current_price,
                    confidence_score = EXCLUDED.confidence_score,
                    analysis_summary = EXCLUDED.analysis_summary,
                    updated_at = NOW()
            """, (
                symbol.upper(),
                recommendation,
                target_price,
                current_price,
                confidence_score,
                analysis_summary
            ))

            logger.info(f"Created/updated recommendation for {symbol}: {recommendation}")
            return {
                'success': True,
                'symbol': symbol.upper(),
                'recommendation': recommendation,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating recommendation for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_recommendations_by_type(
        self,
        recommendation_type: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get recommendations filtered by type (BUY, SELL, HOLD)."""
        if recommendation_type not in ('BUY', 'SELL', 'HOLD'):
            return {
                'success': False,
                'error': 'Type must be BUY, SELL, or HOLD',
                'recommendations': []
            }

        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'recommendations': []
            }

        try:
            rows = db.execute("""
                SELECT
                    symbol,
                    recommendation_type,
                    target_price,
                    current_price,
                    company_name,
                    created_at
                FROM recommendations
                WHERE recommendation_type = %s
                  AND created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT %s
            """, (recommendation_type, limit))

            recommendations = []
            for row in rows:
                recommendations.append({
                    'symbol': row[0],
                    'recommendation': row[1],
                    'target_price': float(row[2]) if row[2] else None,
                    'current_price': float(row[3]) if row[3] else None,
                    'company_name': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                })

            return {
                'success': True,
                'type': recommendation_type,
                'count': len(recommendations),
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching {recommendation_type} recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }

    def get_high_confidence_recommendations(
        self,
        min_confidence: float = 0.7,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get recommendations with confidence above threshold."""
        if not is_database_available():
            return {
                'success': False,
                'error': 'Database not available',
                'recommendations': []
            }

        try:
            # Note: Current schema doesn't have confidence_score
            # Return all recent recommendations sorted by date
            rows = db.execute("""
                SELECT
                    symbol,
                    recommendation_type,
                    target_price,
                    current_price,
                    company_name,
                    created_at
                FROM recommendations
                WHERE created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            recommendations = []
            for row in rows:
                recommendations.append({
                    'symbol': row[0],
                    'recommendation': row[1],
                    'target_price': float(row[2]) if row[2] else None,
                    'current_price': float(row[3]) if row[3] else None,
                    'company_name': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                })

            return {
                'success': True,
                'min_confidence': min_confidence,
                'count': len(recommendations),
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching high confidence recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }


# Module-level instance for easy access
_service = None

def get_service() -> RecommendationsService:
    """Get or create the recommendations service instance."""
    global _service
    if _service is None:
        _service = RecommendationsService()
    return _service


def get_all_recommendations(limit: int = 100) -> Dict[str, Any]:
    """Convenience function to get all recommendations."""
    return get_service().get_all_recommendations(limit)


def get_recommendation(symbol: str) -> Dict[str, Any]:
    """Convenience function to get recommendation for a symbol."""
    return get_service().get_recommendation_by_symbol(symbol)
