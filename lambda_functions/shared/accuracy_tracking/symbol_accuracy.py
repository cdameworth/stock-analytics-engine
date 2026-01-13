"""
Symbol Accuracy Aggregator - PostgreSQL Implementation.

Tracks and analyzes prediction accuracy on a per-symbol basis.
Identifies symbols that consistently underperform and may need
model retraining or exclusion.

Features:
    - Per-symbol accuracy calculation
    - Recommendation type breakdown (BUY/SELL/HOLD)
    - Retraining candidate identification
    - Symbol ranking by performance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..database import db, DatabaseError
from .audit_logger import AccuracyAuditLogger

logger = logging.getLogger(__name__)


class SymbolAccuracyAggregator:
    """
    Aggregate and analyze accuracy by symbol using PostgreSQL.

    Tracks which symbols the model performs well on and which need
    attention or retraining.
    """

    # Accuracy threshold below which a symbol is flagged for retraining
    RETRAINING_THRESHOLD = 0.50

    # Accuracy threshold for "correct" prediction
    ACCURACY_THRESHOLD = 95.0

    # Minimum predictions required for valid accuracy calculation
    MIN_PREDICTIONS = 5

    def __init__(self):
        """Initialize symbol accuracy aggregator."""
        self.audit_logger = AccuracyAuditLogger()

    def aggregate_all_symbols(
        self,
        model_type: str = 'price_prediction',
        lookback_days: int = 30
    ) -> Dict:
        """
        Aggregate accuracy for all symbols.

        Args:
            model_type: Type of model to analyze
            lookback_days: Days to include in analysis

        Returns:
            Dictionary with symbol rankings and retraining candidates

        Example:
            result = aggregator.aggregate_all_symbols(lookback_days=30)
            for symbol in result['retraining_candidates']:
                print(f"{symbol['symbol']}: {symbol['accuracy_rate']:.2%}")
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Select appropriate table
        if model_type == 'time_prediction':
            table = 'time_predictions'
        else:
            table = 'price_predictions'

        query = f"""
            SELECT
                symbol,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN accuracy_pct >= %s THEN 1 ELSE 0 END) as accurate_predictions,
                AVG(error_magnitude) as avg_error_magnitude,
                AVG(confidence_score) as avg_confidence,

                -- Breakdown by recommendation
                SUM(CASE WHEN recommendation = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN recommendation = 'BUY' AND accuracy_pct >= %s THEN 1 ELSE 0 END) as buy_accurate,
                SUM(CASE WHEN recommendation = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN recommendation = 'SELL' AND accuracy_pct >= %s THEN 1 ELSE 0 END) as sell_accurate,
                SUM(CASE WHEN recommendation = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                SUM(CASE WHEN recommendation = 'HOLD' AND accuracy_pct >= %s THEN 1 ELSE 0 END) as hold_accurate
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC
        """

        try:
            results = db.execute(query, (
                self.ACCURACY_THRESHOLD,
                self.ACCURACY_THRESHOLD,
                self.ACCURACY_THRESHOLD,
                self.ACCURACY_THRESHOLD,
                cutoff_date,
                self.MIN_PREDICTIONS
            ))

            # Process results
            symbols = []
            retraining_candidates = []

            for row in results:
                total = row['total_predictions']
                accurate = row['accurate_predictions']
                accuracy_rate = accurate / total if total > 0 else 0
                needs_retraining = accuracy_rate < self.RETRAINING_THRESHOLD

                # Calculate per-recommendation accuracy
                buy_accuracy = (
                    row['buy_accurate'] / row['buy_count']
                    if row['buy_count'] > 0 else None
                )
                sell_accuracy = (
                    row['sell_accurate'] / row['sell_count']
                    if row['sell_count'] > 0 else None
                )
                hold_accuracy = (
                    row['hold_accurate'] / row['hold_count']
                    if row['hold_count'] > 0 else None
                )

                symbol_data = {
                    'symbol': row['symbol'],
                    'total_predictions': total,
                    'accurate_predictions': accurate,
                    'accuracy_rate': round(accuracy_rate, 4),
                    'avg_error_magnitude': (
                        round(float(row['avg_error_magnitude']), 4)
                        if row['avg_error_magnitude'] else None
                    ),
                    'avg_confidence': (
                        round(float(row['avg_confidence']), 4)
                        if row['avg_confidence'] else None
                    ),
                    'needs_retraining': needs_retraining,
                    'buy_count': row['buy_count'],
                    'buy_accuracy': round(buy_accuracy, 4) if buy_accuracy else None,
                    'sell_count': row['sell_count'],
                    'sell_accuracy': round(sell_accuracy, 4) if sell_accuracy else None,
                    'hold_count': row['hold_count'],
                    'hold_accuracy': round(hold_accuracy, 4) if hold_accuracy else None
                }

                symbols.append(symbol_data)
                if needs_retraining:
                    retraining_candidates.append(symbol_data)

                # Store to database
                self._store_symbol_metrics(symbol_data, model_type, lookback_days)

            # Sort by accuracy (best first)
            symbols.sort(key=lambda x: x['accuracy_rate'], reverse=True)
            retraining_candidates.sort(key=lambda x: x['accuracy_rate'])

            result = {
                'symbols': symbols,
                'total_symbols': len(symbols),
                'retraining_candidates': retraining_candidates,
                'retraining_count': len(retraining_candidates),
                'model_type': model_type,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Audit log
            self.audit_logger.log_event('symbol_aggregation', {
                'model_type': model_type,
                'total_symbols': len(symbols),
                'retraining_count': len(retraining_candidates),
                'lookback_days': lookback_days
            })

            return result

        except DatabaseError as e:
            logger.error(f"Failed to aggregate symbol accuracy: {e}")
            raise

    def _store_symbol_metrics(
        self,
        data: Dict,
        model_type: str,
        lookback_days: int
    ):
        """Store symbol metrics to database."""
        period_end = datetime.utcnow().date()
        period_start = period_end - timedelta(days=lookback_days)

        query = """
            INSERT INTO symbol_accuracy
            (symbol, model_type, period, total_predictions, accurate_predictions,
             accuracy_rate, avg_error_magnitude, avg_confidence, needs_retraining,
             buy_count, buy_accuracy, sell_count, sell_accuracy,
             hold_count, hold_accuracy, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, model_type, period, period_end)
            DO UPDATE SET
                total_predictions = EXCLUDED.total_predictions,
                accurate_predictions = EXCLUDED.accurate_predictions,
                accuracy_rate = EXCLUDED.accuracy_rate,
                avg_error_magnitude = EXCLUDED.avg_error_magnitude,
                avg_confidence = EXCLUDED.avg_confidence,
                needs_retraining = EXCLUDED.needs_retraining,
                buy_count = EXCLUDED.buy_count,
                buy_accuracy = EXCLUDED.buy_accuracy,
                sell_count = EXCLUDED.sell_count,
                sell_accuracy = EXCLUDED.sell_accuracy,
                hold_count = EXCLUDED.hold_count,
                hold_accuracy = EXCLUDED.hold_accuracy
        """

        try:
            db.execute(query, (
                data['symbol'], model_type, 'weekly',
                data['total_predictions'], data['accurate_predictions'],
                data['accuracy_rate'], data['avg_error_magnitude'],
                data['avg_confidence'], data['needs_retraining'],
                data['buy_count'], data['buy_accuracy'],
                data['sell_count'], data['sell_accuracy'],
                data['hold_count'], data['hold_accuracy'],
                period_start, period_end
            ))
        except DatabaseError as e:
            logger.warning(f"Failed to store symbol metrics for {data['symbol']}: {e}")

    def get_symbol_ranking(
        self,
        model_type: str = 'price_prediction',
        limit: int = 50
    ) -> Dict:
        """
        Get symbols ranked by accuracy.

        Uses the get_symbol_ranking database function.

        Args:
            model_type: Type of model
            limit: Maximum symbols to return

        Returns:
            Dictionary with ranked symbols
        """
        query = """
            SELECT * FROM get_symbol_ranking(%s, 'weekly', %s)
        """

        try:
            results = db.execute(query, (model_type, limit))

            return {
                'ranking': [
                    {
                        'rank': r['rank'],
                        'symbol': r['symbol'],
                        'accuracy_rate': float(r['accuracy_rate']),
                        'total_predictions': r['total_predictions'],
                        'needs_retraining': r['needs_retraining']
                    }
                    for r in results
                ],
                'model_type': model_type,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to get symbol ranking: {e}")
            raise

    def get_symbol_detail(
        self,
        symbol: str,
        model_type: str = 'price_prediction',
        limit: int = 10
    ) -> Dict:
        """
        Get detailed accuracy for specific symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            model_type: Type of model
            limit: Maximum history records

        Returns:
            Dictionary with symbol details and history
        """
        symbol = symbol.upper()

        # Get latest metrics
        query = """
            SELECT *
            FROM symbol_accuracy
            WHERE symbol = %s
              AND model_type = %s
            ORDER BY period_end DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, (symbol, model_type, limit))

            if not results:
                return {
                    'symbol': symbol,
                    'found': False,
                    'message': f'No data found for symbol {symbol}'
                }

            latest = results[0]
            history = [
                {
                    'period_end': r['period_end'].isoformat() if r['period_end'] else None,
                    'accuracy_rate': float(r['accuracy_rate']),
                    'total_predictions': r['total_predictions'],
                    'needs_retraining': r['needs_retraining']
                }
                for r in results
            ]

            return {
                'symbol': symbol,
                'found': True,
                'model_type': model_type,
                'current': {
                    'accuracy_rate': float(latest['accuracy_rate']),
                    'total_predictions': latest['total_predictions'],
                    'accurate_predictions': latest['accurate_predictions'],
                    'needs_retraining': latest['needs_retraining'],
                    'avg_error_magnitude': (
                        float(latest['avg_error_magnitude'])
                        if latest['avg_error_magnitude'] else None
                    ),
                    'buy_accuracy': (
                        float(latest['buy_accuracy'])
                        if latest['buy_accuracy'] else None
                    ),
                    'sell_accuracy': (
                        float(latest['sell_accuracy'])
                        if latest['sell_accuracy'] else None
                    ),
                    'hold_accuracy': (
                        float(latest['hold_accuracy'])
                        if latest['hold_accuracy'] else None
                    )
                },
                'history': history,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to get symbol detail: {e}")
            raise

    def identify_retraining_candidates(
        self,
        model_type: str = 'price_prediction'
    ) -> List[str]:
        """
        Get list of symbols that need retraining.

        Uses the v_retraining_candidates view.

        Returns:
            List of symbol strings
        """
        query = """
            SELECT symbol, accuracy_rate
            FROM v_retraining_candidates
            WHERE model_type = %s
            ORDER BY accuracy_rate ASC
        """

        try:
            results = db.execute(query, (model_type,))
            return [r['symbol'] for r in results]

        except DatabaseError as e:
            logger.error(f"Failed to identify retraining candidates: {e}")
            raise

    def get_best_performing_symbols(
        self,
        model_type: str = 'price_prediction',
        limit: int = 10
    ) -> List[Dict]:
        """
        Get top performing symbols.

        Args:
            model_type: Type of model
            limit: Number of symbols to return

        Returns:
            List of top performing symbol records
        """
        ranking = self.get_symbol_ranking(model_type, limit)
        return ranking.get('ranking', [])

    def get_worst_performing_symbols(
        self,
        model_type: str = 'price_prediction',
        limit: int = 10
    ) -> List[Dict]:
        """
        Get worst performing symbols.

        Args:
            model_type: Type of model
            limit: Number of symbols to return

        Returns:
            List of worst performing symbol records
        """
        query = """
            SELECT symbol, accuracy_rate, total_predictions, needs_retraining
            FROM symbol_accuracy
            WHERE model_type = %s
              AND period = 'weekly'
              AND period_end = (
                  SELECT MAX(period_end)
                  FROM symbol_accuracy
                  WHERE model_type = %s AND period = 'weekly'
              )
            ORDER BY accuracy_rate ASC
            LIMIT %s
        """

        try:
            results = db.execute(query, (model_type, model_type, limit))
            return [
                {
                    'symbol': r['symbol'],
                    'accuracy_rate': float(r['accuracy_rate']),
                    'total_predictions': r['total_predictions'],
                    'needs_retraining': r['needs_retraining']
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get worst performing symbols: {e}")
            raise

    def get_accuracy_by_recommendation(
        self,
        model_type: str = 'price_prediction'
    ) -> Dict:
        """
        Get accuracy breakdown by recommendation type.

        Returns:
            Dictionary with BUY/SELL/HOLD accuracy stats
        """
        query = """
            SELECT
                SUM(buy_count) as total_buy,
                SUM(CASE WHEN buy_accuracy IS NOT NULL THEN buy_count * buy_accuracy ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN buy_accuracy IS NOT NULL THEN buy_count ELSE 0 END), 0) as avg_buy_accuracy,
                SUM(sell_count) as total_sell,
                SUM(CASE WHEN sell_accuracy IS NOT NULL THEN sell_count * sell_accuracy ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN sell_accuracy IS NOT NULL THEN sell_count ELSE 0 END), 0) as avg_sell_accuracy,
                SUM(hold_count) as total_hold,
                SUM(CASE WHEN hold_accuracy IS NOT NULL THEN hold_count * hold_accuracy ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN hold_accuracy IS NOT NULL THEN hold_count ELSE 0 END), 0) as avg_hold_accuracy
            FROM symbol_accuracy
            WHERE model_type = %s
              AND period = 'weekly'
              AND period_end = (
                  SELECT MAX(period_end)
                  FROM symbol_accuracy
                  WHERE model_type = %s AND period = 'weekly'
              )
        """

        try:
            result = db.execute_one(query, (model_type, model_type))

            if not result:
                return {'error': 'No data available'}

            return {
                'BUY': {
                    'count': result['total_buy'] or 0,
                    'accuracy': (
                        round(float(result['avg_buy_accuracy']), 4)
                        if result['avg_buy_accuracy'] else None
                    )
                },
                'SELL': {
                    'count': result['total_sell'] or 0,
                    'accuracy': (
                        round(float(result['avg_sell_accuracy']), 4)
                        if result['avg_sell_accuracy'] else None
                    )
                },
                'HOLD': {
                    'count': result['total_hold'] or 0,
                    'accuracy': (
                        round(float(result['avg_hold_accuracy']), 4)
                        if result['avg_hold_accuracy'] else None
                    )
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to get accuracy by recommendation: {e}")
            raise
