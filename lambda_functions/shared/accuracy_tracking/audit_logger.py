"""
Accuracy Audit Logger - PostgreSQL Implementation.

Creates and queries immutable audit records for accuracy tracking events.
All accuracy-related events are logged for compliance, verification, and
historical analysis.

Event Types:
    - validation_run: Daily/weekly validation execution
    - accuracy_report: Accuracy metrics calculation
    - deployment_gate: Model deployment gate evaluation
    - threshold_change: Configuration threshold changes
    - manual_override: Manual gate approval
    - calibration_run: Confidence calibration calculation
    - symbol_aggregation: Per-symbol accuracy aggregation
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..database import db, DatabaseError

logger = logging.getLogger(__name__)


class AccuracyAuditLogger:
    """
    Create and query immutable audit records using PostgreSQL.

    All records are append-only - no updates or deletes are performed
    on existing audit entries.
    """

    # Valid event types
    EVENT_TYPES = [
        'validation_run',
        'accuracy_report',
        'deployment_gate',
        'threshold_change',
        'manual_override',
        'calibration_run',
        'symbol_aggregation',
        'market_condition',
        'error_distribution'
    ]

    def __init__(self):
        """Initialize audit logger."""
        pass

    def log_event(
        self,
        event_type: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create immutable audit record.

        Args:
            event_type: Type of event (see EVENT_TYPES)
            metrics: Dictionary of metrics/data to snapshot
            context: Optional execution context

        Returns:
            Audit record ID (UUID string)

        Example:
            audit_id = logger.log_event(
                'deployment_gate',
                {'accuracy': 0.72, 'passed': True},
                {'trigger_type': 'scheduled'}
            )
        """
        if event_type not in self.EVENT_TYPES:
            logger.warning(f"Unknown event type: {event_type}")

        context = context or {}

        query = """
            INSERT INTO audit_trail
            (event_type, model_type, metrics_snapshot, source_function,
             trigger_type, execution_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        try:
            result = db.execute_one(query, (
                event_type,
                metrics.get('model_type'),
                json.dumps(metrics, default=str),
                context.get('source_function'),
                context.get('trigger_type', 'automated'),
                context.get('execution_id')
            ))

            audit_id = str(result['id']) if result else None
            logger.info(f"Audit record created: {audit_id} ({event_type})")

            return audit_id

        except DatabaseError as e:
            logger.error(f"Failed to create audit record: {e}")
            raise

    def log_validation_run(
        self,
        model_type: str,
        predictions_validated: int,
        accuracy_rate: float,
        lookback_days: int
    ) -> str:
        """Log a validation run event."""
        return self.log_event(
            'validation_run',
            {
                'model_type': model_type,
                'predictions_validated': predictions_validated,
                'accuracy_rate': accuracy_rate,
                'lookback_days': lookback_days
            },
            {'trigger_type': 'scheduled'}
        )

    def log_deployment_gate(
        self,
        model_type: str,
        passed: bool,
        checks: Dict[str, bool],
        metrics: Dict[str, float],
        failure_reasons: Optional[List[str]] = None
    ) -> str:
        """Log a deployment gate evaluation."""
        return self.log_event(
            'deployment_gate',
            {
                'model_type': model_type,
                'passed': passed,
                'checks': checks,
                'metrics': metrics,
                'failure_reasons': failure_reasons or []
            },
            {'trigger_type': 'scheduled'}
        )

    def log_manual_override(
        self,
        gate_id: str,
        approved_by: str,
        reason: Optional[str] = None
    ) -> str:
        """Log a manual gate override."""
        return self.log_event(
            'manual_override',
            {
                'gate_id': gate_id,
                'approved_by': approved_by,
                'reason': reason
            },
            {'trigger_type': 'manual'}
        )

    def get_audit_trail(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_type: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Retrieve audit trail for verification.

        Args:
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            model_type: Filter by model type
            event_type: Filter by event type
            limit: Maximum records to return

        Returns:
            Dictionary with records and metadata

        Example:
            trail = logger.get_audit_trail(
                start_date='2024-01-01',
                model_type='price_prediction'
            )
        """
        conditions = ["1=1"]
        params = []

        if start_date:
            conditions.append("created_at >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("created_at <= %s")
            params.append(end_date)

        if model_type:
            conditions.append("model_type = %s")
            params.append(model_type)

        if event_type:
            conditions.append("event_type = %s")
            params.append(event_type)

        params.append(limit)

        query = f"""
            SELECT id, event_type, model_type, metrics_snapshot,
                   source_function, trigger_type, execution_id, created_at
            FROM audit_trail
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, tuple(params))

            records = [
                {
                    'audit_id': str(r['id']),
                    'event_type': r['event_type'],
                    'model_type': r['model_type'],
                    'metrics': r['metrics_snapshot'],
                    'source': r['source_function'],
                    'trigger': r['trigger_type'],
                    'execution_id': r['execution_id'],
                    'timestamp': r['created_at'].isoformat() if r['created_at'] else None
                }
                for r in results
            ]

            return {
                'records': records,
                'count': len(records),
                'filters': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'model_type': model_type,
                    'event_type': event_type
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to retrieve audit trail: {e}")
            raise

    def verify_historical_claim(
        self,
        timestamp: str,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Verify accuracy claim at specific point in time.

        Finds the most recent audit record before the given timestamp
        for the specified model type.

        Args:
            timestamp: ISO format timestamp to verify against
            model_type: Model type to verify

        Returns:
            Dictionary with verification result

        Example:
            result = logger.verify_historical_claim(
                '2024-06-15T12:00:00Z',
                'price_prediction'
            )
            if result['found']:
                print(f"Accuracy was: {result['metrics']['accuracy']}")
        """
        query = """
            SELECT id, event_type, metrics_snapshot, created_at
            FROM audit_trail
            WHERE model_type = %s
              AND created_at <= %s
              AND event_type IN ('validation_run', 'accuracy_report', 'deployment_gate')
            ORDER BY created_at DESC
            LIMIT 1
        """

        try:
            result = db.execute_one(query, (model_type, timestamp))

            if result:
                return {
                    'found': True,
                    'audit_id': str(result['id']),
                    'event_type': result['event_type'],
                    'metrics': result['metrics_snapshot'],
                    'recorded_at': result['created_at'].isoformat()
                }

            return {
                'found': False,
                'message': f'No audit record found for {model_type} before {timestamp}'
            }

        except DatabaseError as e:
            logger.error(f"Failed to verify historical claim: {e}")
            raise

    def get_event_counts(
        self,
        lookback_days: int = 30
    ) -> Dict[str, int]:
        """
        Get count of events by type for recent period.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary mapping event type to count
        """
        query = """
            SELECT event_type, COUNT(*) as count
            FROM audit_trail
            WHERE created_at >= NOW() - INTERVAL '%s days'
            GROUP BY event_type
            ORDER BY count DESC
        """

        try:
            results = db.execute(query, (lookback_days,))
            return {r['event_type']: r['count'] for r in results}

        except DatabaseError as e:
            logger.error(f"Failed to get event counts: {e}")
            raise

    def get_recent_gates(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent deployment gate events.

        Args:
            limit: Maximum number of events

        Returns:
            List of gate event records
        """
        query = """
            SELECT id, metrics_snapshot, created_at
            FROM audit_trail
            WHERE event_type = 'deployment_gate'
            ORDER BY created_at DESC
            LIMIT %s
        """

        try:
            results = db.execute(query, (limit,))
            return [
                {
                    'audit_id': str(r['id']),
                    'metrics': r['metrics_snapshot'],
                    'timestamp': r['created_at'].isoformat()
                }
                for r in results
            ]

        except DatabaseError as e:
            logger.error(f"Failed to get recent gates: {e}")
            raise
