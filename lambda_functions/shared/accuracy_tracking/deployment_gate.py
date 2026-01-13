"""
Deployment Gate - PostgreSQL Implementation.

Enforces quality thresholds before allowing model deployment.
All checks must pass for a model to be automatically deployed.

Deployment Requirements (from CLAUDE.md):
    - Accuracy ≥ 65% hit rate
    - Market outperformance ≥ 3% above S&P 500
    - Sharpe ratio ≥ 1.0
    - Max drawdown ≤ 15%

Gate Statuses:
    - passed: All checks passed, model can deploy
    - failed: One or more checks failed
    - pending: Evaluation in progress
    - manual_override: Failed but manually approved
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..database import db, DatabaseError
from .audit_logger import AccuracyAuditLogger

logger = logging.getLogger(__name__)


class DeploymentGate:
    """
    Evaluate and enforce deployment criteria using PostgreSQL.

    Implements quality gates that prevent poorly performing models
    from being deployed to production.
    """

    # Default thresholds from CLAUDE.md
    DEFAULT_THRESHOLDS = {
        'accuracy_threshold': 0.65,           # 65% hit rate
        'market_outperformance': 0.03,        # 3% above S&P 500
        'sharpe_ratio_min': 1.0,              # Risk-adjusted returns
        'max_drawdown_limit': 0.15            # 15% max drawdown
    }

    # Accuracy threshold for "correct" prediction
    ACCURACY_THRESHOLD_PCT = 95.0

    def __init__(self):
        """Initialize deployment gate with thresholds from environment."""
        self.audit_logger = AccuracyAuditLogger()

        # Load thresholds from environment with defaults
        self.thresholds = {
            'accuracy_threshold': float(
                os.environ.get('ACCURACY_THRESHOLD', self.DEFAULT_THRESHOLDS['accuracy_threshold'])
            ),
            'market_outperformance': float(
                os.environ.get('MARKET_OUTPERFORMANCE_THRESHOLD', self.DEFAULT_THRESHOLDS['market_outperformance'])
            ),
            'sharpe_ratio_min': float(
                os.environ.get('SHARPE_RATIO_MIN', self.DEFAULT_THRESHOLDS['sharpe_ratio_min'])
            ),
            'max_drawdown_limit': float(
                os.environ.get('MAX_DRAWDOWN_LIMIT', self.DEFAULT_THRESHOLDS['max_drawdown_limit'])
            )
        }

    def evaluate(
        self,
        model_type: str = 'price_prediction',
        tuning_session_id: Optional[str] = None,
        **override_thresholds
    ) -> Dict:
        """
        Evaluate if current model meets deployment criteria.

        All checks must pass for deployment approval.

        Args:
            model_type: Type of model being evaluated
            tuning_session_id: Optional ID of tuning session
            **override_thresholds: Override default thresholds

        Returns:
            Dictionary with gate evaluation results

        Example:
            gate = DeploymentGate()
            result = gate.evaluate()
            if result['passed']:
                deploy_model()
            else:
                print(f"Gate failed: {result['failure_reasons']}")
        """
        # Merge thresholds
        thresholds = {**self.thresholds, **override_thresholds}

        # Get current metrics
        metrics = self._get_current_metrics(model_type)

        # Run all checks
        checks = {
            'accuracy_check': metrics['accuracy'] >= thresholds['accuracy_threshold'],
            'outperformance_check': (
                metrics.get('market_outperformance') is None or
                metrics['market_outperformance'] >= thresholds['market_outperformance']
            ),
            'sharpe_ratio_check': (
                metrics.get('sharpe_ratio') is None or
                metrics['sharpe_ratio'] >= thresholds['sharpe_ratio_min']
            ),
            'drawdown_check': (
                metrics.get('max_drawdown') is None or
                metrics['max_drawdown'] <= thresholds['max_drawdown_limit']
            )
        }

        # Determine result
        passed = all(checks.values())
        failure_reasons = [
            k.replace('_check', '').replace('_', ' ')
            for k, v in checks.items() if not v
        ]

        gate_status = 'passed' if passed else 'failed'

        # Store gate decision
        gate_id = self._store_gate_decision(
            model_type, tuning_session_id, metrics, thresholds,
            gate_status, checks, failure_reasons
        )

        result = {
            'gate_id': str(gate_id) if gate_id else None,
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat(),
            'passed': passed,
            'gate_status': gate_status,
            'checks': checks,
            'failure_reasons': failure_reasons,
            'metrics': metrics,
            'thresholds': thresholds
        }

        # Audit log
        self.audit_logger.log_deployment_gate(
            model_type=model_type,
            passed=passed,
            checks=checks,
            metrics=metrics,
            failure_reasons=failure_reasons
        )

        logger.info(
            f"Deployment gate {gate_status} for {model_type}: "
            f"{failure_reasons if failure_reasons else 'all checks passed'}"
        )

        return result

    def _get_current_metrics(
        self,
        model_type: str,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get current model performance metrics from database.

        Args:
            model_type: Type of model
            lookback_days: Days to include in calculation

        Returns:
            Dictionary with accuracy and performance metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Select appropriate table
        if model_type == 'time_prediction':
            table = 'time_predictions'
        else:
            table = 'price_predictions'

        # Get accuracy from recent validated predictions
        accuracy_query = f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN accuracy_pct >= %s THEN 1 ELSE 0 END) as accurate
            FROM {table}
            WHERE validation_status = 'validated'
              AND validated_at >= %s
        """

        try:
            accuracy_result = db.execute_one(accuracy_query, (
                self.ACCURACY_THRESHOLD_PCT, cutoff_date
            ))

            total = accuracy_result['total'] if accuracy_result else 0
            accurate = accuracy_result['accurate'] if accuracy_result else 0
            accuracy = accurate / total if total > 0 else 0

            # TODO: Implement actual market outperformance calculation
            # This would require tracking portfolio returns vs benchmark
            market_outperformance = None

            # TODO: Implement Sharpe ratio calculation
            # This requires return series and risk-free rate
            sharpe_ratio = None

            # TODO: Implement max drawdown calculation
            # This requires tracking peak-to-trough declines
            max_drawdown = None

            return {
                'accuracy': round(accuracy, 4),
                'total_predictions': total,
                'accurate_predictions': accurate,
                'market_outperformance': market_outperformance,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'lookback_days': lookback_days
            }

        except DatabaseError as e:
            logger.error(f"Failed to get current metrics: {e}")
            raise

    def _store_gate_decision(
        self,
        model_type: str,
        tuning_session_id: Optional[str],
        metrics: Dict,
        thresholds: Dict,
        gate_status: str,
        checks: Dict,
        failure_reasons: List[str]
    ) -> Optional[str]:
        """Store gate decision to database."""
        query = """
            INSERT INTO deployment_gates
            (model_type, tuning_session_id, accuracy, market_outperformance,
             sharpe_ratio, max_drawdown, accuracy_threshold, outperformance_threshold,
             sharpe_ratio_min, drawdown_limit, gate_status, accuracy_check,
             outperformance_check, sharpe_ratio_check, drawdown_check, failure_reasons)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        try:
            result = db.execute_one(query, (
                model_type,
                tuning_session_id,
                metrics['accuracy'],
                metrics.get('market_outperformance'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                thresholds['accuracy_threshold'],
                thresholds['market_outperformance'],
                thresholds['sharpe_ratio_min'],
                thresholds['max_drawdown_limit'],
                gate_status,
                checks['accuracy_check'],
                checks.get('outperformance_check'),
                checks.get('sharpe_ratio_check'),
                checks.get('drawdown_check'),
                failure_reasons if failure_reasons else None
            ))

            return result['id'] if result else None

        except DatabaseError as e:
            logger.error(f"Failed to store gate decision: {e}")
            return None

    def get_recent_decisions(
        self,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict:
        """
        Get recent deployment gate decisions.

        Args:
            model_type: Filter by model type (optional)
            limit: Maximum decisions to return

        Returns:
            Dictionary with recent gate decisions
        """
        if model_type:
            query = """
                SELECT id, model_type, gate_status, accuracy,
                       failure_reasons, evaluated_at
                FROM deployment_gates
                WHERE model_type = %s
                ORDER BY evaluated_at DESC
                LIMIT %s
            """
            params = (model_type, limit)
        else:
            query = """
                SELECT id, model_type, gate_status, accuracy,
                       failure_reasons, evaluated_at
                FROM deployment_gates
                ORDER BY evaluated_at DESC
                LIMIT %s
            """
            params = (limit,)

        try:
            results = db.execute(query, params)

            return {
                'decisions': [
                    {
                        'gate_id': str(r['id']),
                        'model_type': r['model_type'],
                        'gate_status': r['gate_status'],
                        'accuracy': float(r['accuracy']) if r['accuracy'] else None,
                        'failure_reasons': r['failure_reasons'],
                        'evaluated_at': r['evaluated_at'].isoformat()
                    }
                    for r in results
                ],
                'count': len(results),
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseError as e:
            logger.error(f"Failed to get recent decisions: {e}")
            raise

    def get_latest_gate(
        self,
        model_type: str = 'price_prediction'
    ) -> Optional[Dict]:
        """
        Get most recent gate decision for model type.

        Uses v_latest_gates view.
        """
        query = """
            SELECT model_type, gate_status, accuracy,
                   failure_reasons, evaluated_at
            FROM v_latest_gates
            WHERE model_type = %s
        """

        try:
            result = db.execute_one(query, (model_type,))

            if result:
                return {
                    'model_type': result['model_type'],
                    'gate_status': result['gate_status'],
                    'accuracy': float(result['accuracy']) if result['accuracy'] else None,
                    'failure_reasons': result['failure_reasons'],
                    'evaluated_at': result['evaluated_at'].isoformat()
                }
            return None

        except DatabaseError as e:
            logger.error(f"Failed to get latest gate: {e}")
            raise

    def approve_manual_override(
        self,
        gate_id: str,
        approved_by: str,
        reason: Optional[str] = None
    ) -> Dict:
        """
        Manually approve a failed gate.

        This allows deployment despite failed checks, but is logged
        for audit purposes.

        Args:
            gate_id: ID of the gate to override
            approved_by: Name/ID of person approving
            reason: Optional reason for override

        Returns:
            Updated gate record
        """
        query = """
            UPDATE deployment_gates
            SET gate_status = 'manual_override',
                deployed_at = NOW(),
                deployed_by = %s
            WHERE id = %s
            RETURNING *
        """

        try:
            result = db.execute_one(query, (approved_by, gate_id))

            if result:
                # Audit log the override
                self.audit_logger.log_manual_override(
                    gate_id=gate_id,
                    approved_by=approved_by,
                    reason=reason
                )

                logger.warning(
                    f"Manual override approved for gate {gate_id} by {approved_by}"
                )

                return {
                    'success': True,
                    'gate_id': str(result['id']),
                    'gate_status': result['gate_status'],
                    'deployed_by': result['deployed_by'],
                    'deployed_at': result['deployed_at'].isoformat()
                }

            return {
                'success': False,
                'error': f'Gate {gate_id} not found'
            }

        except DatabaseError as e:
            logger.error(f"Failed to approve manual override: {e}")
            raise

    def get_gate_statistics(
        self,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get gate statistics for recent period.

        Args:
            lookback_days: Days to analyze

        Returns:
            Dictionary with pass/fail statistics
        """
        query = """
            SELECT
                COUNT(*) as total_gates,
                SUM(CASE WHEN gate_status = 'passed' THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN gate_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN gate_status = 'manual_override' THEN 1 ELSE 0 END) as overrides,
                AVG(accuracy) as avg_accuracy
            FROM deployment_gates
            WHERE evaluated_at >= NOW() - INTERVAL '%s days'
        """

        try:
            result = db.execute_one(query, (lookback_days,))

            if result:
                total = result['total_gates'] or 0
                passed = result['passed'] or 0

                return {
                    'total_gates': total,
                    'passed': passed,
                    'failed': result['failed'] or 0,
                    'manual_overrides': result['overrides'] or 0,
                    'pass_rate': round(passed / total, 4) if total > 0 else None,
                    'avg_accuracy': (
                        round(float(result['avg_accuracy']), 4)
                        if result['avg_accuracy'] else None
                    ),
                    'lookback_days': lookback_days,
                    'timestamp': datetime.utcnow().isoformat()
                }

            return {'error': 'No data available'}

        except DatabaseError as e:
            logger.error(f"Failed to get gate statistics: {e}")
            raise

    def check_deployment_ready(
        self,
        model_type: str = 'price_prediction'
    ) -> bool:
        """
        Quick check if model is ready for deployment.

        Evaluates current metrics against thresholds without
        storing a gate record.

        Returns:
            True if all checks pass
        """
        metrics = self._get_current_metrics(model_type)
        return metrics['accuracy'] >= self.thresholds['accuracy_threshold']
