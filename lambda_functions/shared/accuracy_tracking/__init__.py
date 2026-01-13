"""
Accuracy Tracking Module (PostgreSQL)

Provides comprehensive prediction accuracy tracking, analysis, and reporting
for Railway deployment. This module replaces AWS DynamoDB-based tracking with
PostgreSQL-native implementations.

Components:
    - ConfidenceCalibrationTracker: Calculate Expected Calibration Error (ECE)
    - SymbolAccuracyAggregator: Track per-symbol prediction accuracy
    - DeploymentGate: Enforce quality thresholds before model deployment
    - AccuracyAuditLogger: Create immutable audit records
    - MarketConditionTracker: Track market regimes and correlate with accuracy
    - ErrorDistributionAnalyzer: Analyze error magnitude distribution

Usage:
    from lambda_functions.shared.accuracy_tracking import (
        ConfidenceCalibrationTracker,
        SymbolAccuracyAggregator,
        DeploymentGate
    )

    # Calculate ECE
    tracker = ConfidenceCalibrationTracker()
    result = tracker.calculate_ece(lookback_days=30)
    print(f"ECE: {result['ece']}, Quality: {result['calibration_quality']}")

    # Check deployment gate
    gate = DeploymentGate()
    result = gate.evaluate()
    if result['passed']:
        print("Model ready for deployment!")
    else:
        print(f"Gate failed: {result['failure_reasons']}")
"""

from .audit_logger import AccuracyAuditLogger
from .confidence_calibration import ConfidenceCalibrationTracker
from .symbol_accuracy import SymbolAccuracyAggregator
from .deployment_gate import DeploymentGate
from .market_conditions import MarketConditionTracker
from .error_distribution import ErrorDistributionAnalyzer

__all__ = [
    'AccuracyAuditLogger',
    'ConfidenceCalibrationTracker',
    'SymbolAccuracyAggregator',
    'DeploymentGate',
    'MarketConditionTracker',
    'ErrorDistributionAnalyzer'
]

__version__ = '1.0.0'
