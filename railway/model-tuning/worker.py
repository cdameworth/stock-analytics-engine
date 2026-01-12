"""
Model Tuning Worker for Railway deployment - Railway-native implementation.
No AWS dependencies - uses PostgreSQL for all data storage.

Scheduled Jobs:
    - Daily validation (6 AM EST / 11:00 UTC)
    - Daily confidence calibration (7 AM EST / 12:00 UTC)
    - Daily symbol accuracy aggregation (7:30 AM EST / 12:30 UTC)
    - Daily market condition tracking (4:30 PM EST / 21:30 UTC)
    - Weekly deployment gate evaluation (Sunday 3 AM EST / 08:00 UTC)
"""

import os
import sys
import time
import schedule
from datetime import datetime
import pytz

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lambda_functions.shared.error_handling import StructuredLogger

# Import accuracy tracking modules (PostgreSQL-based, Railway-compatible)
from lambda_functions.shared.accuracy_tracking import (
    ConfidenceCalibrationTracker,
    SymbolAccuracyAggregator,
    DeploymentGate,
    MarketConditionTracker,
    ErrorDistributionAnalyzer
)

logger = StructuredLogger(__name__)


def run_daily_validation():
    """Run daily model validation and accuracy tracking."""
    logger.log_info("Running daily validation and accuracy tracking")
    run_confidence_calibration()
    run_symbol_accuracy_aggregation()


def run_confidence_calibration():
    """Calculate Expected Calibration Error by confidence bucket."""
    try:
        logger.log_info("Starting confidence calibration analysis")

        tracker = ConfidenceCalibrationTracker()

        # Run for both model types
        for model_type in ['price_prediction', 'time_prediction']:
            result = tracker.calculate_ece(model_type=model_type, lookback_days=30)

            if result.get('ece') is not None:
                logger.log_info(
                    f"Calibration [{model_type}]: ECE={result['ece']:.4f} "
                    f"({result['calibration_quality']}), "
                    f"predictions={result['total_predictions']}"
                )
            else:
                logger.log_info(f"Calibration [{model_type}]: insufficient data")

    except Exception as e:
        logger.log_error(e, context={'operation': 'confidence_calibration'})


def run_symbol_accuracy_aggregation():
    """Aggregate accuracy by symbol and identify retraining candidates."""
    try:
        logger.log_info("Starting symbol accuracy aggregation")

        aggregator = SymbolAccuracyAggregator()
        result = aggregator.aggregate_all_symbols(lookback_days=30)

        logger.log_info(
            f"Symbol accuracy: {result['total_symbols']} symbols analyzed, "
            f"{result['retraining_count']} need retraining"
        )

        # Log top 5 and bottom 5 symbols
        if result['symbols']:
            top_5 = result['symbols'][:5]
            bottom_5 = result['symbols'][-5:] if len(result['symbols']) > 5 else []

            if top_5:
                logger.log_info("Top 5 symbols: " + ", ".join(
                    f"{s['symbol']}({s['accuracy_rate']:.1%})" for s in top_5
                ))

            if bottom_5:
                logger.log_info("Bottom 5 symbols: " + ", ".join(
                    f"{s['symbol']}({s['accuracy_rate']:.1%})" for s in bottom_5
                ))

    except Exception as e:
        logger.log_error(e, context={'operation': 'symbol_accuracy_aggregation'})


def run_deployment_gate_evaluation():
    """Evaluate if current model meets deployment criteria."""
    try:
        logger.log_info("Starting deployment gate evaluation")

        gate = DeploymentGate()

        for model_type in ['price_prediction', 'time_prediction']:
            result = gate.evaluate(model_type=model_type)

            if result['passed']:
                logger.log_info(
                    f"Deployment gate [{model_type}]: PASSED "
                    f"(accuracy={result['metrics']['accuracy']:.1%})"
                )
            else:
                logger.log_info(
                    f"Deployment gate [{model_type}]: FAILED "
                    f"(reasons: {', '.join(result['failure_reasons'])})"
                )

    except Exception as e:
        logger.log_error(e, context={'operation': 'deployment_gate_evaluation'})


def run_market_condition_tracking():
    """Track current market regime and store conditions."""
    try:
        logger.log_info("Starting market condition tracking")

        tracker = MarketConditionTracker()

        # Classify current regime
        regime = tracker.classify_current_regime()
        logger.log_info(f"Current market regime: {regime}")

        # Get correlation with accuracy
        correlation = tracker.correlate_accuracy_with_regime()

        if 'regimes' in correlation:
            for regime_name, stats in correlation['regimes'].items():
                if stats.get('accuracy_rate') is not None:
                    logger.log_info(
                        f"Accuracy in {regime_name} market: {stats['accuracy_rate']:.1%}"
                    )

    except Exception as e:
        logger.log_error(e, context={'operation': 'market_condition_tracking'})


def run_error_distribution_analysis():
    """Analyze error magnitude distribution."""
    try:
        logger.log_info("Starting error distribution analysis")

        analyzer = ErrorDistributionAnalyzer()

        for model_type in ['price_prediction', 'time_prediction']:
            result = analyzer.calculate_distribution(model_type=model_type, lookback_days=30)

            if 'error' not in result and result.get('p50') is not None:
                logger.log_info(
                    f"Error distribution [{model_type}]: "
                    f"median={result['p50']:.2%}, "
                    f"mean={result['mean']:.2%}, "
                    f"outliers={result['outlier_count']}"
                )
            else:
                logger.log_info(f"Error distribution [{model_type}]: insufficient data")

    except Exception as e:
        logger.log_error(e, context={'operation': 'error_distribution_analysis'})


def setup_schedules():
    """Set up the job schedules."""
    # ============================================================
    # DAILY SCHEDULES
    # ============================================================

    # Daily accuracy validation at 6 AM EST (11:00 UTC)
    schedule.every().day.at("11:00").do(run_daily_validation)
    logger.log_info("Scheduled daily validation at 6:00 AM EST (11:00 UTC)")

    # Daily confidence calibration at 7 AM EST (12:00 UTC)
    schedule.every().day.at("12:00").do(run_confidence_calibration)
    logger.log_info("Scheduled confidence calibration at 7:00 AM EST (12:00 UTC)")

    # Daily symbol accuracy aggregation at 7:30 AM EST (12:30 UTC)
    schedule.every().day.at("12:30").do(run_symbol_accuracy_aggregation)
    logger.log_info("Scheduled symbol accuracy aggregation at 7:30 AM EST (12:30 UTC)")

    # Daily error distribution analysis at 8 AM EST (13:00 UTC)
    schedule.every().day.at("13:00").do(run_error_distribution_analysis)
    logger.log_info("Scheduled error distribution analysis at 8:00 AM EST (13:00 UTC)")

    # Daily market condition tracking at 4:30 PM EST (21:30 UTC)
    schedule.every().day.at("21:30").do(run_market_condition_tracking)
    logger.log_info("Scheduled market condition tracking at 4:30 PM EST (21:30 UTC)")

    # ============================================================
    # WEEKLY SCHEDULES
    # ============================================================

    # Weekly deployment gate evaluation on Sunday at 3 AM EST (8:00 UTC)
    schedule.every().sunday.at("08:00").do(run_deployment_gate_evaluation)
    logger.log_info("Scheduled deployment gate evaluation on Sunday at 3:00 AM EST (8:00 UTC)")

    # ============================================================
    # OPTIONAL CONTINUOUS MONITORING
    # ============================================================

    if os.environ.get('ENABLE_CONTINUOUS_MONITORING', 'false').lower() == 'true':
        schedule.every(6).hours.do(run_confidence_calibration)
        schedule.every(6).hours.do(run_symbol_accuracy_aggregation)
        logger.log_info("Enabled continuous monitoring: accuracy check every 6 hours")


def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Model Tuning Worker (Railway-native)")
    logger.log_info(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    logger.log_info("AWS dependencies: None")

    # Setup schedules
    setup_schedules()

    # Run initial accuracy tracking on startup (after 2 minute delay)
    logger.log_info("Waiting 2 minutes before initial accuracy tracking")
    time.sleep(120)

    logger.log_info("Running initial accuracy tracking")
    run_confidence_calibration()
    run_symbol_accuracy_aggregation()

    # Main loop
    logger.log_info("Worker running - waiting for scheduled jobs")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            break
        except Exception as e:
            logger.log_error(e, context={'operation': 'worker_loop'})
            time.sleep(300)  # Wait 5 minutes before retrying


if __name__ == '__main__':
    main()
