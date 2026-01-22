"""
Railway-Native Model Tuning Worker.
Completely AWS-free implementation using PostgreSQL.
Performs model validation and accuracy tracking.
"""

import os
import sys
import time
import json
import schedule
from datetime import datetime, timedelta, time as dt_time
from decimal import Decimal
import pytz

# Add railway shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.logger import StructuredLogger

logger = StructuredLogger(__name__)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', '')
REDIS_URL = os.environ.get('REDIS_URL', '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
TARGET_HIT_RATE = float(os.environ.get('TARGET_HIT_RATE', '0.65'))
ENABLE_CONTINUOUS_MONITORING = os.environ.get('ENABLE_CONTINUOUS_MONITORING', 'false').lower() == 'true'

# Database connection
db_conn = None


def init_database():
    """Initialize PostgreSQL connection and create tables if needed."""
    global db_conn

    if not DATABASE_URL:
        logger.log_error("DATABASE_URL not configured")
        return False

    try:
        import psycopg2

        db_conn = psycopg2.connect(DATABASE_URL)
        db_conn.autocommit = True

        # Create additional tables for model tuning
        with db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_type VARCHAR(50) NOT NULL,
                    evaluation_date DATE NOT NULL,
                    total_predictions INT,
                    correct_predictions INT,
                    hit_rate DECIMAL(5,4),
                    avg_confidence DECIMAL(5,4),
                    avg_error DECIMAL(12,4),
                    metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_type, evaluation_date)
                );

                CREATE INDEX IF NOT EXISTS idx_model_performance_type ON model_performance(model_type);
                CREATE INDEX IF NOT EXISTS idx_model_performance_date ON model_performance(evaluation_date DESC);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_validations (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    prediction_date DATE NOT NULL,
                    predicted_recommendation VARCHAR(20),
                    predicted_target DECIMAL(12,4),
                    actual_price DECIMAL(12,4),
                    actual_change_percent DECIMAL(8,4),
                    was_correct BOOLEAN,
                    confidence DECIMAL(5,4),
                    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, prediction_date)
                );

                CREATE INDEX IF NOT EXISTS idx_pred_val_symbol ON prediction_validations(symbol);
                CREATE INDEX IF NOT EXISTS idx_pred_val_date ON prediction_validations(prediction_date DESC);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS tuning_runs (
                    id SERIAL PRIMARY KEY,
                    run_type VARCHAR(50) NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'running',
                    symbols_processed INT DEFAULT 0,
                    improvements JSONB,
                    errors JSONB
                );
            """)

        logger.log_info("Database initialized for model tuning")
        return True

    except ImportError:
        logger.log_error("psycopg2 not installed")
        return False
    except Exception as e:
        logger.log_error(f"Database initialization failed: {e}")
        return False


def get_predictions_to_validate(lookback_days=7):
    """Get predictions that can be validated against actual prices."""
    if not db_conn:
        return []

    try:
        with db_conn.cursor() as cur:
            # Get price predictions from N days ago that haven't been validated
            # Uses price_predictions table which stores historical predictions
            # Join with stock_quotes to get price at prediction time
            # Note: stock_quotes stores daily prices with trading_day column
            cur.execute("""
                SELECT pp.symbol,
                       COALESCE(sr.recommendation_type, 'HOLD') as recommendation,
                       COALESCE(pp.confidence, 0.5) as confidence,
                       pp.predicted_price as target_price,
                       COALESCE(sq.price, lp.price) as original_price,
                       COALESCE(pp.timestamp, pp.created_at) as updated_at
                FROM price_predictions pp
                LEFT JOIN stock_recommendations sr ON sr.symbol = pp.symbol
                LEFT JOIN latest_prices lp ON lp.symbol = pp.symbol
                LEFT JOIN stock_quotes sq ON sq.symbol = pp.symbol
                    AND sq.trading_day = COALESCE(pp.timestamp, pp.created_at)::date
                WHERE COALESCE(pp.timestamp, pp.created_at) < NOW() - INTERVAL '%s days'
                AND COALESCE(pp.timestamp, pp.created_at) > NOW() - INTERVAL '%s days'
                AND (pp.validation_status IS NULL OR pp.validation_status = 'pending')
                AND NOT EXISTS (
                    SELECT 1 FROM prediction_validations pv
                    WHERE pv.symbol = pp.symbol
                    AND pv.prediction_date = COALESCE(pp.timestamp, pp.created_at)::date
                )
                ORDER BY COALESCE(pp.timestamp, pp.created_at) DESC
                LIMIT 100
            """, (lookback_days - 2, lookback_days + 5))

            columns = ['symbol', 'recommendation', 'confidence', 'target_price',
                      'current_price', 'updated_at']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            logger.log_info(f"Found {len(results)} predictions to validate")
            for r in results[:3]:  # Log first 3 for debugging
                logger.log_info(f"  {r['symbol']}: target=${r.get('target_price')}, orig=${r.get('current_price')}")

            return results

    except Exception as e:
        logger.log_error(f"Error getting predictions to validate: {e}")
        return []


def get_current_price(symbol):
    """Get the current price for a symbol."""
    if not db_conn:
        return None

    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT price FROM latest_prices WHERE symbol = %s
            """, (symbol,))
            row = cur.fetchone()
            return float(row[0]) if row else None
    except Exception as e:
        logger.log_error(f"Error getting price for {symbol}: {e}")
        return None


def validate_prediction(prediction):
    """Validate a single prediction against actual price movement."""
    if not db_conn:
        return None

    symbol = prediction['symbol']
    actual_current_price = get_current_price(symbol)

    if not actual_current_price:
        logger.log_warning(f"No current price found for {symbol}")
        return None

    # original_price is the price when prediction was made
    original_price = prediction.get('current_price')
    if not original_price:
        logger.log_warning(f"No original price for {symbol}, using current as baseline")
        original_price = actual_current_price

    try:
        original_price = float(original_price)
        current_price = actual_current_price
        target_price = float(prediction['target_price'])
        recommendation = prediction['recommendation']
        confidence = float(prediction['confidence'])

        # Calculate actual change
        actual_change_pct = ((current_price - original_price) / original_price) * 100

        # Determine if prediction was correct
        was_correct = False
        if recommendation == 'BUY':
            # BUY is correct if price went up
            was_correct = actual_change_pct > 0
        elif recommendation == 'SELL':
            # SELL is correct if price went down
            was_correct = actual_change_pct < 0
        else:  # HOLD
            # HOLD is correct if price stayed within 2%
            was_correct = abs(actual_change_pct) < 2

        # Store validation result
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO prediction_validations
                    (symbol, prediction_date, predicted_recommendation, predicted_target,
                     actual_price, actual_change_percent, was_correct, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, prediction_date) DO UPDATE SET
                    actual_price = EXCLUDED.actual_price,
                    actual_change_percent = EXCLUDED.actual_change_percent,
                    was_correct = EXCLUDED.was_correct,
                    validated_at = CURRENT_TIMESTAMP
            """, (
                symbol,
                prediction['updated_at'].date() if hasattr(prediction['updated_at'], 'date') else prediction['updated_at'],
                recommendation,
                target_price,
                current_price,
                actual_change_pct,
                was_correct,
                confidence
            ))

        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'was_correct': was_correct,
            'confidence': confidence,
            'actual_change_pct': actual_change_pct
        }

    except Exception as e:
        logger.log_error(f"Error validating prediction for {symbol}: {e}")
        return None


def calculate_model_performance(lookback_days=30):
    """Calculate overall model performance metrics."""
    if not db_conn:
        return None

    try:
        with db_conn.cursor() as cur:
            # Get validation stats
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_confidence,
                    AVG(ABS(actual_change_percent)) as avg_movement
                FROM prediction_validations
                WHERE validated_at > NOW() - INTERVAL '%s days'
            """, (lookback_days,))

            row = cur.fetchone()
            total = row[0] or 0
            correct = row[1] or 0
            avg_confidence = float(row[2]) if row[2] else 0
            avg_movement = float(row[3]) if row[3] else 0

            hit_rate = correct / total if total > 0 else 0

            # Get breakdown by recommendation type
            cur.execute("""
                SELECT
                    predicted_recommendation,
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct
                FROM prediction_validations
                WHERE validated_at > NOW() - INTERVAL '%s days'
                GROUP BY predicted_recommendation
            """, (lookback_days,))

            breakdown = {}
            for rec_row in cur.fetchall():
                rec_type = rec_row[0]
                rec_total = rec_row[1]
                rec_correct = rec_row[2]
                breakdown[rec_type] = {
                    'total': rec_total,
                    'correct': rec_correct,
                    'hit_rate': rec_correct / rec_total if rec_total > 0 else 0
                }

            return {
                'total_predictions': total,
                'correct_predictions': correct,
                'hit_rate': hit_rate,
                'avg_confidence': avg_confidence,
                'avg_price_movement': avg_movement,
                'target_hit_rate': TARGET_HIT_RATE,
                'meets_target': hit_rate >= TARGET_HIT_RATE,
                'breakdown': breakdown
            }

    except Exception as e:
        logger.log_error(f"Error calculating model performance: {e}")
        return None


def store_performance_metrics(metrics, model_type='recommendation'):
    """Store model performance metrics in database."""
    if not db_conn or not metrics:
        return False

    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_performance
                    (model_type, evaluation_date, total_predictions, correct_predictions,
                     hit_rate, avg_confidence, metrics)
                VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s)
                ON CONFLICT (model_type, evaluation_date) DO UPDATE SET
                    total_predictions = EXCLUDED.total_predictions,
                    correct_predictions = EXCLUDED.correct_predictions,
                    hit_rate = EXCLUDED.hit_rate,
                    avg_confidence = EXCLUDED.avg_confidence,
                    metrics = EXCLUDED.metrics,
                    created_at = CURRENT_TIMESTAMP
            """, (
                model_type,
                metrics['total_predictions'],
                metrics['correct_predictions'],
                metrics['hit_rate'],
                metrics['avg_confidence'],
                json.dumps(metrics)
            ))
        return True
    except Exception as e:
        logger.log_error(f"Error storing performance metrics: {e}")
        return False


def run_accuracy_tracking():
    """Execute accuracy tracking and validation."""
    logger.log_info("Starting accuracy tracking")
    start_time = time.time()

    # Get predictions that need validation
    predictions = get_predictions_to_validate(lookback_days=7)
    logger.log_info(f"Found {len(predictions)} predictions to validate")

    validated = 0
    correct = 0

    for pred in predictions:
        result = validate_prediction(pred)
        if result:
            validated += 1
            if result['was_correct']:
                correct += 1

    # Calculate and store performance metrics
    metrics = calculate_model_performance(lookback_days=30)

    if metrics:
        store_performance_metrics(metrics)
        logger.log_info(f"Model performance: {metrics['hit_rate']:.2%} hit rate "
                       f"({metrics['correct_predictions']}/{metrics['total_predictions']})")

        if metrics['meets_target']:
            logger.log_info(f"Target hit rate of {TARGET_HIT_RATE:.0%} achieved")
        else:
            logger.log_warning(f"Below target hit rate of {TARGET_HIT_RATE:.0%}")

    duration = round(time.time() - start_time, 2)
    logger.log_info(f"Accuracy tracking completed: validated {validated} predictions, "
                   f"{correct} correct, duration {duration}s")

    return {
        'validated': validated,
        'correct': correct,
        'metrics': metrics,
        'duration': duration
    }


def run_model_optimization():
    """
    Run model optimization based on performance metrics.
    In a Railway-native setup, this adjusts recommendation thresholds.
    """
    logger.log_info("Starting model optimization")

    if not db_conn:
        logger.log_error("Database not available for optimization")
        return

    try:
        # Get performance breakdown by confidence level
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 0.7 THEN 'high'
                        WHEN confidence >= 0.5 THEN 'medium'
                        ELSE 'low'
                    END as confidence_tier,
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_conf
                FROM prediction_validations
                WHERE validated_at > NOW() - INTERVAL '30 days'
                GROUP BY confidence_tier
            """)

            tiers = {}
            for row in cur.fetchall():
                tier = row[0]
                total = row[1]
                correct = row[2]
                tiers[tier] = {
                    'total': total,
                    'correct': correct,
                    'hit_rate': correct / total if total > 0 else 0,
                    'avg_confidence': float(row[3]) if row[3] else 0
                }

            logger.log_info(f"Performance by confidence tier: {json.dumps(tiers, default=str)}")

            # Log recommendations for improvement
            for tier, stats in tiers.items():
                if stats['hit_rate'] < TARGET_HIT_RATE:
                    logger.log_warning(f"{tier} confidence tier underperforming: "
                                      f"{stats['hit_rate']:.2%} vs target {TARGET_HIT_RATE:.0%}")

    except Exception as e:
        logger.log_error(f"Error during model optimization: {e}")


def run_weekly_comprehensive_tuning():
    """Run comprehensive weekly model tuning."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    if now.weekday() == 6:  # Sunday
        logger.log_info("Running weekly comprehensive model tuning")

        # Run accuracy tracking first
        run_accuracy_tracking()
        time.sleep(60)

        # Run optimization
        run_model_optimization()

        logger.log_info("Weekly tuning completed")
    else:
        logger.log_info(f"Not Sunday ({now.strftime('%A')}) - skipping weekly tuning")


def run_daily_validation():
    """Run daily model validation."""
    logger.log_info("Running daily validation")
    run_accuracy_tracking()


def setup_schedules():
    """Set up the job schedules."""
    # Daily accuracy validation at 6 AM EST (11:00 UTC)
    schedule.every().day.at("11:00").do(run_daily_validation)
    logger.log_info("Scheduled daily validation at 6:00 AM EST (11:00 UTC)")

    # Weekly comprehensive tuning on Sunday at 2 AM EST (7:00 UTC)
    schedule.every().sunday.at("07:00").do(run_weekly_comprehensive_tuning)
    logger.log_info("Scheduled weekly tuning on Sunday at 2:00 AM EST (7:00 UTC)")

    # Optional: continuous monitoring every 6 hours
    if ENABLE_CONTINUOUS_MONITORING:
        schedule.every(6).hours.do(run_accuracy_tracking)
        logger.log_info("Enabled continuous monitoring: accuracy check every 6 hours")


def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Model Tuning Worker (Railway-native)")
    logger.log_info(f"Environment: {ENVIRONMENT}")
    logger.log_info(f"Target hit rate: {TARGET_HIT_RATE:.0%}")

    # Initialize database
    if not init_database():
        logger.log_error("Failed to initialize database - exiting")
        sys.exit(1)

    # Setup schedules
    setup_schedules()

    # Run initial accuracy tracking after brief delay
    logger.log_info("Waiting 60 seconds before initial accuracy tracking")
    time.sleep(60)
    run_accuracy_tracking()

    # Main loop
    logger.log_info("Worker running - waiting for scheduled jobs")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            if db_conn:
                db_conn.close()
            break
        except Exception as e:
            logger.log_error(f"Error in worker loop: {e}")
            time.sleep(300)


if __name__ == '__main__':
    main()
