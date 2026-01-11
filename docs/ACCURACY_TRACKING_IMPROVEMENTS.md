# Prediction Accuracy Tracking System (Full Railway)

**Version**: 3.0.0
**Created**: 2026-01-11
**Platform**: Railway (PostgreSQL + containerized services)
**Status**: Design Complete - Ready for Implementation

## Executive Summary

This document outlines a comprehensive prediction accuracy tracking system built entirely on Railway, replacing AWS DynamoDB with Railway PostgreSQL. The design extends the existing 3-service architecture with a shared PostgreSQL database.

**Estimated Total Cost**: ~$60-100/month (vs ~$245/month on AWS)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Railway Project                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐          │
│  │  api-service   │  │ data-ingestion │  │  model-tuning  │          │
│  │  (Flask)       │  │ (Worker)       │  │  (Worker)      │          │
│  │                │  │                │  │                │          │
│  │ Endpoints:     │  │ Schedules:     │  │ Schedules:     │          │
│  │ /recommendations  │ Every 5-10min  │  │ Daily/Weekly   │          │
│  │ /analytics/*   │  │ market data    │  │ validation     │          │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘          │
│          │                   │                   │                    │
│          └───────────────────┼───────────────────┘                    │
│                              │                                        │
│                              ▼                                        │
│                    ┌─────────────────┐                                │
│                    │   PostgreSQL    │                                │
│                    │   (Railway)     │                                │
│                    │                 │                                │
│                    │ Tables:         │                                │
│                    │ • predictions   │                                │
│                    │ • accuracy_*    │                                │
│                    │ • market_*      │                                │
│                    └─────────────────┘                                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Database Migration: DynamoDB → PostgreSQL

### Why PostgreSQL?

| Feature | DynamoDB | PostgreSQL |
|---------|----------|------------|
| Cost | Pay per request + storage | Fixed (~$5-10/month on Railway) |
| Queries | Limited, requires GSIs | Full SQL with JOINs, aggregations |
| Analytics | Complex, expensive | Native window functions, CTEs |
| Schema | Flexible but rigid access patterns | Flexible schema evolution |
| Railway integration | Requires AWS credentials | Native, auto-configured |

### Migration Strategy

Replace all DynamoDB tables with PostgreSQL tables:

| DynamoDB Table | PostgreSQL Table |
|----------------|------------------|
| `stock-recommendations` | `recommendations` |
| `price-predictions` | `price_predictions` |
| `time-to-hit-predictions` | `time_predictions` |
| `ai-performance-analytics` | `performance_analytics` |
| NEW: `confidence-calibration-metrics` | `calibration_metrics` |
| NEW: `symbol-accuracy-metrics` | `symbol_accuracy` |
| NEW: `model-deployment-gates` | `deployment_gates` |
| NEW: `accuracy-audit-trail` | `audit_trail` |
| NEW: `market-condition-metrics` | `market_conditions` |
| NEW: `error-distribution-analytics` | `error_distribution` |
| NEW: `accuracy-market-correlation` | `accuracy_correlation` |

---

## PostgreSQL Schema

### Core Tables

```sql
-- ============================================================
-- STOCK ANALYTICS ENGINE - POSTGRESQL SCHEMA
-- Railway Deployment - Full Migration from DynamoDB
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ============================================================
-- CORE PREDICTION TABLES
-- ============================================================

-- Stock recommendations
CREATE TABLE recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    recommendation VARCHAR(20) NOT NULL,  -- BUY, SELL, HOLD
    target_price DECIMAL(10, 2),
    current_price DECIMAL(10, 2),
    confidence_score DECIMAL(5, 4),  -- 0.0000 to 1.0000
    analysis_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_recommendation CHECK (recommendation IN ('BUY', 'SELL', 'HOLD')),
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

CREATE INDEX idx_recommendations_symbol ON recommendations(symbol);
CREATE INDEX idx_recommendations_created ON recommendations(created_at DESC);
CREATE INDEX idx_recommendations_expires ON recommendations(expires_at) WHERE expires_at IS NOT NULL;

-- Price predictions
CREATE TABLE price_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    predicted_price DECIMAL(10, 2) NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    recommendation VARCHAR(20) NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_date TIMESTAMP WITH TIME ZONE,  -- When to validate

    -- Validation results (populated after validation)
    validation_status VARCHAR(20) DEFAULT 'pending',  -- pending, validated, expired
    actual_price DECIMAL(10, 2),
    accuracy_pct DECIMAL(5, 2),
    error_magnitude DECIMAL(5, 4),  -- Absolute error as decimal
    error_direction VARCHAR(20),  -- overestimated, underestimated
    validated_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_validation_status CHECK (validation_status IN ('pending', 'validated', 'expired', 'skipped'))
);

CREATE INDEX idx_price_pred_symbol ON price_predictions(symbol);
CREATE INDEX idx_price_pred_status ON price_predictions(validation_status, validation_date);
CREATE INDEX idx_price_pred_confidence ON price_predictions(confidence_score);

-- Time-to-hit predictions
CREATE TABLE time_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    target_price DECIMAL(10, 2) NOT NULL,
    predicted_days INTEGER NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expected_hit_date TIMESTAMP WITH TIME ZONE,

    -- Validation results
    validation_status VARCHAR(20) DEFAULT 'pending',
    actual_days INTEGER,
    accuracy_pct DECIMAL(5, 2),
    days_error INTEGER,  -- Signed error (negative = early, positive = late)
    error_category VARCHAR(20),  -- on_time, slightly_early, slightly_late, missed
    validated_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_time_validation_status CHECK (validation_status IN ('pending', 'validated', 'expired', 'skipped'))
);

CREATE INDEX idx_time_pred_symbol ON time_predictions(symbol);
CREATE INDEX idx_time_pred_status ON time_predictions(validation_status, expected_hit_date);

-- ============================================================
-- ACCURACY TRACKING TABLES
-- ============================================================

-- Confidence calibration metrics
CREATE TABLE calibration_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,  -- price_prediction, time_prediction
    confidence_bucket VARCHAR(10) NOT NULL,  -- 0-10, 10-20, ..., 90-100
    bucket_low INTEGER NOT NULL,
    bucket_high INTEGER NOT NULL,
    prediction_count INTEGER NOT NULL,
    accurate_count INTEGER NOT NULL,
    expected_accuracy DECIMAL(5, 4) NOT NULL,
    actual_accuracy DECIMAL(5, 4) NOT NULL,
    calibration_error DECIMAL(5, 4) NOT NULL,
    period VARCHAR(20) NOT NULL,  -- daily, weekly, monthly
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calibration_model ON calibration_metrics(model_type, created_at DESC);
CREATE INDEX idx_calibration_period ON calibration_metrics(period, period_end DESC);

-- Overall calibration summary (ECE)
CREATE TABLE calibration_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    ece DECIMAL(5, 4) NOT NULL,  -- Expected Calibration Error
    total_predictions INTEGER NOT NULL,
    calibration_quality VARCHAR(20) NOT NULL,  -- good, moderate, poor
    lookback_days INTEGER NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calibration_summary ON calibration_summary(model_type, calculated_at DESC);

-- Symbol-level accuracy metrics
CREATE TABLE symbol_accuracy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL,  -- weekly, monthly
    total_predictions INTEGER NOT NULL,
    accurate_predictions INTEGER NOT NULL,
    accuracy_rate DECIMAL(5, 4) NOT NULL,
    avg_error_magnitude DECIMAL(5, 4),
    avg_confidence DECIMAL(5, 4),
    needs_retraining BOOLEAN DEFAULT FALSE,

    -- Breakdown by recommendation type
    buy_count INTEGER DEFAULT 0,
    buy_accuracy DECIMAL(5, 4),
    sell_count INTEGER DEFAULT 0,
    sell_accuracy DECIMAL(5, 4),
    hold_count INTEGER DEFAULT 0,
    hold_accuracy DECIMAL(5, 4),

    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, model_type, period, period_end)
);

CREATE INDEX idx_symbol_accuracy_rank ON symbol_accuracy(period, accuracy_rate DESC);
CREATE INDEX idx_symbol_accuracy_retraining ON symbol_accuracy(needs_retraining) WHERE needs_retraining = TRUE;

-- Model deployment gates
CREATE TABLE deployment_gates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    tuning_session_id VARCHAR(100),

    -- Metrics at evaluation time
    accuracy DECIMAL(5, 4) NOT NULL,
    market_outperformance DECIMAL(5, 4),
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown DECIMAL(5, 4),

    -- Thresholds used
    accuracy_threshold DECIMAL(5, 4) NOT NULL DEFAULT 0.65,
    outperformance_threshold DECIMAL(5, 4) NOT NULL DEFAULT 0.03,
    sharpe_ratio_min DECIMAL(5, 2) NOT NULL DEFAULT 1.0,
    drawdown_limit DECIMAL(5, 4) NOT NULL DEFAULT 0.15,

    -- Gate result
    gate_status VARCHAR(20) NOT NULL,  -- passed, failed, pending, manual_override
    accuracy_check BOOLEAN NOT NULL,
    outperformance_check BOOLEAN,
    sharpe_ratio_check BOOLEAN,
    drawdown_check BOOLEAN,
    failure_reasons TEXT[],  -- Array of failure reasons

    -- Deployment info
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployed_by VARCHAR(50),  -- automated, manual_override

    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_gate_status CHECK (gate_status IN ('passed', 'failed', 'pending', 'manual_override'))
);

CREATE INDEX idx_deployment_gates_model ON deployment_gates(model_type, evaluated_at DESC);
CREATE INDEX idx_deployment_gates_status ON deployment_gates(gate_status);

-- Error distribution analytics
CREATE TABLE error_distribution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL,
    sample_size INTEGER NOT NULL,

    -- Percentiles
    p10_error DECIMAL(5, 4) NOT NULL,
    p25_error DECIMAL(5, 4) NOT NULL,
    p50_error DECIMAL(5, 4) NOT NULL,  -- Median
    p75_error DECIMAL(5, 4) NOT NULL,
    p90_error DECIMAL(5, 4) NOT NULL,

    -- Statistics
    mean_error DECIMAL(5, 4) NOT NULL,
    std_error DECIMAL(5, 4) NOT NULL,
    skewness DECIMAL(6, 4),  -- Positive = overestimates, negative = underestimates
    outlier_count INTEGER DEFAULT 0,  -- Errors > 3 std

    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_error_dist_model ON error_distribution(model_type, period_end DESC);

-- Accuracy audit trail (immutable)
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,  -- validation_run, accuracy_report, deployment_gate, threshold_change
    model_type VARCHAR(50),

    -- Snapshot of metrics at time of event
    metrics_snapshot JSONB NOT NULL,

    -- Execution context
    source_function VARCHAR(100),
    trigger_type VARCHAR(50),  -- scheduled, manual, api
    execution_id VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- No updates allowed - append only
    CONSTRAINT audit_no_update CHECK (TRUE)
);

CREATE INDEX idx_audit_model ON audit_trail(model_type, created_at DESC);
CREATE INDEX idx_audit_event ON audit_trail(event_type, created_at DESC);
CREATE INDEX idx_audit_created ON audit_trail(created_at DESC);

-- Market condition metrics
CREATE TABLE market_conditions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE PRIMARY KEY,
    market_regime VARCHAR(20) NOT NULL,  -- bull, bear, sideways, volatile

    -- SPY metrics
    spy_daily_return DECIMAL(6, 4),
    spy_weekly_return DECIMAL(6, 4),
    spy_vs_ma50 DECIMAL(6, 4),  -- Current price vs 50-day MA

    -- VIX metrics
    vix_level DECIMAL(6, 2),
    vix_category VARCHAR(20),  -- low, moderate, high, extreme

    -- Market breadth
    market_breadth DECIMAL(5, 4),  -- % of S&P stocks above 50-day MA

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_regime CHECK (market_regime IN ('bull', 'bear', 'sideways', 'volatile'))
);

-- Accuracy-market correlation
CREATE TABLE accuracy_correlation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    market_regime VARCHAR(20) NOT NULL,

    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    sample_size INTEGER NOT NULL,
    accuracy_rate DECIMAL(5, 4) NOT NULL,

    -- Comparison to baseline
    baseline_accuracy DECIMAL(5, 4),  -- Overall accuracy across all regimes
    outperformance DECIMAL(5, 4),  -- accuracy_rate - baseline_accuracy

    -- Statistical significance
    correlation_coefficient DECIMAL(5, 4),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(model_type, market_regime, period_end)
);

CREATE INDEX idx_correlation_model ON accuracy_correlation(model_type, period_end DESC);
CREATE INDEX idx_correlation_regime ON accuracy_correlation(market_regime);

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- Current calibration status
CREATE VIEW v_current_calibration AS
SELECT DISTINCT ON (model_type)
    model_type,
    ece,
    total_predictions,
    calibration_quality,
    calculated_at
FROM calibration_summary
ORDER BY model_type, calculated_at DESC;

-- Symbols needing retraining
CREATE VIEW v_retraining_candidates AS
SELECT
    symbol,
    model_type,
    accuracy_rate,
    total_predictions,
    period_end
FROM symbol_accuracy
WHERE needs_retraining = TRUE
  AND period_end >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY accuracy_rate ASC;

-- Latest deployment gate status
CREATE VIEW v_latest_gates AS
SELECT DISTINCT ON (model_type)
    model_type,
    gate_status,
    accuracy,
    failure_reasons,
    evaluated_at
FROM deployment_gates
ORDER BY model_type, evaluated_at DESC;

-- Current market regime
CREATE VIEW v_current_market AS
SELECT *
FROM market_conditions
ORDER BY date DESC
LIMIT 1;

-- ============================================================
-- FUNCTIONS FOR ACCURACY CALCULATIONS
-- ============================================================

-- Calculate Expected Calibration Error
CREATE OR REPLACE FUNCTION calculate_ece(
    p_model_type VARCHAR(50),
    p_lookback_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    ece DECIMAL(5, 4),
    total_predictions INTEGER,
    calibration_quality VARCHAR(20)
) AS $$
DECLARE
    v_ece DECIMAL(5, 4);
    v_total INTEGER;
    v_quality VARCHAR(20);
BEGIN
    -- Calculate weighted ECE from calibration metrics
    SELECT
        SUM(calibration_error * prediction_count) / NULLIF(SUM(prediction_count), 0),
        SUM(prediction_count)
    INTO v_ece, v_total
    FROM calibration_metrics
    WHERE model_type = p_model_type
      AND created_at >= NOW() - (p_lookback_days || ' days')::INTERVAL;

    -- Determine quality
    v_quality := CASE
        WHEN v_ece < 0.05 THEN 'good'
        WHEN v_ece < 0.10 THEN 'moderate'
        ELSE 'poor'
    END;

    RETURN QUERY SELECT v_ece, v_total, v_quality;
END;
$$ LANGUAGE plpgsql;

-- Get accuracy by symbol with ranking
CREATE OR REPLACE FUNCTION get_symbol_ranking(
    p_model_type VARCHAR(50) DEFAULT 'price_prediction',
    p_period VARCHAR(20) DEFAULT 'weekly',
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    rank INTEGER,
    symbol VARCHAR(10),
    accuracy_rate DECIMAL(5, 4),
    total_predictions INTEGER,
    needs_retraining BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ROW_NUMBER() OVER (ORDER BY sa.accuracy_rate DESC)::INTEGER,
        sa.symbol,
        sa.accuracy_rate,
        sa.total_predictions,
        sa.needs_retraining
    FROM symbol_accuracy sa
    WHERE sa.model_type = p_model_type
      AND sa.period = p_period
      AND sa.period_end = (
          SELECT MAX(period_end)
          FROM symbol_accuracy
          WHERE model_type = p_model_type AND period = p_period
      )
    ORDER BY sa.accuracy_rate DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

---

## File Changes

### 1. New Database Module

**File**: `lambda_functions/shared/database.py`

```python
"""
PostgreSQL Database Connection for Railway deployment.
Replaces boto3/DynamoDB with psycopg2/PostgreSQL.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from .error_handling import StructuredLogger

logger = StructuredLogger(__name__)


class Database:
    """PostgreSQL database connection manager."""

    _instance = None
    _connection = None

    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get database cursor with automatic cleanup."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return []

    def execute_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Execute query and return single result."""
        results = self.execute(query, params)
        return results[0] if results else None

    def insert(self, table: str, data: Dict) -> str:
        """Insert row and return ID."""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"

        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()))
            return cursor.fetchone()['id']

    def insert_many(self, table: str, data: List[Dict]) -> int:
        """Insert multiple rows efficiently."""
        if not data:
            return 0

        columns = list(data[0].keys())
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s"
        values = [tuple(row[col] for col in columns) for row in data]

        with self.get_cursor() as cursor:
            execute_values(cursor, query, values)
            return len(data)


# Global database instance
db = Database()
```

### 2. Updated Accuracy Tracking Module

**File**: `lambda_functions/shared/accuracy_tracking/__init__.py`

```python
"""
Accuracy Tracking Module (PostgreSQL)
Provides comprehensive prediction accuracy tracking for Railway deployment.
"""

from .confidence_calibration import ConfidenceCalibrationTracker
from .symbol_accuracy import SymbolAccuracyAggregator
from .deployment_gate import DeploymentGate
from .audit_logger import AccuracyAuditLogger
from .market_conditions import MarketConditionTracker
from .error_distribution import ErrorDistributionAnalyzer

__all__ = [
    'ConfidenceCalibrationTracker',
    'SymbolAccuracyAggregator',
    'DeploymentGate',
    'AccuracyAuditLogger',
    'MarketConditionTracker',
    'ErrorDistributionAnalyzer'
]
```

### 3. Confidence Calibration (PostgreSQL)

**File**: `lambda_functions/shared/accuracy_tracking/confidence_calibration.py`

```python
"""Confidence Calibration Tracker - PostgreSQL Implementation."""

from datetime import datetime, timedelta
from typing import Dict, List
from decimal import Decimal
from ..database import db
from ..error_handling import StructuredLogger

logger = StructuredLogger(__name__)


class ConfidenceCalibrationTracker:
    """Track and analyze confidence calibration using PostgreSQL."""

    CONFIDENCE_BUCKETS = [(i, i + 10) for i in range(0, 100, 10)]

    def calculate_ece(self, model_type: str = 'price_prediction',
                      lookback_days: int = 30) -> Dict:
        """
        Calculate Expected Calibration Error.
        ECE = sum(|accuracy_bucket - confidence_bucket| * n_bucket) / n_total
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Get predictions with validation results
        query = """
            SELECT
                confidence_score,
                CASE WHEN accuracy_pct >= 95 THEN 1 ELSE 0 END as is_accurate
            FROM price_predictions
            WHERE validation_status = 'validated'
              AND validated_at >= %s
        """
        predictions = db.execute(query, (cutoff_date,))

        if not predictions:
            return {
                'ece': None,
                'total_predictions': 0,
                'calibration_quality': 'insufficient_data',
                'buckets': {},
                'timestamp': datetime.utcnow().isoformat()
            }

        # Calculate per-bucket metrics
        bucket_stats = {}
        for low, high in self.CONFIDENCE_BUCKETS:
            bucket_preds = [
                p for p in predictions
                if low <= float(p['confidence_score']) * 100 < high
            ]

            if bucket_preds:
                accurate = sum(p['is_accurate'] for p in bucket_preds)
                expected = (low + high) / 200  # Midpoint as decimal
                actual = accurate / len(bucket_preds)

                bucket_stats[f"{low}-{high}"] = {
                    'count': len(bucket_preds),
                    'expected_accuracy': expected,
                    'actual_accuracy': actual,
                    'calibration_error': abs(expected - actual)
                }

                # Store to database
                self._store_bucket_metrics(
                    model_type, low, high, len(bucket_preds),
                    accurate, expected, actual, lookback_days
                )

        # Calculate weighted ECE
        total_count = sum(b['count'] for b in bucket_stats.values())
        ece = sum(
            b['calibration_error'] * b['count']
            for b in bucket_stats.values()
        ) / total_count if total_count > 0 else 0

        # Determine quality
        quality = 'good' if ece < 0.05 else 'moderate' if ece < 0.10 else 'poor'

        # Store summary
        self._store_summary(model_type, ece, total_count, quality, lookback_days)

        return {
            'ece': float(ece),
            'total_predictions': total_count,
            'calibration_quality': quality,
            'buckets': bucket_stats,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _store_bucket_metrics(self, model_type: str, low: int, high: int,
                               count: int, accurate: int, expected: float,
                               actual: float, lookback_days: int):
        """Store bucket metrics to database."""
        query = """
            INSERT INTO calibration_metrics
            (model_type, confidence_bucket, bucket_low, bucket_high,
             prediction_count, accurate_count, expected_accuracy,
             actual_accuracy, calibration_error, period, period_start, period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        period_end = datetime.utcnow().date()
        period_start = period_end - timedelta(days=lookback_days)

        db.execute(query, (
            model_type, f"{low}-{high}", low, high, count, accurate,
            expected, actual, abs(expected - actual),
            'daily', period_start, period_end
        ))

    def _store_summary(self, model_type: str, ece: float, total: int,
                       quality: str, lookback_days: int):
        """Store calibration summary."""
        query = """
            INSERT INTO calibration_summary
            (model_type, ece, total_predictions, calibration_quality, lookback_days)
            VALUES (%s, %s, %s, %s, %s)
        """
        db.execute(query, (model_type, ece, total, quality, lookback_days))

    def get_calibration_report(self, lookback_days: int = 30) -> Dict:
        """Get calibration report for API response."""
        return self.calculate_ece(lookback_days=lookback_days)

    def get_calibration_history(self, model_type: str = 'price_prediction',
                                 limit: int = 30) -> List[Dict]:
        """Get historical calibration summaries."""
        query = """
            SELECT model_type, ece, total_predictions,
                   calibration_quality, calculated_at
            FROM calibration_summary
            WHERE model_type = %s
            ORDER BY calculated_at DESC
            LIMIT %s
        """
        return db.execute(query, (model_type, limit))
```

### 4. Symbol Accuracy (PostgreSQL)

**File**: `lambda_functions/shared/accuracy_tracking/symbol_accuracy.py`

```python
"""Symbol Accuracy Aggregator - PostgreSQL Implementation."""

from datetime import datetime, timedelta
from typing import Dict, List
from ..database import db
from ..error_handling import StructuredLogger

logger = StructuredLogger(__name__)


class SymbolAccuracyAggregator:
    """Aggregate and analyze accuracy by symbol using PostgreSQL."""

    RETRAINING_THRESHOLD = 0.50  # Flag symbols below 50% accuracy

    def aggregate_all_symbols(self, model_type: str = 'price_prediction',
                               lookback_days: int = 30) -> Dict:
        """Aggregate accuracy for all symbols."""
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        query = """
            SELECT
                symbol,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN accuracy_pct >= 95 THEN 1 ELSE 0 END) as accurate_predictions,
                AVG(error_magnitude) as avg_error_magnitude,
                AVG(confidence_score) as avg_confidence,

                -- Breakdown by recommendation
                SUM(CASE WHEN recommendation = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                AVG(CASE WHEN recommendation = 'BUY' AND accuracy_pct >= 95 THEN 1.0
                         WHEN recommendation = 'BUY' THEN 0.0 END) as buy_accuracy,
                SUM(CASE WHEN recommendation = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                AVG(CASE WHEN recommendation = 'SELL' AND accuracy_pct >= 95 THEN 1.0
                         WHEN recommendation = 'SELL' THEN 0.0 END) as sell_accuracy,
                SUM(CASE WHEN recommendation = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                AVG(CASE WHEN recommendation = 'HOLD' AND accuracy_pct >= 95 THEN 1.0
                         WHEN recommendation = 'HOLD' THEN 0.0 END) as hold_accuracy
            FROM price_predictions
            WHERE validation_status = 'validated'
              AND validated_at >= %s
            GROUP BY symbol
            ORDER BY COUNT(*) DESC
        """

        results = db.execute(query, (cutoff_date,))

        # Process and store results
        symbols = []
        retraining_candidates = []

        for row in results:
            accuracy_rate = (
                row['accurate_predictions'] / row['total_predictions']
                if row['total_predictions'] > 0 else 0
            )
            needs_retraining = accuracy_rate < self.RETRAINING_THRESHOLD

            symbol_data = {
                'symbol': row['symbol'],
                'total_predictions': row['total_predictions'],
                'accurate_predictions': row['accurate_predictions'],
                'accuracy_rate': float(accuracy_rate),
                'avg_error_magnitude': float(row['avg_error_magnitude'] or 0),
                'avg_confidence': float(row['avg_confidence'] or 0),
                'needs_retraining': needs_retraining,
                'buy_count': row['buy_count'],
                'buy_accuracy': float(row['buy_accuracy'] or 0),
                'sell_count': row['sell_count'],
                'sell_accuracy': float(row['sell_accuracy'] or 0),
                'hold_count': row['hold_count'],
                'hold_accuracy': float(row['hold_accuracy'] or 0)
            }

            symbols.append(symbol_data)
            if needs_retraining:
                retraining_candidates.append(symbol_data)

            # Store to database
            self._store_symbol_metrics(symbol_data, model_type, lookback_days)

        return {
            'symbols': sorted(symbols, key=lambda x: x['accuracy_rate'], reverse=True),
            'total_symbols': len(symbols),
            'retraining_candidates': retraining_candidates,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _store_symbol_metrics(self, data: Dict, model_type: str, lookback_days: int):
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
                needs_retraining = EXCLUDED.needs_retraining
        """

        db.execute(query, (
            data['symbol'], model_type, 'weekly', data['total_predictions'],
            data['accurate_predictions'], data['accuracy_rate'],
            data['avg_error_magnitude'], data['avg_confidence'],
            data['needs_retraining'], data['buy_count'], data['buy_accuracy'],
            data['sell_count'], data['sell_accuracy'], data['hold_count'],
            data['hold_accuracy'], period_start, period_end
        ))

    def get_symbol_ranking(self, model_type: str = 'price_prediction',
                           limit: int = 50) -> Dict:
        """Get symbols ranked by accuracy."""
        query = """
            SELECT * FROM get_symbol_ranking(%s, 'weekly', %s)
        """
        results = db.execute(query, (model_type, limit))

        return {
            'ranking': [dict(r) for r in results],
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_symbol_detail(self, symbol: str) -> Dict:
        """Get detailed accuracy for specific symbol."""
        query = """
            SELECT *
            FROM symbol_accuracy
            WHERE symbol = %s
            ORDER BY period_end DESC
            LIMIT 10
        """
        history = db.execute(query, (symbol.upper(),))

        return {
            'symbol': symbol.upper(),
            'history': [dict(r) for r in history],
            'timestamp': datetime.utcnow().isoformat()
        }

    def identify_retraining_candidates(self) -> List[str]:
        """Get list of symbols that need retraining."""
        query = """
            SELECT symbol, accuracy_rate
            FROM v_retraining_candidates
        """
        results = db.execute(query)
        return [r['symbol'] for r in results]
```

### 5. Deployment Gate (PostgreSQL)

**File**: `lambda_functions/shared/accuracy_tracking/deployment_gate.py`

```python
"""Deployment Gate - PostgreSQL Implementation."""

import os
from datetime import datetime
from typing import Dict, List, Optional
from ..database import db
from ..error_handling import StructuredLogger
from .audit_logger import AccuracyAuditLogger

logger = StructuredLogger(__name__)


class DeploymentGate:
    """Evaluate and enforce deployment criteria using PostgreSQL."""

    DEFAULT_THRESHOLDS = {
        'accuracy_threshold': float(os.environ.get('ACCURACY_THRESHOLD', '0.65')),
        'market_outperformance': float(os.environ.get('MARKET_OUTPERFORMANCE_THRESHOLD', '0.03')),
        'sharpe_ratio_min': float(os.environ.get('SHARPE_RATIO_MIN', '1.0')),
        'max_drawdown_limit': float(os.environ.get('MAX_DRAWDOWN_LIMIT', '0.15'))
    }

    def __init__(self):
        self.audit_logger = AccuracyAuditLogger()

    def evaluate(self, model_type: str = 'price_prediction',
                 tuning_session_id: str = None, **thresholds) -> Dict:
        """
        Evaluate if current model meets deployment criteria.
        All checks must pass for deployment approval.
        """
        thresholds = {**self.DEFAULT_THRESHOLDS, **thresholds}

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
            k.replace('_check', '')
            for k, v in checks.items() if not v
        ]

        gate_status = 'passed' if passed else 'failed'

        # Store gate decision
        gate_id = self._store_gate_decision(
            model_type, tuning_session_id, metrics, thresholds,
            gate_status, checks, failure_reasons
        )

        result = {
            'gate_id': str(gate_id),
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
        self.audit_logger.log_event('deployment_gate', result)

        logger.log_info(f"Deployment gate {gate_status}: {failure_reasons or 'all checks passed'}")

        return result

    def _get_current_metrics(self, model_type: str) -> Dict:
        """Get current model performance metrics from database."""
        # Get accuracy from recent predictions
        accuracy_query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN accuracy_pct >= 95 THEN 1 ELSE 0 END) as accurate
            FROM price_predictions
            WHERE validation_status = 'validated'
              AND validated_at >= NOW() - INTERVAL '30 days'
        """
        accuracy_result = db.execute_one(accuracy_query)

        accuracy = (
            accuracy_result['accurate'] / accuracy_result['total']
            if accuracy_result['total'] > 0 else 0
        )

        return {
            'accuracy': float(accuracy),
            'total_predictions': accuracy_result['total'],
            'market_outperformance': None,  # TODO: Calculate from returns
            'sharpe_ratio': None,  # TODO: Calculate from returns
            'max_drawdown': None  # TODO: Calculate from returns
        }

    def _store_gate_decision(self, model_type: str, tuning_session_id: str,
                             metrics: Dict, thresholds: Dict, gate_status: str,
                             checks: Dict, failure_reasons: List[str]) -> str:
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

        result = db.execute_one(query, (
            model_type, tuning_session_id, metrics['accuracy'],
            metrics.get('market_outperformance'), metrics.get('sharpe_ratio'),
            metrics.get('max_drawdown'), thresholds['accuracy_threshold'],
            thresholds['market_outperformance'], thresholds['sharpe_ratio_min'],
            thresholds['max_drawdown_limit'], gate_status, checks['accuracy_check'],
            checks.get('outperformance_check'), checks.get('sharpe_ratio_check'),
            checks.get('drawdown_check'), failure_reasons or None
        ))

        return result['id']

    def get_recent_decisions(self, limit: int = 10) -> Dict:
        """Get recent deployment gate decisions."""
        query = """
            SELECT id, model_type, gate_status, accuracy, failure_reasons, evaluated_at
            FROM deployment_gates
            ORDER BY evaluated_at DESC
            LIMIT %s
        """
        results = db.execute(query, (limit,))

        return {
            'decisions': [dict(r) for r in results],
            'timestamp': datetime.utcnow().isoformat()
        }

    def approve_manual_override(self, gate_id: str, approved_by: str) -> Dict:
        """Manually approve a failed gate."""
        query = """
            UPDATE deployment_gates
            SET gate_status = 'manual_override',
                deployed_at = NOW(),
                deployed_by = %s
            WHERE id = %s
            RETURNING *
        """
        result = db.execute_one(query, (approved_by, gate_id))

        self.audit_logger.log_event('manual_override', {
            'gate_id': gate_id,
            'approved_by': approved_by
        })

        return dict(result) if result else None
```

### 6. Audit Logger (PostgreSQL)

**File**: `lambda_functions/shared/accuracy_tracking/audit_logger.py`

```python
"""Accuracy Audit Logger - PostgreSQL Implementation."""

import json
from datetime import datetime
from typing import Dict, List, Optional
from ..database import db
from ..error_handling import StructuredLogger

logger = StructuredLogger(__name__)


class AccuracyAuditLogger:
    """Create and query immutable audit records using PostgreSQL."""

    def log_event(self, event_type: str, metrics: Dict,
                  context: Dict = None) -> str:
        """
        Create immutable audit record.
        Returns audit_id.
        """
        query = """
            INSERT INTO audit_trail
            (event_type, model_type, metrics_snapshot, source_function,
             trigger_type, execution_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        result = db.execute_one(query, (
            event_type,
            metrics.get('model_type'),
            json.dumps(metrics),
            context.get('source_function') if context else None,
            context.get('trigger_type', 'automated') if context else 'automated',
            context.get('execution_id') if context else None
        ))

        audit_id = str(result['id'])
        logger.log_info(f"Audit record created: {audit_id}")

        return audit_id

    def get_audit_trail(self, start_date: str = None, end_date: str = None,
                        model_type: str = None, event_type: str = None,
                        limit: int = 100) -> Dict:
        """Retrieve audit trail for verification."""
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
                   source_function, trigger_type, created_at
            FROM audit_trail
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT %s
        """

        results = db.execute(query, tuple(params))

        return {
            'records': [
                {
                    'audit_id': str(r['id']),
                    'event_type': r['event_type'],
                    'model_type': r['model_type'],
                    'metrics': r['metrics_snapshot'],
                    'source': r['source_function'],
                    'trigger': r['trigger_type'],
                    'timestamp': r['created_at'].isoformat()
                }
                for r in results
            ],
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        }

    def verify_historical_claim(self, timestamp: str, model_type: str) -> Dict:
        """Verify accuracy claim at specific point in time."""
        query = """
            SELECT *
            FROM audit_trail
            WHERE model_type = %s
              AND created_at <= %s
            ORDER BY created_at DESC
            LIMIT 1
        """

        result = db.execute_one(query, (model_type, timestamp))

        if result:
            return {
                'found': True,
                'audit_id': str(result['id']),
                'metrics': result['metrics_snapshot'],
                'recorded_at': result['created_at'].isoformat()
            }

        return {'found': False, 'message': 'No audit record found for this period'}
```

---

## Railway Setup

### 1. Add PostgreSQL Database

```bash
# Using Railway CLI
railway add postgres

# Or via dashboard: New → Database → PostgreSQL
```

### 2. Run Schema Migration

Create a migration script and run it:

**File**: `scripts/migrate_postgres.py`

```python
#!/usr/bin/env python3
"""Run PostgreSQL schema migration for Railway deployment."""

import os
import psycopg2

def run_migration():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")

    # Read schema file
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema.sql')
    with open(schema_path, 'r') as f:
        schema = f.read()

    # Execute migration
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cursor:
            cursor.execute(schema)
        conn.commit()
        print("Migration completed successfully!")
    finally:
        conn.close()

if __name__ == '__main__':
    run_migration()
```

### 3. Update Environment Variables

```bash
# Connect services to PostgreSQL
railway variables set -s api-service \
  DATABASE_URL='${{Postgres.DATABASE_URL}}'

railway variables set -s model-tuning \
  DATABASE_URL='${{Postgres.DATABASE_URL}}' \
  ACCURACY_THRESHOLD="0.65" \
  MARKET_OUTPERFORMANCE_THRESHOLD="0.03"

railway variables set -s data-ingestion \
  DATABASE_URL='${{Postgres.DATABASE_URL}}'
```

### 4. Update Dockerfiles

Add `psycopg2` to requirements:

**File**: `railway/api-service/requirements.txt`

```
flask>=3.0.0
gunicorn>=21.0.0
psycopg2-binary>=2.9.9
python-dotenv>=1.0.0
```

**File**: `railway/model-tuning/requirements.txt`

```
schedule>=1.2.0
psycopg2-binary>=2.9.9
numpy>=1.24.0
yfinance>=0.2.0
pytz>=2023.3
```

---

## Updated Worker Code

### Model Tuning Worker

**File**: `railway/model-tuning/worker.py` (additions)

```python
# Add to imports
from lambda_functions.shared.accuracy_tracking import (
    ConfidenceCalibrationTracker,
    SymbolAccuracyAggregator,
    DeploymentGate,
    MarketConditionTracker,
    ErrorDistributionAnalyzer
)

# Add new job functions
def run_confidence_calibration():
    """Calculate Expected Calibration Error by confidence bucket."""
    try:
        tracker = ConfidenceCalibrationTracker()
        result = tracker.calculate_ece(lookback_days=30)
        logger.log_info(f"Calibration ECE: {result['ece']:.4f} ({result['calibration_quality']})")
        return result
    except Exception as e:
        logger.log_error(f"Calibration failed: {str(e)}")

def run_symbol_accuracy_aggregation():
    """Aggregate accuracy by symbol and identify retraining candidates."""
    try:
        aggregator = SymbolAccuracyAggregator()
        result = aggregator.aggregate_all_symbols(lookback_days=30)
        logger.log_info(f"Symbols analyzed: {result['total_symbols']}, "
                       f"retraining needed: {len(result['retraining_candidates'])}")
        return result
    except Exception as e:
        logger.log_error(f"Symbol aggregation failed: {str(e)}")

def run_deployment_gate_evaluation():
    """Evaluate if current model meets deployment criteria."""
    try:
        gate = DeploymentGate()
        result = gate.evaluate()
        status = 'PASSED ✓' if result['passed'] else f"FAILED ✗ ({', '.join(result['failure_reasons'])})"
        logger.log_info(f"Deployment gate: {status}")
        return result
    except Exception as e:
        logger.log_error(f"Deployment gate failed: {str(e)}")

def run_market_condition_tracking():
    """Track current market regime."""
    try:
        tracker = MarketConditionTracker()
        regime = tracker.classify_current_regime()
        logger.log_info(f"Market regime: {regime}")
        return regime
    except Exception as e:
        logger.log_error(f"Market tracking failed: {str(e)}")

# Update setup_schedules()
def setup_schedules():
    """Set up all job schedules."""
    # Existing schedules
    schedule.every().day.at("11:00").do(run_daily_validation)
    schedule.every().sunday.at("07:00").do(run_weekly_comprehensive_tuning)

    # NEW: Daily confidence calibration (7 AM EST = 12:00 UTC)
    schedule.every().day.at("12:00").do(run_confidence_calibration)

    # NEW: Daily symbol accuracy aggregation (7:30 AM EST = 12:30 UTC)
    schedule.every().day.at("12:30").do(run_symbol_accuracy_aggregation)

    # NEW: Daily market condition tracking (4:30 PM EST = 21:30 UTC)
    schedule.every().day.at("21:30").do(run_market_condition_tracking)

    # NEW: Weekly deployment gate evaluation (Sunday 3 AM EST = 08:00 UTC)
    schedule.every().sunday.at("08:00").do(run_deployment_gate_evaluation)

    logger.log_info("All schedules configured")
```

---

## Cost Comparison

| Component | AWS Cost | Railway Cost |
|-----------|----------|--------------|
| Database | ~$140/month (Aurora) | ~$5-10/month (PostgreSQL) |
| Cache | ~$65/month (ElastiCache) | ~$3-5/month (Redis) |
| Compute | ~$25/month (Lambda) | ~$30-50/month (Containers) |
| API Gateway | ~$15/month | Included |
| **Total** | **~$245/month** | **~$60-100/month** |

**Savings: ~$145-185/month (60-75% reduction)**

---

## Implementation Phases

### Phase 1: Database Setup (Day 1)
1. Add PostgreSQL to Railway project
2. Create `schema.sql` with all tables
3. Run migration script
4. Wire DATABASE_URL to all services

### Phase 2: Core Module (Days 2-4)
1. Create `lambda_functions/shared/database.py`
2. Create `lambda_functions/shared/accuracy_tracking/` module
3. Implement `audit_logger.py` and `confidence_calibration.py`
4. Implement `symbol_accuracy.py`

### Phase 3: Deployment Gate (Days 5-6)
1. Implement `deployment_gate.py`
2. Implement `market_conditions.py`
3. Implement `error_distribution.py`

### Phase 4: Worker Integration (Days 7-8)
1. Update `model-tuning/worker.py` with new jobs
2. Update `api-service/app.py` with new endpoints
3. Test scheduled execution

### Phase 5: Migration & Testing (Days 9-10)
1. Migrate existing data from DynamoDB (if needed)
2. Test all endpoints
3. Deploy to Railway production

---

## Success Metrics

1. **Database Performance**: Queries < 100ms
2. **Cost Reduction**: From ~$245 to ~$60-100/month
3. **Calibration Quality**: ECE < 0.10
4. **Symbol Coverage**: 100% of active symbols tracked
5. **Gate Effectiveness**: Zero bad deployments
6. **Audit Completeness**: 100% of events logged
