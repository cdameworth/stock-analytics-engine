-- ============================================================
-- STOCK ANALYTICS ENGINE - POSTGRESQL SCHEMA
-- Railway Deployment - Full Migration from DynamoDB
-- Version: 1.0.0
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ============================================================
-- CORE PREDICTION TABLES
-- ============================================================

-- Stock recommendations
CREATE TABLE IF NOT EXISTS recommendations (
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

CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_recommendations_created ON recommendations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_expires ON recommendations(expires_at) WHERE expires_at IS NOT NULL;

-- Price predictions
CREATE TABLE IF NOT EXISTS price_predictions (
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

    CONSTRAINT valid_price_validation_status CHECK (validation_status IN ('pending', 'validated', 'expired', 'skipped'))
);

CREATE INDEX IF NOT EXISTS idx_price_pred_symbol ON price_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_price_pred_status ON price_predictions(validation_status, validation_date);
CREATE INDEX IF NOT EXISTS idx_price_pred_confidence ON price_predictions(confidence_score);
CREATE INDEX IF NOT EXISTS idx_price_pred_validated ON price_predictions(validated_at DESC) WHERE validated_at IS NOT NULL;

-- Time-to-hit predictions
CREATE TABLE IF NOT EXISTS time_predictions (
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

CREATE INDEX IF NOT EXISTS idx_time_pred_symbol ON time_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_time_pred_status ON time_predictions(validation_status, expected_hit_date);

-- ============================================================
-- ACCURACY TRACKING TABLES
-- ============================================================

-- Confidence calibration metrics (per bucket)
CREATE TABLE IF NOT EXISTS calibration_metrics (
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

CREATE INDEX IF NOT EXISTS idx_calibration_model ON calibration_metrics(model_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calibration_period ON calibration_metrics(period, period_end DESC);

-- Overall calibration summary (ECE)
CREATE TABLE IF NOT EXISTS calibration_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    ece DECIMAL(5, 4) NOT NULL,  -- Expected Calibration Error
    total_predictions INTEGER NOT NULL,
    calibration_quality VARCHAR(20) NOT NULL,  -- good, moderate, poor
    lookback_days INTEGER NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calibration_summary ON calibration_summary(model_type, calculated_at DESC);

-- Symbol-level accuracy metrics
CREATE TABLE IF NOT EXISTS symbol_accuracy (
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

CREATE INDEX IF NOT EXISTS idx_symbol_accuracy_rank ON symbol_accuracy(period, accuracy_rate DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_accuracy_retraining ON symbol_accuracy(needs_retraining) WHERE needs_retraining = TRUE;
CREATE INDEX IF NOT EXISTS idx_symbol_accuracy_symbol ON symbol_accuracy(symbol);

-- Model deployment gates
CREATE TABLE IF NOT EXISTS deployment_gates (
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

CREATE INDEX IF NOT EXISTS idx_deployment_gates_model ON deployment_gates(model_type, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_deployment_gates_status ON deployment_gates(gate_status);

-- Error distribution analytics
CREATE TABLE IF NOT EXISTS error_distribution (
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

CREATE INDEX IF NOT EXISTS idx_error_dist_model ON error_distribution(model_type, period_end DESC);

-- Accuracy audit trail (immutable)
CREATE TABLE IF NOT EXISTS audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,  -- validation_run, accuracy_report, deployment_gate, threshold_change
    model_type VARCHAR(50),

    -- Snapshot of metrics at time of event
    metrics_snapshot JSONB NOT NULL,

    -- Execution context
    source_function VARCHAR(100),
    trigger_type VARCHAR(50),  -- scheduled, manual, api
    execution_id VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_trail(model_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_trail(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_trail(created_at DESC);

-- Market condition metrics
CREATE TABLE IF NOT EXISTS market_conditions (
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
CREATE TABLE IF NOT EXISTS accuracy_correlation (
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

CREATE INDEX IF NOT EXISTS idx_correlation_model ON accuracy_correlation(model_type, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_correlation_regime ON accuracy_correlation(market_regime);

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- Current calibration status
CREATE OR REPLACE VIEW v_current_calibration AS
SELECT DISTINCT ON (model_type)
    model_type,
    ece,
    total_predictions,
    calibration_quality,
    calculated_at
FROM calibration_summary
ORDER BY model_type, calculated_at DESC;

-- Symbols needing retraining
CREATE OR REPLACE VIEW v_retraining_candidates AS
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
CREATE OR REPLACE VIEW v_latest_gates AS
SELECT DISTINCT ON (model_type)
    model_type,
    gate_status,
    accuracy,
    failure_reasons,
    evaluated_at
FROM deployment_gates
ORDER BY model_type, evaluated_at DESC;

-- Current market regime
CREATE OR REPLACE VIEW v_current_market AS
SELECT *
FROM market_conditions
ORDER BY date DESC
LIMIT 1;

-- Pending validations summary
CREATE OR REPLACE VIEW v_pending_validations AS
SELECT
    'price' as prediction_type,
    COUNT(*) as pending_count,
    MIN(validation_date) as earliest_validation,
    MAX(validation_date) as latest_validation
FROM price_predictions
WHERE validation_status = 'pending'
UNION ALL
SELECT
    'time' as prediction_type,
    COUNT(*) as pending_count,
    MIN(expected_hit_date) as earliest_validation,
    MAX(expected_hit_date) as latest_validation
FROM time_predictions
WHERE validation_status = 'pending';

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
        SUM(cm.calibration_error * cm.prediction_count) / NULLIF(SUM(cm.prediction_count), 0),
        SUM(cm.prediction_count)::INTEGER
    INTO v_ece, v_total
    FROM calibration_metrics cm
    WHERE cm.model_type = p_model_type
      AND cm.created_at >= NOW() - (p_lookback_days || ' days')::INTERVAL;

    -- Determine quality
    v_quality := CASE
        WHEN v_ece IS NULL THEN 'insufficient_data'
        WHEN v_ece < 0.05 THEN 'good'
        WHEN v_ece < 0.10 THEN 'moderate'
        ELSE 'poor'
    END;

    RETURN QUERY SELECT v_ece, COALESCE(v_total, 0), v_quality;
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
        ROW_NUMBER() OVER (ORDER BY sa.accuracy_rate DESC)::INTEGER AS rank,
        sa.symbol,
        sa.accuracy_rate,
        sa.total_predictions,
        sa.needs_retraining
    FROM symbol_accuracy sa
    WHERE sa.model_type = p_model_type
      AND sa.period = p_period
      AND sa.period_end = (
          SELECT MAX(sa2.period_end)
          FROM symbol_accuracy sa2
          WHERE sa2.model_type = p_model_type AND sa2.period = p_period
      )
    ORDER BY sa.accuracy_rate DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Get validation statistics
CREATE OR REPLACE FUNCTION get_validation_stats(
    p_lookback_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    model_type VARCHAR(50),
    total_predictions BIGINT,
    validated_count BIGINT,
    pending_count BIGINT,
    validation_rate DECIMAL(5, 4),
    avg_accuracy DECIMAL(5, 4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'price_prediction'::VARCHAR(50) AS model_type,
        COUNT(*)::BIGINT AS total_predictions,
        COUNT(*) FILTER (WHERE validation_status = 'validated')::BIGINT AS validated_count,
        COUNT(*) FILTER (WHERE validation_status = 'pending')::BIGINT AS pending_count,
        (COUNT(*) FILTER (WHERE validation_status = 'validated'))::DECIMAL / NULLIF(COUNT(*), 0) AS validation_rate,
        AVG(accuracy_pct) FILTER (WHERE validation_status = 'validated') / 100.0 AS avg_accuracy
    FROM price_predictions
    WHERE created_at >= NOW() - (p_lookback_days || ' days')::INTERVAL

    UNION ALL

    SELECT
        'time_prediction'::VARCHAR(50) AS model_type,
        COUNT(*)::BIGINT AS total_predictions,
        COUNT(*) FILTER (WHERE validation_status = 'validated')::BIGINT AS validated_count,
        COUNT(*) FILTER (WHERE validation_status = 'pending')::BIGINT AS pending_count,
        (COUNT(*) FILTER (WHERE validation_status = 'validated'))::DECIMAL / NULLIF(COUNT(*), 0) AS validation_rate,
        AVG(accuracy_pct) FILTER (WHERE validation_status = 'validated') / 100.0 AS avg_accuracy
    FROM time_predictions
    WHERE created_at >= NOW() - (p_lookback_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- TRIGGER FOR UPDATED_AT TIMESTAMP
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER update_recommendations_updated_at
    BEFORE UPDATE ON recommendations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- INITIAL DATA / SEED (optional)
-- ============================================================

-- Insert initial market condition if table is empty
INSERT INTO market_conditions (date, market_regime, vix_level, vix_category)
SELECT CURRENT_DATE, 'sideways', 15.0, 'low'
WHERE NOT EXISTS (SELECT 1 FROM market_conditions LIMIT 1);

-- ============================================================
-- GRANTS (adjust based on your Railway user)
-- ============================================================

-- These will be handled by Railway's default user permissions
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO railway_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO railway_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO railway_user;
