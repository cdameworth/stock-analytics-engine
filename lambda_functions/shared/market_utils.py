"""
Market Session and Symbol Classification Utilities
Provides market timing and symbol importance classification for business-aware tracing
"""

import logging
from datetime import datetime, time, timezone
from enum import Enum
from typing import Dict, List, Optional, Set
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    # Fallback timezone handling without pytz

logger = logging.getLogger(__name__)

class MarketSession(Enum):
    """Market session states for business-aware sampling"""
    MARKET_HOURS = "market_hours"      # 9:30 AM - 4:00 PM EST
    PRE_MARKET = "pre_market"          # 4:00 AM - 9:30 AM EST
    AFTER_HOURS = "after_hours"        # 4:00 PM - 8:00 PM EST
    CLOSED = "closed"                  # 8:00 PM - 4:00 AM EST
    WEEKEND = "weekend"                # Saturday-Sunday

class SymbolTier(Enum):
    """Symbol importance tiers for business-aware sampling"""
    MEGA_CAP = "mega_cap"              # >$500B market cap - 100% sampling
    LARGE_CAP = "large_cap"            # $10B-$500B - 75% sampling
    MID_CAP = "mid_cap"                # $2B-$10B - 50% sampling
    SMALL_CAP = "small_cap"            # <$2B - 25% sampling
    ETF = "etf"                        # ETFs and indexes - 90% sampling
    UNKNOWN = "unknown"                # Default tier - 25% sampling

# Market cap-based symbol classification
MEGA_CAP_SYMBOLS = {
    # Tech Giants (>$500B)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Financial/Healthcare Giants
    'BRK.A', 'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'MA', 'XOM', 'PG'
}

LARGE_CAP_SYMBOLS = {
    # Major Technology
    'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'UBER', 'SHOP', 'SNOW',
    'PLTR', 'CRWD', 'OKTA', 'TWLO', 'DDOG', 'NET', 'MDB', 'QCOM', 'AVGO',
    # Major Financial
    'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'AXP',
    # Major Healthcare
    'PFE', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY', 'AMGN', 'GILD',
    # Major Consumer
    'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'LULU', 'TJX', 'LOW', 'TGT', 'COST'
}

ETF_SYMBOLS = {
    # Major Index ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'SCHD', 'VYM', 'VUG', 'VTV', 'VEA',
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE', 'XLC', 'XLY',
    # Commodity/International ETFs
    'GDX', 'SLV', 'GLD', 'USO', 'UNG', 'VNQ', 'EFA', 'EEM', 'FXI', 'EWJ'
}

def get_market_session(dt: Optional[datetime] = None) -> MarketSession:
    """
    Determine current market session for business-aware sampling

    Args:
        dt: Optional datetime to check, defaults to current UTC time

    Returns:
        MarketSession enum indicating current market state
    """
    if dt is None:
        dt = datetime.utcnow()

    # Convert to Eastern Time (market timezone)
    if PYTZ_AVAILABLE:
        eastern = pytz.timezone('US/Eastern')
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        et_time = dt.astimezone(eastern)
    else:
        # Fallback: use UTC-5 for Eastern Time (approximation)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Simple UTC-5 offset (not accounting for DST)
        from datetime import timedelta
        et_time = dt - timedelta(hours=5)

    # Check for weekend
    if et_time.weekday() >= 5:  # Saturday=5, Sunday=6
        return MarketSession.WEEKEND

    current_time = et_time.time()

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)

    # Extended hours
    pre_market_start = time(4, 0)    # 4:00 AM ET
    after_hours_end = time(20, 0)    # 8:00 PM ET

    if market_open <= current_time < market_close:
        return MarketSession.MARKET_HOURS
    elif pre_market_start <= current_time < market_open:
        return MarketSession.PRE_MARKET
    elif market_close <= current_time < after_hours_end:
        return MarketSession.AFTER_HOURS
    else:
        return MarketSession.CLOSED

def classify_symbol(symbol: str) -> SymbolTier:
    """
    Classify symbol by market cap tier for sampling strategy

    Args:
        symbol: Stock symbol to classify

    Returns:
        SymbolTier enum indicating symbol importance
    """
    if not symbol:
        return SymbolTier.UNKNOWN

    symbol = symbol.upper()

    if symbol in MEGA_CAP_SYMBOLS:
        return SymbolTier.MEGA_CAP
    elif symbol in LARGE_CAP_SYMBOLS:
        return SymbolTier.LARGE_CAP
    elif symbol in ETF_SYMBOLS:
        return SymbolTier.ETF
    else:
        # Default classification for unknown symbols
        return SymbolTier.SMALL_CAP

def get_sampling_rate(
    symbol: str,
    market_session: Optional[MarketSession] = None,
    is_error: bool = False,
    confidence_score: Optional[float] = None
) -> float:
    """
    Calculate business-aware sampling rate based on market conditions

    Args:
        symbol: Stock symbol being processed
        market_session: Current market session, auto-detected if None
        is_error: Whether this is an error trace (always 100% sampled)
        confidence_score: ML confidence score (low confidence = higher sampling)

    Returns:
        Sampling rate as float between 0.0 and 1.0
    """
    # Always sample errors and low confidence predictions
    if is_error:
        return 1.0

    if confidence_score is not None and confidence_score < 0.7:
        return 1.0  # 100% sampling for low confidence predictions

    if market_session is None:
        market_session = get_market_session()

    symbol_tier = classify_symbol(symbol)

    # Base sampling rates by symbol tier
    tier_rates = {
        SymbolTier.MEGA_CAP: 1.0,    # 100% - Always sample major stocks
        SymbolTier.LARGE_CAP: 0.75,  # 75% - High sampling for large caps
        SymbolTier.ETF: 0.9,         # 90% - High sampling for market indicators
        SymbolTier.MID_CAP: 0.5,     # 50% - Moderate sampling
        SymbolTier.SMALL_CAP: 0.25,  # 25% - Lower sampling
        SymbolTier.UNKNOWN: 0.25     # 25% - Conservative default
    }

    base_rate = tier_rates[symbol_tier]

    # Adjust based on market session
    session_multipliers = {
        MarketSession.MARKET_HOURS: 1.0,     # Full sampling during market
        MarketSession.PRE_MARKET: 0.8,       # 80% during pre-market
        MarketSession.AFTER_HOURS: 0.8,      # 80% during after-hours
        MarketSession.CLOSED: 0.3,           # 30% when market closed
        MarketSession.WEEKEND: 0.1           # 10% on weekends
    }

    session_multiplier = session_multipliers[market_session]
    final_rate = min(base_rate * session_multiplier, 1.0)

    logger.debug(f"Sampling rate for {symbol}: {final_rate:.2f} "
                f"(tier: {symbol_tier.value}, session: {market_session.value})")

    return final_rate

def get_financial_attributes(
    symbol: str,
    current_price: Optional[float] = None,
    target_price: Optional[float] = None,
    confidence_score: Optional[float] = None,
    recommendation: Optional[str] = None
) -> Dict[str, any]:
    """
    Generate comprehensive financial domain attributes for tracing

    Args:
        symbol: Stock symbol
        current_price: Current stock price
        target_price: Predicted target price
        confidence_score: ML model confidence
        recommendation: Buy/sell/hold recommendation

    Returns:
        Dictionary of financial attributes for OpenTelemetry spans
    """
    market_session = get_market_session()
    symbol_tier = classify_symbol(symbol)

    attributes = {
        "finance.symbol": symbol,
        "finance.symbol_tier": symbol_tier.value,
        "finance.market_session": market_session.value,
        "finance.is_market_hours": market_session == MarketSession.MARKET_HOURS,
        "finance.is_major_symbol": symbol_tier in [SymbolTier.MEGA_CAP, SymbolTier.LARGE_CAP]
    }

    if current_price is not None:
        attributes["finance.current_price"] = current_price

    if target_price is not None:
        attributes["finance.target_price"] = target_price

        if current_price is not None:
            price_change_pct = ((target_price - current_price) / current_price) * 100
            attributes["finance.price_change_pct"] = price_change_pct
            attributes["finance.price_direction"] = "up" if price_change_pct > 0 else "down"
            attributes["finance.price_change_magnitude"] = abs(price_change_pct)

    if confidence_score is not None:
        attributes["finance.confidence_score"] = confidence_score
        attributes["finance.high_confidence"] = confidence_score >= 0.8
        attributes["finance.low_confidence"] = confidence_score < 0.7

    if recommendation is not None:
        attributes["finance.recommendation"] = recommendation
        attributes["finance.is_buy_signal"] = recommendation.lower() == "buy"
        attributes["finance.is_sell_signal"] = recommendation.lower() == "sell"

    return attributes

def should_trace_operation(
    operation_type: str,
    symbol: Optional[str] = None,
    confidence_score: Optional[float] = None,
    is_error: bool = False
) -> bool:
    """
    Determine if an operation should be traced based on business rules

    Args:
        operation_type: Type of operation (e.g., 'price_prediction', 'data_ingestion')
        symbol: Stock symbol being processed
        confidence_score: ML confidence score
        is_error: Whether this is an error condition

    Returns:
        Boolean indicating whether to create traces
    """
    # Always trace errors
    if is_error:
        return True

    # Always trace during market hours for major symbols
    market_session = get_market_session()
    if market_session == MarketSession.MARKET_HOURS:
        if symbol and classify_symbol(symbol) in [SymbolTier.MEGA_CAP, SymbolTier.LARGE_CAP]:
            return True

    # Always trace low confidence predictions
    if confidence_score is not None and confidence_score < 0.7:
        return True

    # Use sampling rate for other cases
    if symbol:
        sampling_rate = get_sampling_rate(symbol, market_session, is_error, confidence_score)
        # Simple deterministic sampling based on symbol hash
        import hashlib
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        return (symbol_hash % 100) < (sampling_rate * 100)

    # Default to standard sampling for operations without symbols
    return True

# For testing and validation
def get_market_status_summary() -> Dict[str, any]:
    """Get current market status for monitoring and debugging"""
    current_session = get_market_session()

    return {
        "current_session": current_session.value,
        "timestamp": datetime.utcnow().isoformat(),
        "is_trading_hours": current_session == MarketSession.MARKET_HOURS,
        "sampling_rates": {
            "mega_cap": get_sampling_rate("AAPL"),
            "large_cap": get_sampling_rate("AMD"),
            "etf": get_sampling_rate("SPY"),
            "small_cap": get_sampling_rate("UNKNOWN_SYMBOL")
        }
    }