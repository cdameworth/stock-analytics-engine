"""
Stock Data Ingestion Lambda Function
Fetches stock data from Alpha Vantage API and processes it
"""
import json
import boto3
import logging
from datetime import datetime
from decimal import Decimal
import os
import urllib.request
import urllib.parse
import time
import socket

# Enhanced logging and observability
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# OpenTelemetry imports for business-aware tracing
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Enhanced tracing utilities with EventBridge support
try:
    from shared.business_tracing import (
        get_financial_tracer, trace_data_ingestion, propagate_correlation_context
    )
    from shared.market_utils import get_market_session, classify_symbol
    from shared.eventbridge_tracing import (
        get_eventbridge_integration, trace_eventbridge_handler
    )
    BUSINESS_TRACING_AVAILABLE = True
except ImportError:
    BUSINESS_TRACING_AVAILABLE = False
    logger.warning("Business tracing modules not available, using basic tracing")

# Initialize tracer
if BUSINESS_TRACING_AVAILABLE:
    financial_tracer = get_financial_tracer("stock_data_ingestion")
else:
    tracer = trace.get_tracer(__name__)

# Initialize CloudWatch for custom metrics
cloudwatch = boto3.client('cloudwatch')

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
secrets_manager = boto3.client('secretsmanager')

# Global variable to cache API key
ALPHA_VANTAGE_API_KEY = None

CONNECT_TEST_HOST = os.environ.get('CONNECT_TEST_HOST', 'www.alphavantage.co')
CONNECT_TEST_PORT = 443
CONNECT_TEST_TIMEOUT = int(os.environ.get('CONNECT_TEST_TIMEOUT', '2'))
ABORT_AFTER_SEC = int(os.environ.get('ABORT_AFTER_SEC', '40'))
ALPHA_VANTAGE_API_KEY_SECRET_ARN = os.environ.get('ALPHA_VANTAGE_API_KEY_SECRET_ARN', '')
USE_PREMIUM_API_KEY = os.environ.get('USE_PREMIUM_API_KEY', 'false').lower() == 'true'
PREMIUM_API_CALLS_PER_MINUTE = int(os.environ.get('PREMIUM_API_CALLS_PER_MINUTE', '5'))
S3_BUCKET             = os.environ['S3_BUCKET']
DYNAMODB_TABLE        = os.environ['DYNAMODB_TABLE']

# Prefer VALKEY_ENDPOINT; fallback to legacy REDIS_ENDPOINT
VALKEY_ENDPOINT = os.environ.get('VALKEY_ENDPOINT') or os.environ.get('REDIS_ENDPOINT')

# MAXIMIZED throughput configuration
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '12'))  # Increased from 5 to 12
PER_CALL_TIMEOUT    = int(os.environ.get('PER_CALL_TIMEOUT', '6'))      # Reduced from 8 to 6 seconds for faster processing
ML_INFERENCE_FUNCTION_NAME = os.environ.get('ML_INFERENCE_FUNCTION_NAME', 'ml-model-inference-tier')

MAJOR_INDEXES   = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'SCHD', 'VYM', 'VUG', 'VTV', 'VEA']

# MAXIMIZED stock universe for comprehensive market coverage
# Target: 300+ stocks to fully utilize 1,440 daily capacity (5 stocks × 288 runs)

MEGA_CAP_STOCKS = [
    # Tech Giants (>$500B market cap)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'BRK.A', 'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG'
]

LARGE_CAP_TECH = [
    # Major Technology Companies (Expanded)
    'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'UBER', 'SHOP', 'SNOW',
    'PLTR', 'DOCU', 'ZM', 'CRWD', 'OKTA', 'TWLO', 'DDOG', 'NET', 'MDB',
    'TEAM', 'WDAY', 'VEEV', 'SPLK', 'PANW', 'FTNT', 'INTU', 'NOW', 'ADSK',
    'ANSS', 'CDNS', 'SNPS', 'QCOM', 'AVGO', 'MRVL', 'LRCX', 'KLAC', 'AMAT',
    'MU', 'NVDA', 'TSM', 'ASML', 'CSCO', 'IBM', 'HPQ', 'DELL', 'CRM'
]

FINANCIAL_SECTOR = [
    # Major Banks and Financial Services (Expanded) 
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI',
    'V', 'MA', 'PYPL', 'SQ', 'FIS', 'FISV', 'ADP', 'PAYX', 'TRV',
    'PGR', 'ALL', 'AIG', 'MET', 'PRU', 'AFL', 'CB', 'AON', 'MMC',
    'BRO', 'WRB', 'RE', 'FNF', 'NTRS', 'STT', 'BK', 'RF', 'ZION'
]

HEALTHCARE_SECTOR = [
    # Healthcare and Biotech (Expanded)
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
    'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ZTS', 'CI', 'CVS', 'HUM',
    'ANTM', 'CNC', 'MOH', 'WCG', 'ELV', 'SYK', 'BSX', 'MDT', 'EW', 'HOLX',
    'BDX', 'BAX', 'DXCM', 'ISRG', 'ILMN', 'IDXX', 'IQV', 'A', 'PRGO', 'TEVA',
    'CAH', 'MCK', 'ABC', 'CRL', 'LH', 'DGX', 'DVA', 'UHS', 'THC', 'HCA'
]

CONSUMER_DISCRETIONARY = [
    # Consumer and Retail (Expanded)
    'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'LULU', 'TJX', 'LOW', 'TGT', 'COST',
    'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'F', 'GM', 'TSLA',
    'AMZN', 'EBAY', 'ETSY', 'W', 'WAYFAIR', 'RH', 'WSM', 'BBY', 'ULTA', 'JWN',
    'M', 'KSS', 'DKS', 'FIVE', 'ORLY', 'AZO', 'AAP', 'GPC', 'LKQ', 'AN',
    'LAD', 'PAG', 'ABG', 'KMX', 'SIG', 'LEN', 'DHI', 'NVR', 'PHM', 'TOL'
]

CONSUMER_STAPLES = [
    # Consumer Staples and Beverages (Expanded)
    'KO', 'PEP', 'WMT', 'PG', 'UL', 'MDLZ', 'GIS', 'K', 'HSY', 'CL',
    'KMB', 'CLX', 'CHD', 'SJM', 'CAG', 'CPB', 'MKC', 'STZ', 'DEO', 'BF.B',
    'PM', 'MO', 'BTI', 'KR', 'SYY', 'COST', 'WBA', 'CVS', 'DG', 'DLTR',
    'EXC', 'SO', 'DUK', 'NEE', 'AEP', 'D', 'PCG', 'EIX', 'XEL', 'WEC'
]

INDUSTRIAL_ENERGY = [
    # Industrial and Energy Companies (Expanded)
    'XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX',
    'CAT', 'BA', 'GE', 'HON', 'LMT', 'RTX', 'MMM', 'UNP', 'CSX', 'NSC',
    'KMI', 'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'TRP', 'ENB', 'TC', 'PPL',
    'FTI', 'HAL', 'BKR', 'NOV', 'RRC', 'DVN', 'FANG', 'MRO', 'APA', 'OXY',
    'EMR', 'ITW', 'PH', 'ROK', 'DOV', 'XYL', 'FLS', 'PNR', 'IR', 'IEX'
]

GROWTH_STOCKS = [
    # High-growth companies across sectors (Expanded)
    'NVDA', 'AMD', 'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'OKTA', 'TWLO',
    'ZM', 'DOCU', 'SHOP', 'SQ', 'PYPL', 'ROKU', 'SPOT', 'ZS', 'ESTC',
    'PATH', 'BILL', 'S', 'AI', 'C3AI', 'RBLX', 'U', 'DASH', 'ABNB', 'LYFT',
    'PINS', 'SNAP', 'TWTR', 'FB', 'GOOGL', 'MSFT', 'AAPL', 'AMZN', 'NFLX',
    'CRM', 'ADBE', 'WDAY', 'VEEV', 'SPLK', 'PANW', 'FTNT', 'INTU', 'NOW'
]

VALUE_DIVIDEND_STOCKS = [
    # Value and dividend-focused stocks (Expanded)
    'BRK.B', 'XOM', 'CVX', 'JNJ', 'PG', 'KO', 'MCD', 'VZ', 'T', 'IBM',
    'INTC', 'PFE', 'MRK', 'JPM', 'BAC', 'WFC', 'C', 'KMI', 'ENB', 'TC',
    'O', 'REIT', 'SPG', 'PLD', 'AMT', 'CCI', 'SBAC', 'DLR', 'EXR', 'PSA',
    'AVB', 'EQR', 'UDR', 'CPT', 'MAA', 'ESS', 'AIV', 'BRX', 'ELS', 'LSI'
]

EMERGING_GROWTH = [
    # Emerging growth and mid-cap opportunities
    'SOFI', 'UPST', 'AFRM', 'LC', 'OPEN', 'COMP', 'ROOT', 'LMND', 'MTCH', 'BMBL',
    'ZG', 'Z', 'RDFN', 'EXPI', 'REAL', 'CPNG', 'SE', 'GRAB', 'BABA', 'JD',
    'PDD', 'TCEHY', 'NIO', 'XPEV', 'LI', 'BYDDY', 'TSM', 'ASML', 'SAP', 'SHOP'
]

SECTOR_ETFS = [
    # Major sector ETFs for market coverage
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'SCHD', 'VYM', 'VUG', 'VTV', 'VEA',
    'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE', 'XLC', 'XLY',
    'GDX', 'SLV', 'GLD', 'USO', 'UNG', 'VNQ', 'EFA', 'EEM', 'FXI', 'EWJ'
]

POPULAR_STOCKS = (
    MEGA_CAP_STOCKS + LARGE_CAP_TECH + FINANCIAL_SECTOR + 
    HEALTHCARE_SECTOR + CONSUMER_DISCRETIONARY + CONSUMER_STAPLES +
    INDUSTRIAL_ENERGY + GROWTH_STOCKS + VALUE_DIVIDEND_STOCKS +
    EMERGING_GROWTH + SECTOR_ETFS
)

# Remove duplicates while preserving order
POPULAR_STOCKS = list(dict.fromkeys(POPULAR_STOCKS))

# Cache globals
cache_client = None
cache_available = False

def _init_cache():
    global cache_client, cache_available
    if cache_available or cache_client or not VALKEY_ENDPOINT:
        return
    try:
        import redis  # works with Valkey
        cache_client = redis.Redis(
            host=VALKEY_ENDPOINT,
            port=6379,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=2,
            ssl=True,
            ssl_check_hostname=False,
            ssl_cert_reqs=None
        )
        cache_client.ping()
        cache_available = True
        logger.info("Valkey cache enabled")
    except ImportError:
        logger.info("redis/valkey client not packaged; cache disabled")
    except Exception as e:
        logger.warning(f"Cache init failed ({e}); disabled")

def _get_alpha_vantage_api_key():
    """Get Alpha Vantage API key from AWS Secrets Manager"""
    global ALPHA_VANTAGE_API_KEY
    
    if ALPHA_VANTAGE_API_KEY:
        return ALPHA_VANTAGE_API_KEY
    
    try:
        if ALPHA_VANTAGE_API_KEY_SECRET_ARN:
            response = secrets_manager.get_secret_value(SecretId=ALPHA_VANTAGE_API_KEY_SECRET_ARN)
            ALPHA_VANTAGE_API_KEY = response['SecretString']
            logger.info(f"Retrieved API key from Secrets Manager: {ALPHA_VANTAGE_API_KEY_SECRET_ARN}")
        else:
            # Fallback to environment variable (legacy)
            ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
            logger.warning("Using legacy API key from environment variable")
        
        return ALPHA_VANTAGE_API_KEY
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        return os.environ.get('ALPHA_VANTAGE_API_KEY', '')

def _internet_reachable():
    try:
        with socket.create_connection((CONNECT_TEST_HOST, CONNECT_TEST_PORT), CONNECT_TEST_TIMEOUT):
            logger.info(f"Connectivity OK to {CONNECT_TEST_HOST}:{CONNECT_TEST_PORT}")
            return True
    except Exception as e:
        logger.error(f"No outbound connectivity: {e}")
        return False

@trace_eventbridge_handler("stock_data_ingestion")
def lambda_handler(event, context):
    # Start main span with business context
    if BUSINESS_TRACING_AVAILABLE:
        span = financial_tracer.start_financial_span(
            "data_ingestion.lambda_handler",
            **{
                "lambda.function_name": context.function_name,
                "lambda.request_id": context.aws_request_id,
                "workflow.type": "data_ingestion"
            }
        )
    else:
        span = tracer.start_span("data_ingestion.lambda_handler")

    try:
        start = time.time()
        logger.info("Starting stock data ingestion process")
        _init_cache()

        # Add market session context
        if BUSINESS_TRACING_AVAILABLE:
            market_session = get_market_session()
            span.set_attributes({
                "finance.market_session": market_session.value,
                "finance.is_market_hours": market_session.value == "market_hours"
            })

        if not _internet_reachable():
            return {'statusCode': 502, 'body': json.dumps({'error':'no outbound connectivity','ts':datetime.utcnow().isoformat()})}

        # Enhanced symbol processing with comprehensive market coverage
        api_key = _get_alpha_vantage_api_key()
        if not api_key:
            logger.error("No Alpha Vantage API key available")
            return {'statusCode': 500, 'body': json.dumps({'error': 'API key not configured'})}
        
        # MAXIMIZED processing capacity based on API tier
        if USE_PREMIUM_API_KEY:
            total_symbols = min(MAX_SYMBOLS_PER_RUN, 50)  # INCREASED: Premium capacity (was 30)
            idx_symbols = MAJOR_INDEXES[:min(15, total_symbols // 4)]  # More comprehensive index coverage
        else:
            total_symbols = min(MAX_SYMBOLS_PER_RUN, 20)  # INCREASED: Free tier capacity (was 12)
            idx_symbols = MAJOR_INDEXES[:min(8, total_symbols // 3)]  # Enhanced index coverage
        
        remaining = max(0, total_symbols - len(idx_symbols))
        
        # Comprehensive stock rotation strategy for maximum coverage
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()  # 0=Monday, 6=Sunday
        
        # Multi-tier rotation: sector-based rotation with time-based variety
        if USE_PREMIUM_API_KEY and os.environ.get('COMPREHENSIVE_STOCK_COVERAGE', 'false').lower() == 'true':
            # Premium: Sector-based rotation for comprehensive coverage
            rotation_strategies = [
                MEGA_CAP_STOCKS,
                LARGE_CAP_TECH,
                FINANCIAL_SECTOR,
                HEALTHCARE_SECTOR,
                CONSUMER_DISCRETIONARY,
                INDUSTRIAL_ENERGY,
                GROWTH_STOCKS
            ]
            
            # Use day + hour for rotation to ensure comprehensive coverage
            strategy_idx = (current_day + current_hour // 4) % len(rotation_strategies)
            primary_sector = rotation_strategies[strategy_idx]
            
            # Mix in stocks from other sectors for diversity
            other_stocks = [stock for sector in rotation_strategies for stock in sector 
                          if sector != primary_sector][:remaining//2]
            
            # Combine primary sector with diverse selection
            candidate_stocks = primary_sector + other_stocks
            stk_symbols = list(dict.fromkeys(candidate_stocks))[:remaining]  # Remove duplicates
            
            logger.info(f"Premium: Processing sector {strategy_idx} with {len(primary_sector)} primary + {len(other_stocks)} diverse stocks")
        else:
            # Standard rotation through popular stocks
            rotation_seed = current_hour // 4  # Change every 4 hours
            stock_start_idx = (rotation_seed * remaining) % len(POPULAR_STOCKS)
            
            if remaining > 0:
                rotated_stocks = POPULAR_STOCKS[stock_start_idx:] + POPULAR_STOCKS[:stock_start_idx]
                stk_symbols = rotated_stocks[:remaining]
            else:
                stk_symbols = []

        logger.info(f"Processing {len(idx_symbols)} indexes and {len(stk_symbols)} stocks (Premium: {USE_PREMIUM_API_KEY}, Total universe: {len(POPULAR_STOCKS)} stocks)")

        # Add symbol processing context to span
        all_symbols = idx_symbols + stk_symbols
        if BUSINESS_TRACING_AVAILABLE and all_symbols:
            # Add symbol tier distribution to span
            tier_counts = {}
            for symbol in all_symbols:
                tier = classify_symbol(symbol).value
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            span.set_attributes({
                "data_ingestion.total_symbols": len(all_symbols),
                "data_ingestion.index_symbols": len(idx_symbols),
                "data_ingestion.stock_symbols": len(stk_symbols),
                **{f"data_ingestion.{tier}_symbols": count for tier, count in tier_counts.items()}
            })

        index_data = _process_symbol_list(idx_symbols, category='index')
        stock_data = _process_symbol_list(stk_symbols, category='stock')

        if index_data or stock_data:
            # Create correlation context for ML inference
            if BUSINESS_TRACING_AVAILABLE:
                correlation_context = propagate_correlation_context(
                    parent_operation="data_ingestion",
                    symbols=[item['symbol'] for item in (index_data + stock_data)]
                )
                trigger_ml_inference(index_data, stock_data, correlation_context)
            else:
                trigger_ml_inference(index_data, stock_data)

        duration = round(time.time() - start, 2)
        
        # Publish custom metrics for enhanced observability
        publish_ingestion_metrics(len(index_data), len(stock_data), len(idx_symbols) + len(stk_symbols), duration)

        # Add final success attributes to span
        span.set_attributes({
            "data_ingestion.indexes_processed": len(index_data),
            "data_ingestion.stocks_processed": len(stock_data),
            "data_ingestion.symbols_attempted": len(idx_symbols) + len(stk_symbols),
            "data_ingestion.duration_sec": duration,
            "data_ingestion.ml_inference_triggered": bool(index_data or stock_data),
            "data_ingestion.success": True
        })

        span.set_status(Status(StatusCode.OK))

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Enhanced stock data ingestion completed',
                'indexes_processed': len(index_data),
                'stocks_processed': len(stock_data),
                'symbols_attempted': len(idx_symbols) + len(stk_symbols),
                'duration_sec': duration,
                'ml_inference_triggered': bool(index_data or stock_data),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
    except Exception as e:
        # Add error attributes to span
        span.set_attributes({
            "data_ingestion.success": False,
            "error.type": type(e).__name__,
            "error.message": str(e)
        })
        span.set_status(Status(StatusCode.ERROR, str(e)))
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'timestamp': datetime.utcnow().isoformat()})
        }
    finally:
        span.end()

def _process_symbol_list(symbols, category):
    results = []
    for symbol in symbols:
        try:
            raw = fetch_stock_data(symbol)
            if not raw:
                continue
            processed = process_stock_data(symbol, raw)
            if not processed:
                continue
            store_in_s3(symbol, processed)
            cache_latest_price(symbol, processed)
            results.append(processed)
            logger.info(f"Processed {category} {symbol}")
        except Exception as e:
            logger.error(f"{category} {symbol} failed: {e}")
    return results

def http_get_json(url, params, timeout=PER_CALL_TIMEOUT):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        logger.error(f"HTTP error {e} for {params.get('symbol')}")
        return {}

def fetch_stock_data(symbol):
    try:
        if cache_available and cache_client:
            cached = cache_client.get(f"stock_data:{symbol}")
            if cached:
                logger.info(f"Cache hit {symbol}")
                return json.loads(cached)

        api_key = _get_alpha_vantage_api_key()
        if not api_key:
            logger.error(f"No API key available for {symbol}")
            return None

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'compact',
            'entitlement': 'delayed',
            'apikey': api_key
        }
        data = http_get_json(url, params)
        if not data:
            return None
        if 'Error Message' in data:
            logger.warning(f"API error {symbol}: {data['Error Message']}")
            return None
        if 'Note' in data:
            logger.warning(f"Rate limit note for {symbol}: {data['Note']}")
            return None

        if cache_available and cache_client:
            cache_client.setex(f"stock_data:{symbol}", 300, json.dumps(data))
        return data
    except Exception as e:
        logger.error(f"Fetch error {symbol}: {e}")
        return None

def process_stock_data(symbol, raw_data):
    try:
        if 'Time Series (Daily)' not in raw_data:
            return None
        ts = raw_data['Time Series (Daily)']
        dates = sorted(ts.keys(), reverse=True)
        if not dates:
            return None
        latest_date = dates[0]
        latest = ts[latest_date]
        prices = [float(ts[d]['4. close']) for d in dates[:20]]
        moving_avg_5  = sum(prices[:5]) / 5 if len(prices) >= 5 else prices[0]
        moving_avg_20 = sum(prices) / len(prices)
        if len(prices) > 1:
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            mean_r = sum(returns)/len(returns)
            volatility = (sum((r - mean_r)**2 for r in returns)/len(returns))**0.5
        else:
            volatility = 0.0
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'date': latest_date,
            'open': Decimal(str(latest['1. open'])),
            'high': Decimal(str(latest['2. high'])),
            'low': Decimal(str(latest['3. low'])),
            'close': Decimal(str(latest['4. close'])),
            'volume': int(latest['5. volume']),
            'moving_avg_5': Decimal(str(moving_avg_5)),
            'moving_avg_20': Decimal(str(moving_avg_20)),
            'volatility': Decimal(str(volatility)),
            'metadata': {
                'source': 'alpha_vantage',
                'processed_at': datetime.utcnow().isoformat(),
                'data_quality': 'high' if len(prices) >= 20 else 'medium'
            }
        }
    except Exception as e:
        logger.error(f"Process error {symbol}: {e}")
        return None

def store_in_s3(symbol, data):
    try:
        if not data:
            return
        json_data = json.loads(json.dumps(data, default=decimal_default))
        key = f"stock-data/{symbol}/{datetime.utcnow().strftime('%Y/%m/%d')}/{symbol}_{datetime.utcnow().strftime('%H%M%S')}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(json_data, indent=2),
            ContentType='application/json',
            Metadata={'symbol': symbol, 'timestamp': datetime.utcnow().isoformat(), 'data_type': 'daily_stock_data'}
        )
        logger.info(f"S3 stored {symbol} -> {key}")
    except Exception as e:
        logger.error(f"S3 store error {symbol}: {e}")

def cache_latest_price(symbol, data):
    try:
        if not data:
            return
        table = dynamodb.Table(DYNAMODB_TABLE)
        item = {
            'recommendation_id': f"latest_price:{symbol}",
            'symbol': symbol,
            'price': data['close'],
            'change': data['close'] - data['open'],
            'change_percent': ((data['close'] - data['open']) / data['open']) * 100,
            'volume': data['volume'],
            'timestamp': data['timestamp'],
            'ttl': int(datetime.utcnow().timestamp() + 3600)
        }
        table.put_item(Item=item)
    except Exception as e:
        logger.error(f"DDB cache error {symbol}: {e}")

def trigger_ml_inference(index_data, stock_data, correlation_context=None):
    try:
        lambda_client = boto3.client('lambda')
        payload = {
            'trigger': 'stock_data_ingestion',
            'index_data': json.loads(json.dumps(index_data, default=decimal_default)),
            'stock_data': json.loads(json.dumps(stock_data, default=decimal_default)),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add correlation context if available
        if correlation_context:
            payload.update(correlation_context)
        
        # Trigger existing ML inference (recommendations)
        lambda_client.invoke(
            FunctionName=ML_INFERENCE_FUNCTION_NAME,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        logger.info(f"Triggered ML inference {ML_INFERENCE_FUNCTION_NAME}")
        
        # Trigger new dual prediction system
        trigger_dual_predictions(stock_data)
        
    except Exception as e:
        logger.error(f"Inference trigger error: {e}")

def trigger_dual_predictions(stock_data):
    """Trigger the new dual prediction system (price and time predictions)"""
    try:
        lambda_client = boto3.client('lambda')
        
        # stock_data is a list of processed stock data, convert to symbols and data dict
        if not stock_data:
            logger.info("No stock symbols to process for dual predictions")
            return
        
        # Extract symbols and create data dictionary
        symbols = []
        stock_data_dict = {}
        
        for item in stock_data:
            if 'symbol' in item:
                symbol = item['symbol']
                symbols.append(symbol)
                stock_data_dict[symbol] = item
        
        if not symbols:
            logger.info("No valid stock symbols found for dual predictions")
            return
        
        logger.info(f"Triggering dual predictions for {len(symbols)} symbols: {symbols}")
        
        # Trigger price predictions with actual stock data
        price_payload = {
            'trigger_type': 'data_ingestion',
            'symbols': symbols,
            'stock_data': stock_data_dict,  # Pass the actual price data as dict
            'market_session': 'data_update',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            lambda_client.invoke(
                FunctionName='price-prediction-model',
                InvocationType='Event',
                Payload=json.dumps(price_payload, default=decimal_default)
            )
            logger.info(f"Triggered price predictions for {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to trigger price predictions: {e}")
        
        # Trigger time predictions for a subset of symbols (to avoid timeout)
        time_symbols = symbols[:5]  # Limit to first 5 symbols
        time_payload = {
            'trigger_type': 'data_ingestion',
            'symbols': time_symbols,
            'market_session': 'data_update',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            lambda_client.invoke(
                FunctionName='time-to-hit-predictor',
                InvocationType='Event',
                Payload=json.dumps(time_payload, default=decimal_default)
            )
            logger.info(f"Triggered time predictions for {len(time_symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to trigger time predictions: {e}")
            
    except Exception as e:
        logger.error(f"Error triggering dual predictions: {e}")

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def publish_ingestion_metrics(indexes_processed, stocks_processed, symbols_attempted, duration):
    """
    Publish custom CloudWatch metrics for data ingestion observability
    """
    try:
        metrics = [
            {
                'MetricName': 'IndexesProcessed',
                'Value': indexes_processed,
                'Unit': 'Count'
            },
            {
                'MetricName': 'StocksProcessed',
                'Value': stocks_processed,
                'Unit': 'Count'
            },
            {
                'MetricName': 'SymbolsAttempted',
                'Value': symbols_attempted,
                'Unit': 'Count'
            },
            {
                'MetricName': 'IngestionDuration',
                'Value': duration,
                'Unit': 'Seconds'
            },
            {
                'MetricName': 'IngestionSuccess',
                'Value': 1,
                'Unit': 'Count'
            },
            {
                'MetricName': 'DataQualityScore',
                'Value': (indexes_processed + stocks_processed) / max(symbols_attempted, 1) * 100,
                'Unit': 'Percent'
            }
        ]
        
        cloudwatch.put_metric_data(
            Namespace='StockAnalytics/DataIngestion',
            MetricData=metrics
        )
        
        logger.info(f"Published ingestion metrics - Processed: {indexes_processed + stocks_processed}/{symbols_attempted}, Duration: {duration}s")
        
    except Exception as e:
        logger.error(f"Failed to publish ingestion metrics: {e}")

def publish_error_metric(error_type):
    """
    Publish error metrics for monitoring failures
    """
    try:
        cloudwatch.put_metric_data(
            Namespace='StockAnalytics/DataIngestion',
            MetricData=[
                {
                    'MetricName': 'IngestionError',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'ErrorType',
                            'Value': error_type
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        logger.error(f"Failed to publish error metric: {e}")