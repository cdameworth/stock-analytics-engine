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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

CONNECT_TEST_HOST = os.environ.get('CONNECT_TEST_HOST', 'www.alphavantage.co')
CONNECT_TEST_PORT = 443
CONNECT_TEST_TIMEOUT = int(os.environ.get('CONNECT_TEST_TIMEOUT', '2'))
ABORT_AFTER_SEC = int(os.environ.get('ABORT_AFTER_SEC', '40'))
ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
S3_BUCKET             = os.environ['S3_BUCKET']
DYNAMODB_TABLE        = os.environ['DYNAMODB_TABLE']

# Prefer VALKEY_ENDPOINT; fallback to legacy REDIS_ENDPOINT
VALKEY_ENDPOINT = os.environ.get('VALKEY_ENDPOINT') or os.environ.get('REDIS_ENDPOINT')

# Tuning (override via env)
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '5'))   # keep under rate limit
PER_CALL_TIMEOUT    = int(os.environ.get('PER_CALL_TIMEOUT', '8'))      # seconds per API call
ML_INFERENCE_FUNCTION_NAME = os.environ.get('ML_INFERENCE_FUNCTION_NAME', 'ml-model-inference-lowcost')

MAJOR_INDEXES   = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
POPULAR_STOCKS  = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'IBM']

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
            socket_timeout=1
        )
        cache_client.ping()
        cache_available = True
        logger.info("Valkey cache enabled")
    except ImportError:
        logger.info("redis/valkey client not packaged; cache disabled")
    except Exception as e:
        logger.warning(f"Cache init failed ({e}); disabled")

def _internet_reachable():
    try:
        with socket.create_connection((CONNECT_TEST_HOST, CONNECT_TEST_PORT), CONNECT_TEST_TIMEOUT):
            logger.info(f"Connectivity OK to {CONNECT_TEST_HOST}:{CONNECT_TEST_PORT}")
            return True
    except Exception as e:
        logger.error(f"No outbound connectivity: {e}")
        return False

def lambda_handler(event, context):
    try:
        start = time.time()
        logger.info("Starting stock data ingestion process")
        _init_cache()

        if not _internet_reachable():
            return {'statusCode': 502, 'body': json.dumps({'error':'no outbound connectivity','ts':datetime.utcnow().isoformat()})}

        # Limit symbols this run (indexes first)
        idx_symbols = MAJOR_INDEXES[:MAX_SYMBOLS_PER_RUN]
        remaining = max(0, MAX_SYMBOLS_PER_RUN - len(idx_symbols))
        stk_symbols = POPULAR_STOCKS[:remaining] if remaining else []

        index_data = _process_symbol_list(idx_symbols, category='index')
        stock_data = _process_symbol_list(stk_symbols, category='stock')

        if index_data or stock_data:
            trigger_ml_inference(index_data, stock_data)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Stock data ingestion completed',
                'indexes_processed': len(index_data),
                'stocks_processed': len(stock_data),
                'symbols_attempted': len(idx_symbols) + len(stk_symbols),
                'duration_sec': round(time.time() - start, 2),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'timestamp': datetime.utcnow().isoformat()})
        }

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

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'compact',
            'apikey': ALPHA_VANTAGE_API_KEY
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

def trigger_ml_inference(index_data, stock_data):
    try:
        lambda_client = boto3.client('lambda')
        payload = {
            'trigger': 'stock_data_ingestion',
            'index_data': json.loads(json.dumps(index_data, default=decimal_default)),
            'stock_data': json.loads(json.dumps(stock_data, default=decimal_default)),
            'timestamp': datetime.utcnow().isoformat()
        }
        lambda_client.invoke(
            FunctionName=ML_INFERENCE_FUNCTION_NAME,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        logger.info(f"Triggered ML inference {ML_INFERENCE_FUNCTION_NAME}")
    except Exception as e:
        logger.error(f"Inference trigger error: {e}")

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError