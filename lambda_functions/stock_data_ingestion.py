"""
Stock Data Ingestion Lambda Function
Fetches stock data from Alpha Vantage API and processes it
"""

import json
import boto3
import requests
import logging
from datetime import datetime
from decimal import Decimal
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
redis_client = None

# Configuration
ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
S3_BUCKET = os.environ['S3_BUCKET']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
REDIS_ENDPOINT = os.environ.get('REDIS_ENDPOINT')

# Stock symbols to monitor (major indexes and components)
MAJOR_INDEXES = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
POPULAR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'IBM']

def lambda_handler(event, context):
    """
    Main Lambda handler function
    """
    try:
        logger.info("Starting stock data ingestion process")
        
        # Initialize Redis connection if available
        if REDIS_ENDPOINT:
            import redis
            global redis_client
            redis_client = redis.Redis(host=REDIS_ENDPOINT, port=6379, decode_responses=True)
        
        # Process major indexes first
        index_data = []
        for symbol in MAJOR_INDEXES:
            try:
                data = fetch_stock_data(symbol)
                if data:
                    processed_data = process_stock_data(symbol, data)
                    store_in_s3(symbol, processed_data)
                    cache_latest_price(symbol, processed_data)
                    index_data.append(processed_data)
                    logger.info(f"Successfully processed {symbol}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Process individual stocks
        stock_data = []
        for symbol in POPULAR_STOCKS:
            try:
                data = fetch_stock_data(symbol)
                if data:
                    processed_data = process_stock_data(symbol, data)
                    store_in_s3(symbol, processed_data)
                    cache_latest_price(symbol, processed_data)
                    stock_data.append(processed_data)
                    logger.info(f"Successfully processed {symbol}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Trigger ML inference pipeline
        trigger_ml_inference(index_data, stock_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Stock data ingestion completed successfully',
                'indexes_processed': len(index_data),
                'stocks_processed': len(stock_data),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def fetch_stock_data(symbol):
    """
    Fetch stock data from Alpha Vantage API
    """
    try:
        # Check cache first
        if redis_client:
            cached_data = redis_client.get(f"stock_data:{symbol}")
            if cached_data:
                logger.info(f"Using cached data for {symbol}")
                return json.loads(cached_data)
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'compact',
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            logger.error(f"API Error for {symbol}: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            logger.warning(f"API Rate Limit for {symbol}: {data['Note']}")
            return None
        
        # Cache the data for 5 minutes
        if redis_client:
            redis_client.setex(f"stock_data:{symbol}", 300, json.dumps(data))
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def process_stock_data(symbol, raw_data):
    """
    Process raw stock data into standardized format
    """
    try:
        if 'Time Series (Daily)' not in raw_data:
            logger.error(f"No time series data for {symbol}")
            return None
        
        time_series = raw_data['Time Series (Daily)']
        dates = sorted(time_series.keys(), reverse=True)
        
        if not dates:
            logger.error(f"No data points for {symbol}")
            return None
        
        # Get latest data point
        latest_date = dates[0]
        latest_data = time_series[latest_date]
        
        # Calculate technical indicators
        prices = [float(time_series[date]['4. close']) for date in dates[:20]]  # Last 20 days
        
        moving_avg_5 = sum(prices[:5]) / 5 if len(prices) >= 5 else prices[0]
        moving_avg_20 = sum(prices) / len(prices) if len(prices) >= 20 else sum(prices) / len(prices)
        
        # Calculate volatility (standard deviation of returns)
        if len(prices) > 1:
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            volatility = (sum([(r - sum(returns)/len(returns))**2 for r in returns]) / len(returns))**0.5
        else:
            volatility = 0
        
        processed_data = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'date': latest_date,
            'open': Decimal(str(latest_data['1. open'])),
            'high': Decimal(str(latest_data['2. high'])),
            'low': Decimal(str(latest_data['3. low'])),
            'close': Decimal(str(latest_data['4. close'])),
            'volume': int(latest_data['5. volume']),
            'moving_avg_5': Decimal(str(moving_avg_5)),
            'moving_avg_20': Decimal(str(moving_avg_20)),
            'volatility': Decimal(str(volatility)),
            'metadata': {
                'source': 'alpha_vantage',
                'processed_at': datetime.utcnow().isoformat(),
                'data_quality': 'high' if len(prices) >= 20 else 'medium'
            }
        }
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {str(e)}")
        return None

def store_in_s3(symbol, data):
    """
    Store processed data in S3
    """
    try:
        if not data:
            return
        
        # Convert Decimal objects to float for JSON serialization
        json_data = json.loads(json.dumps(data, default=decimal_default))
        
        key = f"stock-data/{symbol}/{datetime.utcnow().strftime('%Y/%m/%d')}/{symbol}_{datetime.utcnow().strftime('%H%M%S')}.json"
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(json_data, indent=2),
            ContentType='application/json',
            Metadata={
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'data_type': 'daily_stock_data'
            }
        )
        
        logger.info(f"Stored data for {symbol} in S3: {key}")
        
    except Exception as e:
        logger.error(f"Error storing data in S3 for {symbol}: {str(e)}")

def cache_latest_price(symbol, data):
    """
    Cache latest price data in DynamoDB
    """
    try:
        if not data:
            return
        
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Convert Decimal objects for DynamoDB
        item = {
            'cache_key': f"latest_price:{symbol}",
            'symbol': symbol,
            'price': data['close'],
            'change': data['close'] - data['open'],
            'change_percent': ((data['close'] - data['open']) / data['open']) * 100,
            'volume': data['volume'],
            'timestamp': data['timestamp'],
            'ttl': int((datetime.utcnow().timestamp()) + 3600)  # 1 hour TTL
        }
        
        table.put_item(Item=item)
        logger.info(f"Cached latest price for {symbol}")
        
    except Exception as e:
        logger.error(f"Error caching price for {symbol}: {str(e)}")

def trigger_ml_inference(index_data, stock_data):
    """
    Trigger ML inference Lambda function
    """
    try:
        lambda_client = boto3.client('lambda')
        
        payload = {
            'trigger': 'stock_data_ingestion',
            'index_data': json.loads(json.dumps(index_data, default=decimal_default)),
            'stock_data': json.loads(json.dumps(stock_data, default=decimal_default)),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        lambda_client.invoke(
            FunctionName='ml-model-inference',
            InvocationType='Event',  # Async invocation
            Payload=json.dumps(payload)
        )
        
        logger.info("Triggered ML inference pipeline")
        
    except Exception as e:
        logger.error(f"Error triggering ML inference: {str(e)}")

def decimal_default(obj):
    """
    JSON serializer for Decimal objects
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")