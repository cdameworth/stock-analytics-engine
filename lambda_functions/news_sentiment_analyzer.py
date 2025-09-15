#!/usr/bin/env python3
"""
News Sentiment Analysis Service for Stock Analytics Engine
Integrates multiple news APIs and sentiment analysis for enhanced features.
"""

import json
import os
import boto3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Structure for news article data"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None

@dataclass
class SentimentMetrics:
    """Aggregated sentiment metrics for a symbol"""
    overall_sentiment: float  # -1 to 1 scale
    sentiment_momentum: float  # Change in sentiment over time
    news_volume: int  # Number of articles
    average_relevance: float  # How relevant news is to the stock
    sentiment_volatility: float  # Variance in sentiment scores
    bullish_ratio: float  # Percentage of positive sentiment
    bearish_ratio: float  # Percentage of negative sentiment
    neutral_ratio: float  # Percentage of neutral sentiment

class NewsSentimentAnalyzer:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.ssm = boto3.client('ssm', region_name='us-east-1')

        # Cache table for sentiment data
        self.sentiment_cache_table = self.dynamodb.Table('news-sentiment-cache')

        # News API configurations
        self.news_apis = self._initialize_news_apis()

        # Sentiment analysis configuration
        self.sentiment_keywords = {
            'bullish': ['growth', 'profit', 'earnings beat', 'upgrade', 'buy rating', 'positive outlook',
                       'strong results', 'expansion', 'acquisition', 'partnership', 'innovation'],
            'bearish': ['loss', 'decline', 'downgrade', 'sell rating', 'negative outlook', 'miss',
                       'lawsuit', 'investigation', 'bankruptcy', 'layoffs', 'recession', 'crisis'],
            'neutral': ['analyst', 'report', 'conference', 'meeting', 'announcement', 'statement']
        }

    def _initialize_news_apis(self) -> Dict:
        """Initialize news API configurations"""
        try:
            # Get API keys from Parameter Store
            newsapi_key = self._get_parameter('/stock-analytics/newsapi-key', default='demo_key')
            alpha_vantage_key = self._get_parameter('/stock-analytics/alpha-vantage-api-key')

            return {
                'newsapi': {
                    'url': 'https://newsapi.org/v2/everything',
                    'key': newsapi_key,
                    'rate_limit': 1000,  # requests per day
                    'last_request': 0
                },
                'alpha_vantage_news': {
                    'url': 'https://www.alphavantage.co/query',
                    'key': alpha_vantage_key,
                    'rate_limit': 75,  # requests per minute
                    'last_request': 0
                },
                'finnhub': {
                    'url': 'https://finnhub.io/api/v1/news',
                    'key': self._get_parameter('/stock-analytics/finnhub-key', default='demo_key'),
                    'rate_limit': 60,  # requests per minute
                    'last_request': 0
                }
            }
        except Exception as e:
            logger.error(f"Failed to initialize news APIs: {e}")
            return {}

    def _get_parameter(self, parameter_name: str, default: str = None) -> str:
        """Get parameter from AWS Parameter Store"""
        try:
            response = self.ssm.get_parameter(
                Name=parameter_name,
                WithDecryption=True
            )
            return response['Parameter']['Value']
        except Exception as e:
            logger.warning(f"Failed to get parameter {parameter_name}: {e}")
            return default

    def _rate_limit_check(self, api_name: str) -> bool:
        """Check if we can make a request to the API without hitting rate limits"""
        api_config = self.news_apis.get(api_name, {})
        if not api_config:
            return False

        current_time = time.time()
        last_request = api_config.get('last_request', 0)
        rate_limit = api_config.get('rate_limit', 60)

        # Calculate minimum interval between requests
        if api_name == 'newsapi':
            min_interval = 86400 / rate_limit  # Daily limit
        else:
            min_interval = 60 / rate_limit  # Per minute limit

        if current_time - last_request >= min_interval:
            api_config['last_request'] = current_time
            return True
        return False

    def get_news_sentiment(self, symbol: str, lookback_hours: int = 24) -> SentimentMetrics:
        """
        Get comprehensive sentiment analysis for a stock symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            lookback_hours: Hours to look back for news (default 24)

        Returns:
            SentimentMetrics object with aggregated sentiment data
        """
        try:
            # Check cache first
            cached_sentiment = self._get_cached_sentiment(symbol, lookback_hours)
            if cached_sentiment:
                logger.info(f"Using cached sentiment for {symbol}")
                return cached_sentiment

            # Fetch news from multiple sources
            articles = self._fetch_news_articles(symbol, lookback_hours)

            if not articles:
                logger.warning(f"No news articles found for {symbol}")
                return self._create_default_sentiment_metrics()

            # Analyze sentiment for each article
            analyzed_articles = []
            for article in articles:
                sentiment_score = self._analyze_article_sentiment(article)
                relevance_score = self._calculate_relevance_score(article, symbol)

                article.sentiment_score = sentiment_score
                article.relevance_score = relevance_score
                analyzed_articles.append(article)

            # Aggregate sentiment metrics
            sentiment_metrics = self._aggregate_sentiment_metrics(analyzed_articles)

            # Cache the results
            self._cache_sentiment_data(symbol, sentiment_metrics, lookback_hours)

            logger.info(f"Sentiment analysis complete for {symbol}: {sentiment_metrics.overall_sentiment:.3f}")
            return sentiment_metrics

        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return self._create_default_sentiment_metrics()

    def _get_cached_sentiment(self, symbol: str, lookback_hours: int) -> Optional[SentimentMetrics]:
        """Check if we have recent cached sentiment data"""
        try:
            cache_key = f"{symbol}_{lookback_hours}h"
            response = self.sentiment_cache_table.get_item(
                Key={'cache_key': cache_key}
            )

            if 'Item' in response:
                item = response['Item']
                cache_time = datetime.fromisoformat(item['timestamp'])

                # Use cache if less than 1 hour old
                if datetime.utcnow() - cache_time < timedelta(hours=1):
                    return SentimentMetrics(
                        overall_sentiment=float(item['overall_sentiment']),
                        sentiment_momentum=float(item['sentiment_momentum']),
                        news_volume=int(item['news_volume']),
                        average_relevance=float(item['average_relevance']),
                        sentiment_volatility=float(item['sentiment_volatility']),
                        bullish_ratio=float(item['bullish_ratio']),
                        bearish_ratio=float(item['bearish_ratio']),
                        neutral_ratio=float(item['neutral_ratio'])
                    )
        except Exception as e:
            logger.warning(f"Failed to get cached sentiment for {symbol}: {e}")

        return None

    def _fetch_news_articles(self, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch news articles from multiple sources"""
        articles = []

        # Try multiple news sources
        for api_name in ['alpha_vantage_news', 'newsapi', 'finnhub']:
            try:
                if self._rate_limit_check(api_name):
                    source_articles = self._fetch_from_source(api_name, symbol, lookback_hours)
                    articles.extend(source_articles)

                    # Limit total articles to avoid processing overload
                    if len(articles) >= 50:
                        break
            except Exception as e:
                logger.warning(f"Failed to fetch from {api_name}: {e}")
                continue

        return articles[:50]  # Limit to 50 most recent articles

    def _fetch_from_source(self, api_name: str, symbol: str, lookback_hours: int) -> List[NewsArticle]:
        """Fetch articles from a specific news source"""
        api_config = self.news_apis.get(api_name, {})
        if not api_config:
            return []

        articles = []

        try:
            if api_name == 'alpha_vantage_news':
                articles = self._fetch_alpha_vantage_news(symbol, api_config)
            elif api_name == 'newsapi':
                articles = self._fetch_newsapi_articles(symbol, lookback_hours, api_config)
            elif api_name == 'finnhub':
                articles = self._fetch_finnhub_news(symbol, api_config)

        except Exception as e:
            logger.error(f"Error fetching from {api_name}: {e}")

        return articles

    def _fetch_alpha_vantage_news(self, symbol: str, api_config: Dict) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage News & Sentiment API"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_config['key'],
            'limit': 20
        }

        response = requests.get(api_config['url'], params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = []

        if 'feed' in data:
            for item in data['feed']:
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('summary', ''),
                    source=item.get('source', 'Alpha Vantage'),
                    published_at=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                    url=item.get('url', '')
                )
                articles.append(article)

        return articles

    def _fetch_newsapi_articles(self, symbol: str, lookback_hours: int, api_config: Dict) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        from_date = (datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat()

        params = {
            'q': f'"{symbol}" OR "company name"',  # Would need company name lookup
            'from': from_date,
            'sortBy': 'publishedAt',
            'apiKey': api_config['key'],
            'pageSize': 20,
            'language': 'en'
        }

        response = requests.get(api_config['url'], params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = []

        if 'articles' in data:
            for item in data['articles']:
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('description', '') + ' ' + item.get('content', ''),
                    source=item.get('source', {}).get('name', 'NewsAPI'),
                    published_at=datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                    url=item.get('url', '')
                )
                articles.append(article)

        return articles

    def _fetch_finnhub_news(self, symbol: str, api_config: Dict) -> List[NewsArticle]:
        """Fetch news from Finnhub"""
        # Placeholder implementation - would need actual Finnhub integration
        return []

    def _analyze_article_sentiment(self, article: NewsArticle) -> float:
        """
        Analyze sentiment of a news article
        Returns score from -1 (very negative) to 1 (very positive)
        """
        text = (article.title + ' ' + article.content).lower()

        # Simple keyword-based sentiment analysis
        bullish_score = 0
        bearish_score = 0

        for keyword in self.sentiment_keywords['bullish']:
            bullish_score += text.count(keyword)

        for keyword in self.sentiment_keywords['bearish']:
            bearish_score += text.count(keyword)

        # Normalize to -1 to 1 scale
        total_keywords = bullish_score + bearish_score
        if total_keywords == 0:
            return 0.0

        sentiment = (bullish_score - bearish_score) / total_keywords
        return max(-1.0, min(1.0, sentiment))

    def _calculate_relevance_score(self, article: NewsArticle, symbol: str) -> float:
        """Calculate how relevant an article is to the specific stock"""
        text = (article.title + ' ' + article.content).lower()
        symbol_mentions = text.count(symbol.lower())

        # Basic relevance scoring
        relevance = min(1.0, symbol_mentions * 0.3)

        # Boost relevance for financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'shares', 'dividend']
        for keyword in financial_keywords:
            if keyword in text:
                relevance += 0.1

        return min(1.0, relevance)

    def _aggregate_sentiment_metrics(self, articles: List[NewsArticle]) -> SentimentMetrics:
        """Aggregate individual article sentiments into overall metrics"""
        if not articles:
            return self._create_default_sentiment_metrics()

        # Weight articles by relevance
        weighted_sentiments = []
        total_weight = 0

        for article in articles:
            if article.sentiment_score is not None and article.relevance_score is not None:
                weight = article.relevance_score
                weighted_sentiments.append(article.sentiment_score * weight)
                total_weight += weight

        if total_weight == 0:
            return self._create_default_sentiment_metrics()

        # Calculate overall sentiment
        overall_sentiment = sum(weighted_sentiments) / total_weight

        # Calculate sentiment distribution
        positive_count = sum(1 for a in articles if a.sentiment_score and a.sentiment_score > 0.1)
        negative_count = sum(1 for a in articles if a.sentiment_score and a.sentiment_score < -0.1)
        neutral_count = len(articles) - positive_count - negative_count

        total_articles = len(articles)
        bullish_ratio = positive_count / total_articles
        bearish_ratio = negative_count / total_articles
        neutral_ratio = neutral_count / total_articles

        # Calculate sentiment volatility
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        sentiment_volatility = 0.0
        if len(sentiments) > 1:
            avg_sentiment = sum(sentiments) / len(sentiments)
            variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
            sentiment_volatility = variance ** 0.5

        # Calculate sentiment momentum (placeholder - would need historical data)
        sentiment_momentum = 0.0

        # Calculate average relevance
        relevances = [a.relevance_score for a in articles if a.relevance_score is not None]
        average_relevance = sum(relevances) / len(relevances) if relevances else 0.0

        return SentimentMetrics(
            overall_sentiment=overall_sentiment,
            sentiment_momentum=sentiment_momentum,
            news_volume=len(articles),
            average_relevance=average_relevance,
            sentiment_volatility=sentiment_volatility,
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio,
            neutral_ratio=neutral_ratio
        )

    def _cache_sentiment_data(self, symbol: str, metrics: SentimentMetrics, lookback_hours: int):
        """Cache sentiment data in DynamoDB"""
        try:
            cache_key = f"{symbol}_{lookback_hours}h"

            self.sentiment_cache_table.put_item(
                Item={
                    'cache_key': cache_key,
                    'symbol': symbol,
                    'timestamp': datetime.utcnow().isoformat(),
                    'overall_sentiment': float(metrics.overall_sentiment),
                    'sentiment_momentum': float(metrics.sentiment_momentum),
                    'news_volume': int(metrics.news_volume),
                    'average_relevance': float(metrics.average_relevance),
                    'sentiment_volatility': float(metrics.sentiment_volatility),
                    'bullish_ratio': float(metrics.bullish_ratio),
                    'bearish_ratio': float(metrics.bearish_ratio),
                    'neutral_ratio': float(metrics.neutral_ratio),
                    'ttl': int((datetime.utcnow() + timedelta(hours=6)).timestamp())  # Cache for 6 hours
                }
            )
        except Exception as e:
            logger.error(f"Failed to cache sentiment data for {symbol}: {e}")

    def _create_default_sentiment_metrics(self) -> SentimentMetrics:
        """Create default sentiment metrics when no data is available"""
        return SentimentMetrics(
            overall_sentiment=0.0,
            sentiment_momentum=0.0,
            news_volume=0,
            average_relevance=0.0,
            sentiment_volatility=0.0,
            bullish_ratio=0.33,
            bearish_ratio=0.33,
            neutral_ratio=0.34
        )

def lambda_handler(event, context):
    """
    Lambda handler for news sentiment analysis

    Expected event structure:
    {
        "symbol": "AAPL",
        "lookback_hours": 24
    }
    """
    try:
        symbol = event.get('symbol', 'AAPL')
        lookback_hours = event.get('lookback_hours', 24)

        analyzer = NewsSentimentAnalyzer()
        sentiment_metrics = analyzer.get_news_sentiment(symbol, lookback_hours)

        return {
            'statusCode': 200,
            'body': {
                'symbol': symbol,
                'sentiment_metrics': {
                    'overall_sentiment': sentiment_metrics.overall_sentiment,
                    'sentiment_momentum': sentiment_metrics.sentiment_momentum,
                    'news_volume': sentiment_metrics.news_volume,
                    'average_relevance': sentiment_metrics.average_relevance,
                    'sentiment_volatility': sentiment_metrics.sentiment_volatility,
                    'bullish_ratio': sentiment_metrics.bullish_ratio,
                    'bearish_ratio': sentiment_metrics.bearish_ratio,
                    'neutral_ratio': sentiment_metrics.neutral_ratio
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in sentiment analysis lambda: {e}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e)
            }
        }

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    metrics = analyzer.get_news_sentiment('AAPL', 24)
    print(f"AAPL Sentiment: {metrics.overall_sentiment:.3f}")
    print(f"News Volume: {metrics.news_volume}")
    print(f"Bullish/Bearish: {metrics.bullish_ratio:.2f}/{metrics.bearish_ratio:.2f}")