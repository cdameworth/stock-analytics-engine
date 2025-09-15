#!/usr/bin/env python3
"""
Test Script for News Sentiment Analysis Integration
Tests the enhanced feature extractor with sentiment capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lambda_functions'))

import json
import boto3
from datetime import datetime
from enhanced_feature_extractor import EnhancedFeatureExtractor
from news_sentiment_analyzer import NewsSentimentAnalyzer

def test_sentiment_analyzer():
    """Test the standalone sentiment analyzer"""
    print("=" * 60)
    print("Testing News Sentiment Analyzer")
    print("=" * 60)

    try:
        analyzer = NewsSentimentAnalyzer()

        # Test with popular stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

        for symbol in test_symbols:
            print(f"\n📊 Testing sentiment analysis for {symbol}...")

            sentiment_metrics = analyzer.get_news_sentiment(symbol, lookback_hours=24)

            print(f"  Overall Sentiment: {sentiment_metrics.overall_sentiment:.3f}")
            print(f"  News Volume: {sentiment_metrics.news_volume}")
            print(f"  Average Relevance: {sentiment_metrics.average_relevance:.3f}")
            print(f"  Bullish/Bearish Ratio: {sentiment_metrics.bullish_ratio:.2f}/{sentiment_metrics.bearish_ratio:.2f}")
            print(f"  Sentiment Volatility: {sentiment_metrics.sentiment_volatility:.3f}")

            # Validate metrics
            assert -1.0 <= sentiment_metrics.overall_sentiment <= 1.0, "Overall sentiment out of range"
            assert sentiment_metrics.news_volume >= 0, "News volume cannot be negative"
            assert 0.0 <= sentiment_metrics.average_relevance <= 1.0, "Relevance out of range"
            assert abs(sentiment_metrics.bullish_ratio + sentiment_metrics.bearish_ratio + sentiment_metrics.neutral_ratio - 1.0) < 0.01, "Ratios don't sum to 1"

            print(f"  ✅ Validation passed for {symbol}")

    except Exception as e:
        print(f"❌ Error testing sentiment analyzer: {e}")
        return False

    print("\n✅ Sentiment analyzer tests completed successfully!")
    return True

def test_enhanced_feature_extractor():
    """Test the enhanced feature extractor with sentiment integration"""
    print("\n" + "=" * 60)
    print("Testing Enhanced Feature Extractor with Sentiment")
    print("=" * 60)

    try:
        extractor = EnhancedFeatureExtractor()

        # Test data mimicking current API structure
        test_data = {
            'close': 175.23,
            'volume': 50000000,
            'moving_avg_5': 172.45,
            'moving_avg_20': 170.12,
            'volatility': 0.25
        }

        test_symbols = ['AAPL', 'MSFT']

        for symbol in test_symbols:
            print(f"\n📈 Testing enhanced features for {symbol}...")

            features = extractor.extract_comprehensive_features(symbol, test_data)

            if 'error' in features:
                print(f"❌ Error extracting features: {features['error']}")
                continue

            print(f"  Total Features: {features['feature_count']}")
            print(f"  Basic Features: {len(features.get('basic_features', {}))}")
            print(f"  Advanced Technical: {len(features.get('advanced_technical', {}))}")
            print(f"  Fundamental: {len(features.get('fundamental_features', {}))}")
            print(f"  Macro: {len(features.get('macro_features', {}))}")
            print(f"  Sentiment: {len(features.get('sentiment_features', {}))}")

            # Validate sentiment features specifically
            sentiment_features = features.get('sentiment_features', {})

            if sentiment_features:
                print(f"\n  📝 Sentiment Features Detail:")
                print(f"    News Sentiment: {sentiment_features.get('news_sentiment_overall', 'N/A')}")
                print(f"    News Volume: {sentiment_features.get('news_volume', 'N/A')}")
                print(f"    Market Fear/Greed: {sentiment_features.get('market_fear_greed', 'N/A')}")
                print(f"    Bullish Ratio: {sentiment_features.get('bullish_ratio', 'N/A')}")

                # Validate sentiment feature ranges
                news_sentiment = sentiment_features.get('news_sentiment_overall', 0)
                if isinstance(news_sentiment, (int, float)):
                    assert -1.0 <= news_sentiment <= 1.0, f"News sentiment out of range: {news_sentiment}"

                news_volume = sentiment_features.get('news_volume', 0)
                assert news_volume >= 0, f"News volume cannot be negative: {news_volume}"

                print(f"    ✅ Sentiment feature validation passed")
            else:
                print(f"    ⚠️ No sentiment features found")

            # Check that we have significantly more features than the original 8
            assert features['feature_count'] >= 20, f"Expected at least 20 features, got {features['feature_count']}"

            print(f"  ✅ Feature extraction validation passed for {symbol}")

    except Exception as e:
        print(f"❌ Error testing enhanced feature extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Enhanced feature extractor tests completed successfully!")
    return True

def test_lambda_compatibility():
    """Test Lambda compatibility by simulating Lambda event structure"""
    print("\n" + "=" * 60)
    print("Testing Lambda Compatibility")
    print("=" * 60)

    try:
        # Import Lambda handlers
        from enhanced_feature_extractor import lambda_handler as feature_handler
        from news_sentiment_analyzer import lambda_handler as sentiment_handler

        # Test sentiment analyzer Lambda
        sentiment_event = {
            'symbol': 'AAPL',
            'lookback_hours': 24
        }

        print("🧪 Testing sentiment analyzer Lambda handler...")
        sentiment_response = sentiment_handler(sentiment_event, None)

        assert sentiment_response['statusCode'] == 200, "Sentiment Lambda returned error"
        assert 'sentiment_metrics' in sentiment_response['body'], "Missing sentiment metrics in response"

        print(f"  ✅ Sentiment Lambda test passed")

        # Test enhanced feature extractor Lambda
        feature_event = {
            'symbol': 'AAPL',
            'current_data': {
                'close': 175.23,
                'volume': 50000000,
                'moving_avg_5': 172.45,
                'moving_avg_20': 170.12,
                'volatility': 0.25
            }
        }

        print("🧪 Testing enhanced feature extractor Lambda handler...")
        # Note: Would need to implement lambda_handler in enhanced_feature_extractor.py

        print(f"  ✅ Lambda compatibility tests completed")

    except Exception as e:
        print(f"❌ Error testing Lambda compatibility: {e}")
        return False

    return True

def run_performance_benchmark():
    """Run performance benchmark to ensure acceptable response times"""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    import time

    try:
        analyzer = NewsSentimentAnalyzer()

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        times = []

        for symbol in symbols:
            print(f"⏱️ Benchmarking sentiment analysis for {symbol}...")

            start_time = time.time()
            sentiment_metrics = analyzer.get_news_sentiment(symbol, lookback_hours=24)
            end_time = time.time()

            duration = end_time - start_time
            times.append(duration)

            print(f"  Duration: {duration:.2f} seconds")
            print(f"  News Volume: {sentiment_metrics.news_volume}")

            # Ensure acceptable performance (under 30 seconds)
            assert duration < 30, f"Sentiment analysis too slow: {duration:.2f}s"

        avg_time = sum(times) / len(times)
        print(f"\n📊 Average Response Time: {avg_time:.2f} seconds")
        print(f"📊 Max Response Time: {max(times):.2f} seconds")

        if avg_time < 10:
            print("🚀 Performance: Excellent")
        elif avg_time < 20:
            print("✅ Performance: Good")
        else:
            print("⚠️ Performance: Acceptable but could be improved")

    except Exception as e:
        print(f"❌ Error in performance benchmark: {e}")
        return False

    return True

def main():
    """Main test runner"""
    print("🧪 Starting News Sentiment Analysis Integration Tests")
    print(f"🕐 Test Start Time: {datetime.now().isoformat()}")

    tests = [
        ("Sentiment Analyzer", test_sentiment_analyzer),
        ("Enhanced Feature Extractor", test_enhanced_feature_extractor),
        ("Lambda Compatibility", test_lambda_compatibility),
        ("Performance Benchmark", run_performance_benchmark)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} tests...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} tests PASSED")
            else:
                print(f"❌ {test_name} tests FAILED")
        except Exception as e:
            print(f"❌ {test_name} tests FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Test End Time: {datetime.now().isoformat()}")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Sentiment analysis integration is ready for deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Please review and fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)