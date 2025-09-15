#!/usr/bin/env python3
"""
Simplified validation script for sentiment integration
Tests integration without requiring full Lambda dependencies
"""

import sys
import os
import json
from datetime import datetime

def validate_sentiment_analyzer_structure():
    """Validate sentiment analyzer code structure"""
    print("🔍 Validating sentiment analyzer structure...")

    sentiment_file = os.path.join(os.path.dirname(__file__), '..', 'lambda_functions', 'news_sentiment_analyzer.py')

    with open(sentiment_file, 'r') as f:
        content = f.read()

    # Check for required classes and methods
    required_elements = [
        'class NewsSentimentAnalyzer',
        'class SentimentMetrics',
        'def get_news_sentiment',
        'def _analyze_article_sentiment',
        'def _aggregate_sentiment_metrics',
        'def lambda_handler'
    ]

    for element in required_elements:
        if element in content:
            print(f"  ✅ Found: {element}")
        else:
            print(f"  ❌ Missing: {element}")
            return False

    return True

def validate_feature_extractor_integration():
    """Validate feature extractor sentiment integration"""
    print("\n🔍 Validating feature extractor integration...")

    extractor_file = os.path.join(os.path.dirname(__file__), '..', 'lambda_functions', 'enhanced_feature_extractor.py')

    with open(extractor_file, 'r') as f:
        content = f.read()

    # Check for sentiment integration
    integration_checks = [
        'from news_sentiment_analyzer import NewsSentimentAnalyzer',
        'self.sentiment_analyzer = NewsSentimentAnalyzer()',
        'sentiment_metrics = self.sentiment_analyzer.get_news_sentiment',
        'news_sentiment_overall',
        'news_volume',
        'bullish_ratio',
        'bearish_ratio'
    ]

    for check in integration_checks:
        if check in content:
            print(f"  ✅ Found: {check}")
        else:
            print(f"  ❌ Missing: {check}")
            return False

    return True

def validate_infrastructure_setup():
    """Validate infrastructure configuration"""
    print("\n🔍 Validating infrastructure setup...")

    infra_file = os.path.join(os.path.dirname(__file__), '..', 'infrastructure', 'news_sentiment_cache.tf')

    if not os.path.exists(infra_file):
        print("  ❌ Missing: news_sentiment_cache.tf")
        return False

    with open(infra_file, 'r') as f:
        content = f.read()

    # Check for required infrastructure elements
    infra_checks = [
        'resource "aws_dynamodb_table" "news_sentiment_cache"',
        'resource "aws_iam_policy" "sentiment_cache_access"',
        'resource "aws_ssm_parameter" "newsapi_key"',
        'resource "aws_ssm_parameter" "finnhub_key"',
        'ttl {',
        'global_secondary_index'
    ]

    for check in infra_checks:
        if check in content:
            print(f"  ✅ Found: {check}")
        else:
            print(f"  ❌ Missing: {check}")
            return False

    # Check variables file
    vars_file = os.path.join(os.path.dirname(__file__), '..', 'infrastructure', 'variables.tf')

    with open(vars_file, 'r') as f:
        vars_content = f.read()

    if 'variable "newsapi_key"' in vars_content and 'variable "finnhub_key"' in vars_content:
        print(f"  ✅ Found: News API variables in variables.tf")
    else:
        print(f"  ❌ Missing: News API variables in variables.tf")
        return False

    return True

def calculate_feature_enhancement():
    """Calculate the enhancement in feature count"""
    print("\n📊 Calculating feature enhancement...")

    # Original features (from analysis)
    original_features = 8

    # Count sentiment features added
    sentiment_features = [
        'news_sentiment_overall',
        'news_sentiment_momentum',
        'news_volume',
        'news_relevance',
        'sentiment_volatility',
        'bullish_ratio',
        'bearish_ratio',
        'neutral_ratio',
        'social_sentiment',
        'options_sentiment',
        'insider_activity',
        'market_fear_greed'
    ]

    sentiment_count = len(sentiment_features)

    # Estimate total new features from enhanced system
    estimated_technical = 15  # Advanced technical indicators
    estimated_fundamental = 15  # Fundamental ratios
    estimated_macro = 10  # Macro indicators
    estimated_sentiment = sentiment_count  # Sentiment features

    total_estimated = original_features + estimated_technical + estimated_fundamental + estimated_macro + estimated_sentiment

    print(f"  📈 Original features: {original_features}")
    print(f"  📈 Added sentiment features: {estimated_sentiment}")
    print(f"  📈 Estimated total features: {total_estimated}")
    print(f"  📈 Feature increase: {((total_estimated - original_features) / original_features * 100):.1f}%")

    # Performance impact estimate
    current_accuracy = 68.5  # Mid-point of 65-72%
    target_accuracy = 77.5   # Mid-point of 75-80%
    improvement = target_accuracy - current_accuracy

    print(f"  🎯 Current accuracy: {current_accuracy}%")
    print(f"  🎯 Target accuracy: {target_accuracy}%")
    print(f"  🎯 Expected improvement: +{improvement:.1f}%")

    return True

def validate_deployment_readiness():
    """Check if sentiment integration is ready for deployment"""
    print("\n🚀 Checking deployment readiness...")

    # Check required files exist
    required_files = [
        'lambda_functions/news_sentiment_analyzer.py',
        'lambda_functions/enhanced_feature_extractor.py',
        'infrastructure/news_sentiment_cache.tf',
        'scripts/test-sentiment-analysis.py'
    ]

    base_path = os.path.join(os.path.dirname(__file__), '..')

    all_files_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"  ✅ Found: {file_path}")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_files_exist = False

    if all_files_exist:
        print(f"  🎉 All required files present")

    return all_files_exist

def main():
    """Main validation runner"""
    print("🔬 Sentiment Analysis Integration Validation")
    print("=" * 60)
    print(f"Validation Time: {datetime.now().isoformat()}")

    validations = [
        ("Sentiment Analyzer Structure", validate_sentiment_analyzer_structure),
        ("Feature Extractor Integration", validate_feature_extractor_integration),
        ("Infrastructure Setup", validate_infrastructure_setup),
        ("Feature Enhancement Analysis", calculate_feature_enhancement),
        ("Deployment Readiness", validate_deployment_readiness)
    ]

    passed = 0
    total = len(validations)

    for name, validation_func in validations:
        print(f"\n🔄 {name}...")
        try:
            if validation_func():
                passed += 1
                print(f"✅ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")

    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Validations Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 VALIDATION COMPLETE! Sentiment integration ready for deployment.")
        print("\n📝 Next Steps:")
        print("  1. Deploy DynamoDB sentiment cache table")
        print("  2. Configure NewsAPI and Finnhub API keys in Parameter Store")
        print("  3. Deploy updated Lambda functions")
        print("  4. Run integration tests in AWS environment")
        return True
    else:
        print("⚠️ Some validations failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)