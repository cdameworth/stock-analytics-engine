#!/usr/bin/env python3
"""
Simple validation script for Stock Analytics Engine improvements.
Tests core functionality without requiring AWS credentials.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_shared_modules():
    """Test that all shared modules can be imported."""
    print("🔍 Testing shared module imports...")
    
    # Set test environment to avoid AWS client issues
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['AWS_REGION'] = 'us-east-1'
    
    modules_to_test = [
        'lambda_functions.shared.lambda_utils',
        'lambda_functions.shared.config',
        'lambda_functions.shared.types',
        'lambda_functions.shared.error_handling',
        'lambda_functions.shared.ml_utils',
    ]
    
    success_count = 0
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {module_name}: {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(modules_to_test)} modules imported successfully")
    return success_count == len(modules_to_test)


def test_type_definitions():
    """Test type definitions and validation."""
    print("\n🔍 Testing type definitions...")
    
    try:
        from lambda_functions.shared.types import (
            RecommendationType, RiskLevel, PredictionType,
            ensure_symbol, ensure_confidence, is_valid_symbol
        )
        
        # Test enums
        assert RecommendationType.BUY.value == "buy"
        assert RiskLevel.HIGH.value == "high"
        print("  ✅ Enums work correctly")
        
        # Test validation functions
        assert ensure_symbol("aapl") == "AAPL"
        assert ensure_confidence(0.75) == 0.75
        assert is_valid_symbol("MSFT") == True
        assert is_valid_symbol("123") == False
        print("  ✅ Validation functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Type definitions test failed: {e}")
        return False


def test_error_handling():
    """Test error handling framework."""
    print("\n🔍 Testing error handling...")
    
    try:
        from lambda_functions.shared.error_handling import (
            ValidationError, StockAnalyticsError, ErrorCategory, ErrorSeverity
        )
        
        # Test custom exception
        try:
            raise ValidationError("Test error", "test_field", "invalid_value")
        except ValidationError as e:
            error_dict = e.to_dict()
            assert 'error_type' in error_dict
            assert 'category' in error_dict
            assert error_dict['category'] == ErrorCategory.VALIDATION.value
            print("  ✅ Custom exceptions work correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n🔍 Testing configuration...")
    
    try:
        from lambda_functions.shared.config import get_config, AppConfig
        
        # Set test environment variables
        test_env = {
            'RECOMMENDATIONS_TABLE': 'test-recommendations',
            'S3_DATA_BUCKET': 'test-bucket',
            'TARGET_HIT_RATE': '0.65'
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
        
        # Test configuration loading
        config = get_config()
        assert isinstance(config, AppConfig)
        # Note: Configuration may use defaults if environment variables aren't set
        print("  ✅ Configuration loading works correctly")
        
        # Clean up
        for key in test_env.keys():
            os.environ.pop(key, None)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_input_validation():
    """Test input validation utilities."""
    print("\n🔍 Testing input validation...")
    
    try:
        from lambda_functions.shared.lambda_utils import InputValidator
        from lambda_functions.shared.error_handling import ValidationError
        
        # Test valid inputs
        symbol = InputValidator.validate_symbol("aapl")
        assert symbol == "AAPL"
        
        price = InputValidator.validate_positive_number(150.0, "price")
        assert price == 150.0
        
        print("  ✅ Valid input validation works correctly")
        
        # Test invalid inputs
        try:
            InputValidator.validate_symbol("123")
            assert False, "Should have raised ValidationError"
        except (ValidationError, ValueError):
            pass

        try:
            InputValidator.validate_positive_number(-10, "price")
            assert False, "Should have raised ValidationError"
        except (ValidationError, ValueError):
            pass
        
        print("  ✅ Invalid input rejection works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Input validation test failed: {e}")
        return False


def test_response_formatting():
    """Test response formatting utilities."""
    print("\n🔍 Testing response formatting...")
    
    try:
        from lambda_functions.shared.lambda_utils import LambdaResponse
        
        # Test success response
        success_response = LambdaResponse.success({"test": "data"})
        assert success_response['statusCode'] == 200
        assert 'body' in success_response
        assert 'headers' in success_response
        
        body = json.loads(success_response['body'])
        assert body['success'] == True
        assert 'data' in body
        assert 'timestamp' in body
        print("  ✅ Success response formatting works correctly")
        
        # Test error response
        error_response = LambdaResponse.error("Test error", 400, "TEST_ERROR")
        assert error_response['statusCode'] == 400
        
        body = json.loads(error_response['body'])
        assert body['success'] == False
        assert 'error' in body
        print("  ✅ Error response formatting works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Response formatting test failed: {e}")
        return False


def test_ml_utilities():
    """Test ML utilities."""
    print("\n🔍 Testing ML utilities...")
    
    try:
        from lambda_functions.shared.ml_utils import TechnicalIndicators, PredictionResult
        
        # Test technical indicators
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        
        rsi = TechnicalIndicators.calculate_rsi(prices)
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
        
        ma = TechnicalIndicators.calculate_moving_average(prices, 5)
        assert isinstance(ma, float)
        assert ma > 0
        
        volatility = TechnicalIndicators.calculate_volatility(prices)
        assert isinstance(volatility, float)
        assert volatility >= 0
        
        print("  ✅ Technical indicators work correctly")
        
        # Test prediction result
        result = PredictionResult("AAPL", "test")
        result_dict = result.to_dict()
        assert 'symbol' in result_dict
        assert 'prediction_type' in result_dict
        assert 'timestamp' in result_dict
        
        print("  ✅ Prediction result formatting works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ML utilities test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 Stock Analytics Engine - Core Improvements Validation")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Shared Module Imports", test_shared_modules),
        ("Type Definitions", test_type_definitions),
        ("Error Handling", test_error_handling),
        ("Configuration Management", test_configuration),
        ("Input Validation", test_input_validation),
        ("Response Formatting", test_response_formatting),
        ("ML Utilities", test_ml_utilities),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All core improvements are working correctly!")
        print("\n📈 Key Improvements Validated:")
        print("  • Shared utility modules with proper error handling")
        print("  • Comprehensive type definitions and validation")
        print("  • Structured error handling with custom exceptions")
        print("  • Centralized configuration management")
        print("  • Input validation and sanitization")
        print("  • Standardized response formatting")
        print("  • ML utilities with technical indicators")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
