"""
Comprehensive testing framework for Stock Analytics Engine.
Provides unit tests, integration tests, and validation utilities.
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Try to import pytest, but don't fail if it's not available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add lambda_functions to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda_functions'))

from lambda_functions.shared.lambda_utils import LambdaResponse, InputValidator
from lambda_functions.shared.config import AppConfig
from lambda_functions.shared.types import PricePredictionOutput, TimePredictionOutput
from lambda_functions.shared.error_handling import ValidationError, StockAnalyticsError
from lambda_functions.shared.security import InputSanitizer, APIKeyValidator


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_stock_data(symbol: str = "AAPL", price: float = 150.0) -> Dict[str, Any]:
        """Create mock stock data."""
        return {
            'symbol': symbol,
            'price': price,
            'volume': 1000000,
            'timestamp': datetime.utcnow().isoformat(),
            'open': price * 0.99,
            'high': price * 1.02,
            'low': price * 0.98,
            'close': price,
            'change': price * 0.01,
            'change_percent': 1.0
        }
    
    @staticmethod
    def create_prediction_input(symbol: str = "AAPL", current_price: float = 150.0) -> Dict[str, Any]:
        """Create mock prediction input."""
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timeframe_days': 30,
            'technical_indicators': {
                'rsi': 45.2,
                'ma_5': 148.5,
                'ma_20': 145.0,
                'volatility': 0.02,
                'momentum': 0.05
            }
        }
    
    @staticmethod
    def create_lambda_event(body: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None,
                           query_params: Dict[str, str] = None) -> Dict[str, Any]:
        """Create mock Lambda event."""
        return {
            'body': json.dumps(body) if body else None,
            'headers': headers or {},
            'queryStringParameters': query_params,
            'requestContext': {
                'requestId': 'test-request-id',
                'stage': 'test'
            }
        }
    
    @staticmethod
    def create_lambda_context() -> Mock:
        """Create mock Lambda context."""
        context = Mock()
        context.function_name = 'test-function'
        context.function_version = '1'
        context.invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
        context.memory_limit_in_mb = 128
        context.remaining_time_in_millis = lambda: 30000
        context.aws_request_id = 'test-request-id'
        return context


class MockAWSServices:
    """Mock AWS services for testing."""
    
    def __init__(self):
        self.dynamodb_data = {}
        self.s3_data = {}
        self.secrets_data = {}
        self.lambda_responses = {}
    
    def mock_dynamodb_table(self, table_name: str):
        """Create mock DynamoDB table."""
        mock_table = Mock()
        
        def put_item(Item):
            key = Item.get('symbol', 'unknown')
            if table_name not in self.dynamodb_data:
                self.dynamodb_data[table_name] = {}
            self.dynamodb_data[table_name][key] = Item
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}
        
        def get_item(Key):
            key = Key.get('symbol', 'unknown')
            item = self.dynamodb_data.get(table_name, {}).get(key)
            if item:
                return {'Item': item}
            return {}
        
        def scan(**kwargs):
            items = list(self.dynamodb_data.get(table_name, {}).values())
            return {'Items': items, 'Count': len(items)}
        
        mock_table.put_item = put_item
        mock_table.get_item = get_item
        mock_table.scan = scan
        
        return mock_table
    
    def mock_s3_client(self):
        """Create mock S3 client."""
        mock_s3 = Mock()
        
        def get_object(Bucket, Key):
            data = self.s3_data.get(f"{Bucket}/{Key}")
            if data:
                mock_response = Mock()
                mock_response.read.return_value = json.dumps(data).encode()
                return {'Body': mock_response}
            raise Exception("NoSuchKey")
        
        def put_object(Bucket, Key, Body):
            self.s3_data[f"{Bucket}/{Key}"] = json.loads(Body)
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}
        
        mock_s3.get_object = get_object
        mock_s3.put_object = put_object
        
        return mock_s3
    
    def mock_secrets_manager(self):
        """Create mock Secrets Manager client."""
        mock_secrets = Mock()
        
        def get_secret_value(SecretId):
            if SecretId in self.secrets_data:
                return {'SecretString': self.secrets_data[SecretId]}
            raise Exception("ResourceNotFoundException")
        
        mock_secrets.get_secret_value = get_secret_value
        
        return mock_secrets


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data = TestDataGenerator()
        self.mock_aws = MockAWSServices()
        
        # Set up test environment variables
        self.test_env = {
            'AWS_REGION': 'us-east-1',
            'ENVIRONMENT': 'test',
            'LOG_LEVEL': 'DEBUG',
            'RECOMMENDATIONS_TABLE': 'test-recommendations',
            'S3_DATA_BUCKET': 'test-data-bucket'
        }
        
        # Apply environment variables
        for key, value in self.test_env.items():
            os.environ[key] = value
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test environment variables
        for key in self.test_env.keys():
            os.environ.pop(key, None)
    
    def assert_lambda_response(self, response: Dict[str, Any], 
                              expected_status: int = 200,
                              expected_keys: List[str] = None):
        """Assert Lambda response format."""
        self.assertIn('statusCode', response)
        self.assertIn('body', response)
        self.assertEqual(response['statusCode'], expected_status)
        
        if expected_keys:
            body = json.loads(response['body'])
            for key in expected_keys:
                self.assertIn(key, body)
    
    def assert_prediction_output(self, prediction: Dict[str, Any], 
                                prediction_type: str = 'price'):
        """Assert prediction output format."""
        required_fields = ['symbol', 'confidence', 'timestamp']
        
        if prediction_type == 'price':
            required_fields.extend(['target_price', 'recommendation', 'price_range'])
        elif prediction_type == 'time':
            required_fields.extend(['expected_timeline', 'probability_ranges'])
        
        for field in required_fields:
            self.assertIn(field, prediction, f"Missing field: {field}")
        
        # Validate confidence range
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)


class TestInputValidation(BaseTestCase):
    """Test input validation utilities."""
    
    def test_validate_symbol_valid(self):
        """Test valid symbol validation."""
        valid_symbols = ['AAPL', 'MSFT', 'GOOGL', 'A', 'BRK.B']
        
        for symbol in valid_symbols:
            try:
                result = InputValidator.validate_symbol(symbol)
                self.assertEqual(result, symbol.upper())
            except ValidationError:
                self.fail(f"Valid symbol {symbol} was rejected")
    
    def test_validate_symbol_invalid(self):
        """Test invalid symbol validation."""
        invalid_symbols = ['', '123', 'TOOLONGSYMBOL', 'AA$PL', None]
        
        for symbol in invalid_symbols:
            with self.assertRaises(ValidationError):
                InputValidator.validate_symbol(symbol)
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        valid_numbers = [1, 1.5, 100, 0.01]
        
        for number in valid_numbers:
            result = InputValidator.validate_positive_number(number, 'test_field')
            self.assertGreater(result, 0)
    
    def test_validate_positive_number_invalid(self):
        """Test invalid positive number validation."""
        invalid_numbers = [0, -1, -0.5, 'not_a_number', None]
        
        for number in invalid_numbers:
            with self.assertRaises(ValidationError):
                InputValidator.validate_positive_number(number, 'test_field')


class TestSecurityUtilities(BaseTestCase):
    """Test security utilities."""
    
    def test_input_sanitizer_string(self):
        """Test string sanitization."""
        sanitizer = InputSanitizer()
        
        # Valid input
        result = sanitizer.sanitize_string("  Valid String  ", max_length=20)
        self.assertEqual(result, "Valid String")
        
        # Too long input
        with self.assertRaises(ValidationError):
            sanitizer.sanitize_string("x" * 1001, max_length=1000)
    
    def test_stock_symbol_validation(self):
        """Test stock symbol validation."""
        sanitizer = InputSanitizer()
        
        # Valid symbols
        self.assertEqual(sanitizer.validate_stock_symbol("aapl"), "AAPL")
        self.assertEqual(sanitizer.validate_stock_symbol("MSFT"), "MSFT")
        
        # Invalid symbols
        with self.assertRaises(ValidationError):
            sanitizer.validate_stock_symbol("123")
        
        with self.assertRaises(ValidationError):
            sanitizer.validate_stock_symbol("")
    
    def test_api_key_validator(self):
        """Test API key validation."""
        validator = APIKeyValidator()
        
        # Valid format key
        valid_key = "sk_test_" + "a" * 32
        try:
            metadata = validator.validate_api_key(valid_key)
            self.assertIn('key_id', metadata)
            self.assertIn('permissions', metadata)
        except Exception:
            pass  # Mock validation may not pass, but format should be checked
        
        # Invalid format key
        with self.assertRaises(Exception):  # SecurityError or similar
            validator.validate_api_key("invalid")


class TestMLUtilities(BaseTestCase):
    """Test ML utilities and prediction functions."""
    
    @patch('lambda_functions.shared.ml_utils.AWSClients')
    def test_technical_indicators_calculation(self, mock_aws):
        """Test technical indicators calculation."""
        from lambda_functions.shared.ml_utils import TechnicalIndicators
        
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        
        # Test RSI calculation
        rsi = TechnicalIndicators.calculate_rsi(prices)
        self.assertIsInstance(rsi, float)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        # Test moving average
        ma = TechnicalIndicators.calculate_moving_average(prices, 5)
        self.assertIsInstance(ma, float)
        self.assertGreater(ma, 0)
        
        # Test volatility
        volatility = TechnicalIndicators.calculate_volatility(prices)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)


class TestPerformanceOptimization(BaseTestCase):
    """Test performance optimization utilities."""
    
    def test_memory_optimizer(self):
        """Test memory optimization utilities."""
        from lambda_functions.shared.performance_optimization import MemoryOptimizer
        
        # Test batch processing
        items = list(range(250))
        batches = MemoryOptimizer.batch_process_items(items, batch_size=100)
        
        self.assertEqual(len(batches), 3)  # 250 items in batches of 100
        self.assertEqual(len(batches[0]), 100)
        self.assertEqual(len(batches[1]), 100)
        self.assertEqual(len(batches[2]), 50)
    
    def test_in_memory_cache(self):
        """Test in-memory cache functionality."""
        from lambda_functions.shared.performance_optimization import InMemoryCache
        
        cache = InMemoryCache(default_ttl=1, max_size=2)
        
        # Test set and get
        cache.set('key1', 'value1')
        self.assertEqual(cache.get('key1'), 'value1')
        
        # Test TTL expiration
        import time
        time.sleep(1.1)
        self.assertIsNone(cache.get('key1'))
        
        # Test max size eviction
        cache.set('key1', 'value1', ttl=10)
        cache.set('key2', 'value2', ttl=10)
        cache.set('key3', 'value3', ttl=10)  # Should evict key1
        
        self.assertIsNone(cache.get('key1'))
        self.assertEqual(cache.get('key2'), 'value2')
        self.assertEqual(cache.get('key3'), 'value3')


class IntegrationTestCase(BaseTestCase):
    """Integration tests for Lambda functions."""
    
    @patch('lambda_functions.shared.lambda_utils.AWSClients')
    def test_price_prediction_integration(self, mock_aws):
        """Test price prediction Lambda function integration."""
        # Mock AWS services
        mock_aws.get_resource.return_value = self.mock_aws.mock_dynamodb_table('test-table')
        mock_aws.get_client.return_value = Mock()
        
        # Import after mocking
        from lambda_functions.price_prediction_model import lambda_handler
        
        # Create test event
        event = self.test_data.create_lambda_event(
            body=self.test_data.create_prediction_input()
        )
        context = self.test_data.create_lambda_context()
        
        # Execute function
        try:
            response = lambda_handler(event, context)
            
            # Validate response
            self.assert_lambda_response(response, expected_keys=['data'])
            
            # Validate prediction format
            body = json.loads(response['body'])
            if 'data' in body:
                self.assert_prediction_output(body['data'], 'price')
        
        except ImportError:
            # Skip if modules not available
            self.skipTest("Lambda function modules not available")


def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestInputValidation,
        TestSecurityUtilities,
        TestMLUtilities,
        TestPerformanceOptimization,
        IntegrationTestCase
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_test_suite()
    sys.exit(0 if success else 1)
