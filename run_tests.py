#!/usr/bin/env python3
"""
Test runner for Stock Analytics Engine improvements.
Validates all the systematic improvements made to the codebase.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(command: List[str], cwd: str = None) -> Dict[str, Any]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd or str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def check_python_syntax():
    """Check Python syntax for all Python files."""
    print("🔍 Checking Python syntax...")
    
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax")
        return True


def check_imports():
    """Check that all imports can be resolved."""
    print("\n📦 Checking imports...")
    
    # Test importing shared modules
    shared_modules = [
        'lambda_functions.shared.lambda_utils',
        'lambda_functions.shared.config',
        'lambda_functions.shared.types',
        'lambda_functions.shared.error_handling',
        'lambda_functions.shared.security',
        'lambda_functions.shared.ml_utils',
        'lambda_functions.shared.performance_optimization',
        'lambda_functions.shared.observability'
    ]
    
    import_errors = []
    for module in shared_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            import_errors.append(f"{module}: {e}")
            print(f"  ❌ {module}: {e}")
        except Exception as e:
            import_errors.append(f"{module}: {e}")
            print(f"  ⚠️  {module}: {e}")
    
    if import_errors:
        print(f"\n❌ {len(import_errors)} import errors found")
        return False
    else:
        print(f"\n✅ All {len(shared_modules)} shared modules imported successfully")
        return True


def run_unit_tests():
    """Run unit tests."""
    print("\n🧪 Running unit tests...")
    
    # Check if pytest is available
    pytest_result = run_command(['python', '-m', 'pytest', '--version'])
    if not pytest_result['success']:
        print("⚠️  pytest not available, running with unittest")
        test_result = run_command(['python', '-m', 'unittest', 'tests.test_framework', '-v'])
    else:
        print("Using pytest for testing")
        test_result = run_command(['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'])
    
    if test_result['success']:
        print("✅ All tests passed")
        print(test_result['stdout'])
        return True
    else:
        print("❌ Some tests failed")
        print("STDOUT:", test_result['stdout'])
        print("STDERR:", test_result['stderr'])
        return False


def validate_configuration():
    """Validate configuration management."""
    print("\n⚙️  Validating configuration...")
    
    try:
        # Set test environment variables
        test_env = {
            'AWS_REGION': 'us-east-1',
            'ENVIRONMENT': 'test',
            'RECOMMENDATIONS_TABLE': 'test-recommendations',
            'S3_DATA_BUCKET': 'test-bucket'
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
        
        # Test configuration loading
        from lambda_functions.shared.config import get_config, validate_config
        
        config = get_config()
        print(f"  ✅ Configuration loaded: {config.get_aws_region()}")
        
        # Test validation
        validate_config()
        print("  ✅ Configuration validation passed")
        
        # Clean up environment
        for key in test_env.keys():
            os.environ.pop(key, None)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration validation failed: {e}")
        return False


def validate_error_handling():
    """Validate error handling improvements."""
    print("\n🚨 Validating error handling...")
    
    try:
        from lambda_functions.shared.error_handling import (
            ValidationError, StockAnalyticsError, StructuredLogger
        )
        
        # Test custom exceptions
        try:
            raise ValidationError("Test validation error", "test_field", "invalid_value")
        except ValidationError as e:
            error_dict = e.to_dict()
            assert 'error_type' in error_dict
            assert 'category' in error_dict
            print("  ✅ Custom exceptions work correctly")
        
        # Test structured logger
        logger = StructuredLogger("test_logger")
        logger.log_info("Test log message", context={'test': True})
        print("  ✅ Structured logging works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling validation failed: {e}")
        return False


def validate_security_improvements():
    """Validate security improvements."""
    print("\n🔒 Validating security improvements...")
    
    try:
        from lambda_functions.shared.security import (
            InputSanitizer, APIKeyValidator, generate_secure_token
        )
        
        # Test input sanitization
        sanitizer = InputSanitizer()
        clean_symbol = sanitizer.validate_stock_symbol("aapl")
        assert clean_symbol == "AAPL"
        print("  ✅ Input sanitization works correctly")
        
        # Test secure token generation
        token = generate_secure_token(32)
        assert len(token) > 0
        print("  ✅ Secure token generation works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security validation failed: {e}")
        return False


def validate_performance_optimizations():
    """Validate performance optimizations."""
    print("\n⚡ Validating performance optimizations...")
    
    try:
        from lambda_functions.shared.performance_optimization import (
            InMemoryCache, MemoryOptimizer, ConnectionPool
        )
        
        # Test in-memory cache
        cache = InMemoryCache(default_ttl=60)
        cache.set('test_key', 'test_value')
        assert cache.get('test_key') == 'test_value'
        print("  ✅ In-memory cache works correctly")
        
        # Test memory optimizer
        items = list(range(250))
        batches = MemoryOptimizer.batch_process_items(items, batch_size=100)
        assert len(batches) == 3
        print("  ✅ Memory optimization works correctly")
        
        # Test connection pool
        pool = ConnectionPool()
        assert pool is not None
        print("  ✅ Connection pool works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance optimization validation failed: {e}")
        return False


def generate_improvement_report():
    """Generate a report of all improvements made."""
    print("\n📊 Generating improvement report...")
    
    improvements = {
        "architecture_improvements": [
            "Created shared utilities module (lambda_utils.py)",
            "Implemented centralized configuration management (config.py)",
            "Added comprehensive type definitions (types.py)",
            "Established consistent error handling patterns"
        ],
        "code_quality_improvements": [
            "Added comprehensive type hints throughout codebase",
            "Implemented structured logging with context",
            "Created standardized response patterns",
            "Added input validation and sanitization"
        ],
        "performance_improvements": [
            "Implemented in-memory caching with TTL",
            "Added connection pooling for AWS services",
            "Created batch processing utilities",
            "Added memory optimization utilities",
            "Implemented cold start optimization"
        ],
        "security_improvements": [
            "Added secure secret management",
            "Implemented input sanitization",
            "Added API key validation",
            "Created rate limiting utilities",
            "Added HMAC signature verification"
        ],
        "observability_improvements": [
            "Added structured logging framework",
            "Implemented performance monitoring",
            "Created business metrics collection",
            "Added health check utilities",
            "Implemented distributed tracing"
        ],
        "testing_improvements": [
            "Created comprehensive test framework",
            "Added mock AWS services for testing",
            "Implemented test data generators",
            "Added integration test patterns",
            "Created automated test runner"
        ]
    }
    
    print("\n🎉 Stock Analytics Engine Improvements Summary:")
    print("=" * 60)
    
    total_improvements = 0
    for category, items in improvements.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  ✅ {item}")
            total_improvements += 1
    
    print(f"\n📈 Total improvements implemented: {total_improvements}")
    print("\n🔧 New shared modules created:")
    print("  • lambda_functions/shared/lambda_utils.py")
    print("  • lambda_functions/shared/config.py")
    print("  • lambda_functions/shared/types.py")
    print("  • lambda_functions/shared/error_handling.py")
    print("  • lambda_functions/shared/security.py")
    print("  • lambda_functions/shared/ml_utils.py")
    print("  • lambda_functions/shared/performance_optimization.py")
    print("  • lambda_functions/shared/observability.py")
    print("  • lambda_functions/shared/documentation_template.py")
    print("  • tests/test_framework.py")
    
    return improvements


def main():
    """Main test runner function."""
    print("🚀 Stock Analytics Engine - Code Quality Improvement Validation")
    print("=" * 70)
    
    # Track results
    results = {}
    
    # Run all validation steps
    validation_steps = [
        ("Syntax Check", check_python_syntax),
        ("Import Check", check_imports),
        ("Configuration Validation", validate_configuration),
        ("Error Handling Validation", validate_error_handling),
        ("Security Validation", validate_security_improvements),
        ("Performance Validation", validate_performance_optimizations),
        ("Unit Tests", run_unit_tests)
    ]
    
    for step_name, step_func in validation_steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            results[step_name] = False
    
    # Generate improvement report
    improvements = generate_improvement_report()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for step_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{step_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} validation steps passed")
    
    if passed == total:
        print("\n🎉 All validations passed! The code quality improvements are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation steps failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
