"""
Security utilities and best practices for Stock Analytics Engine.
Provides secure secret management, input validation, and security monitoring.
"""

import json
import hashlib
import hmac
import base64
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import secrets

from .lambda_utils import AWSClients
from .config import get_config
from .error_handling import StructuredLogger, ValidationError

config = get_config()
logger = StructuredLogger(__name__)


class SecretManager:
    """Secure secret management with caching and rotation support."""
    
    def __init__(self):
        self.secrets_client = AWSClients.get_client('secretsmanager')
        self._secret_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_secret(self, secret_arn: str, force_refresh: bool = False) -> str:
        """
        Retrieve secret from AWS Secrets Manager with caching.
        
        Args:
            secret_arn: ARN of the secret
            force_refresh: Force refresh from AWS
            
        Returns:
            Secret value as string
            
        Raises:
            SecurityError: If secret cannot be retrieved
        """
        cache_key = secret_arn
        now = datetime.utcnow().timestamp()
        
        # Check cache first
        if not force_refresh and cache_key in self._secret_cache:
            cached_entry = self._secret_cache[cache_key]
            if now - cached_entry['timestamp'] < self._cache_ttl:
                return cached_entry['value']
        
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_arn)
            secret_value = response['SecretString']
            
            # Cache the secret
            self._secret_cache[cache_key] = {
                'value': secret_value,
                'timestamp': now
            }
            
            logger.log_info(f"Retrieved secret: {secret_arn}")
            return secret_value
            
        except Exception as e:
            logger.log_error(e, context={'secret_arn': secret_arn})
            raise SecurityError(f"Failed to retrieve secret: {secret_arn}") from e
    
    def get_secret_json(self, secret_arn: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve JSON secret from AWS Secrets Manager.
        
        Args:
            secret_arn: ARN of the secret
            force_refresh: Force refresh from AWS
            
        Returns:
            Secret value as dictionary
        """
        secret_string = self.get_secret(secret_arn, force_refresh)
        try:
            return json.loads(secret_string)
        except json.JSONDecodeError as e:
            raise SecurityError(f"Secret is not valid JSON: {secret_arn}") from e
    
    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._secret_cache.clear()
        logger.log_info("Secret cache cleared")


class InputSanitizer:
    """Input sanitization and validation utilities."""
    
    # Regex patterns for validation
    PATTERNS = {
        'stock_symbol': re.compile(r'^[A-Z]{1,10}$'),
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
        'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    }
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, 
                       allowed_chars: str = None) -> str:
        """
        Sanitize string input.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            allowed_chars: Regex pattern for allowed characters
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string", "input", value)
        
        # Trim whitespace
        sanitized = value.strip()
        
        # Check length
        if len(sanitized) > max_length:
            raise ValidationError(f"Input too long (max {max_length} chars)", "input", len(sanitized))
        
        # Check allowed characters
        if allowed_chars and not re.match(allowed_chars, sanitized):
            raise ValidationError("Input contains invalid characters", "input", sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_stock_symbol(symbol: str) -> str:
        """Validate and normalize stock symbol."""
        if not symbol:
            raise ValidationError("Stock symbol is required", "symbol", symbol)
        
        normalized = symbol.upper().strip()
        
        if not InputSanitizer.PATTERNS['stock_symbol'].match(normalized):
            raise ValidationError("Invalid stock symbol format", "symbol", symbol)
        
        return normalized
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address."""
        if not email:
            raise ValidationError("Email is required", "email", email)
        
        normalized = email.lower().strip()
        
        if not InputSanitizer.PATTERNS['email'].match(normalized):
            raise ValidationError("Invalid email format", "email", email)
        
        return normalized
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: float = None, 
                              max_val: float = None, field_name: str = "value") -> Union[int, float]:
        """Validate numeric value within range."""
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be numeric", field_name, value)
        
        if min_val is not None and num_value < min_val:
            raise ValidationError(f"{field_name} must be >= {min_val}", field_name, value)
        
        if max_val is not None and num_value > max_val:
            raise ValidationError(f"{field_name} must be <= {max_val}", field_name, value)
        
        return num_value
    
    @staticmethod
    def sanitize_json_input(data: Dict[str, Any], 
                           allowed_keys: List[str] = None,
                           max_depth: int = 10) -> Dict[str, Any]:
        """
        Sanitize JSON input data.
        
        Args:
            data: Input dictionary
            allowed_keys: List of allowed keys (None = allow all)
            max_depth: Maximum nesting depth
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary", "data", type(data))
        
        def _sanitize_recursive(obj: Any, depth: int = 0) -> Any:
            if depth > max_depth:
                raise ValidationError(f"Data nesting too deep (max {max_depth})", "data", depth)
            
            if isinstance(obj, dict):
                sanitized = {}
                for key, value in obj.items():
                    # Validate key
                    if not isinstance(key, str):
                        continue  # Skip non-string keys
                    
                    if allowed_keys and key not in allowed_keys:
                        continue  # Skip disallowed keys
                    
                    # Recursively sanitize value
                    sanitized[key] = _sanitize_recursive(value, depth + 1)
                
                return sanitized
            
            elif isinstance(obj, list):
                return [_sanitize_recursive(item, depth + 1) for item in obj[:100]]  # Limit list size
            
            elif isinstance(obj, str):
                return InputSanitizer.sanitize_string(obj, max_length=10000)
            
            elif isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            
            else:
                # Convert unknown types to string
                return str(obj)[:1000]
        
        return _sanitize_recursive(data)


class SecurityError(Exception):
    """Security-related error."""
    pass


class APIKeyValidator:
    """API key validation and management."""
    
    def __init__(self):
        self.valid_keys_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def validate_api_key(self, api_key: str, required_permissions: List[str] = None) -> Dict[str, Any]:
        """
        Validate API key and check permissions.
        
        Args:
            api_key: API key to validate
            required_permissions: List of required permissions
            
        Returns:
            API key metadata
            
        Raises:
            SecurityError: If API key is invalid
        """
        if not api_key:
            raise SecurityError("API key is required")
        
        # Hash the API key for logging (never log the actual key)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        
        # Check cache first
        now = datetime.utcnow().timestamp()
        if api_key in self.valid_keys_cache:
            cached_entry = self.valid_keys_cache[api_key]
            if now - cached_entry['timestamp'] < self.cache_ttl:
                logger.log_debug(f"API key validated from cache: {key_hash}")
                return cached_entry['metadata']
        
        # Validate against stored keys (this would typically check a database)
        # For now, we'll implement a simple validation
        if self._is_valid_key_format(api_key):
            metadata = {
                'key_id': key_hash,
                'permissions': required_permissions or ['read'],
                'rate_limit': 1000,  # requests per hour
                'expires_at': (datetime.utcnow() + timedelta(days=30)).isoformat()
            }
            
            # Cache the result
            self.valid_keys_cache[api_key] = {
                'metadata': metadata,
                'timestamp': now
            }
            
            logger.log_info(f"API key validated: {key_hash}")
            return metadata
        
        logger.log_warning(f"Invalid API key attempted: {key_hash}")
        raise SecurityError("Invalid API key")
    
    def _is_valid_key_format(self, api_key: str) -> bool:
        """Check if API key has valid format."""
        # Simple format validation - in production, this would check against a database
        return (
            len(api_key) >= 32 and
            len(api_key) <= 128 and
            re.match(r'^[a-zA-Z0-9\-_]+$', api_key)
        )


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = datetime.utcnow().timestamp()
    
    def is_allowed(self, identifier: str, limit: int, window_seconds: int = 3600) -> bool:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP, API key, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed
        """
        now = datetime.utcnow().timestamp()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(now, window_seconds)
            self.last_cleanup = now
        
        # Get request history for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_times = self.requests[identifier]
        
        # Remove requests outside the window
        cutoff_time = now - window_seconds
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if limit exceeded
        if len(request_times) >= limit:
            logger.log_warning(f"Rate limit exceeded for {identifier}: {len(request_times)}/{limit}")
            return False
        
        # Add current request
        request_times.append(now)
        return True
    
    def _cleanup_old_entries(self, now: float, window_seconds: int) -> None:
        """Clean up old request entries."""
        cutoff_time = now - window_seconds
        
        for identifier in list(self.requests.keys()):
            request_times = self.requests[identifier]
            request_times[:] = [t for t in request_times if t > cutoff_time]
            
            # Remove empty entries
            if not request_times:
                del self.requests[identifier]


# Security decorators
def require_api_key(required_permissions: List[str] = None):
    """
    Decorator to require valid API key for function access.
    
    Args:
        required_permissions: List of required permissions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(event: Dict[str, Any], context: Any):
            # Extract API key from headers
            headers = event.get('headers', {})
            api_key = headers.get('X-API-Key') or headers.get('x-api-key')
            
            if not api_key:
                return {
                    'statusCode': 401,
                    'body': json.dumps({'error': 'API key required'})
                }
            
            try:
                validator = APIKeyValidator()
                key_metadata = validator.validate_api_key(api_key, required_permissions)
                
                # Add key metadata to event for use in function
                event['api_key_metadata'] = key_metadata
                
                return func(event, context)
                
            except SecurityError as e:
                return {
                    'statusCode': 401,
                    'body': json.dumps({'error': str(e)})
                }
        
        return wrapper
    return decorator


def rate_limit(requests_per_hour: int = 1000):
    """
    Decorator to apply rate limiting.
    
    Args:
        requests_per_hour: Maximum requests per hour
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter()
        
        @wraps(func)
        def wrapper(event: Dict[str, Any], context: Any):
            # Get identifier (API key, IP, etc.)
            headers = event.get('headers', {})
            api_key = headers.get('X-API-Key') or headers.get('x-api-key')
            source_ip = headers.get('X-Forwarded-For', 'unknown')
            
            identifier = api_key or source_ip
            
            if not limiter.is_allowed(identifier, requests_per_hour):
                return {
                    'statusCode': 429,
                    'body': json.dumps({
                        'error': 'Rate limit exceeded',
                        'limit': requests_per_hour,
                        'window': '1 hour'
                    })
                }
            
            return func(event, context)
        
        return wrapper
    return decorator


# Global instances
secret_manager = SecretManager()
input_sanitizer = InputSanitizer()


# Utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def hash_sensitive_data(data: str, salt: str = None) -> str:
    """Hash sensitive data with optional salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    return hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000).hex()


def verify_hmac_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature for webhook validation."""
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def mask_sensitive_data(data: str, mask_char: str = '*', 
                       visible_chars: int = 4) -> str:
    """Mask sensitive data for logging."""
    if len(data) <= visible_chars * 2:
        return mask_char * len(data)
    
    return data[:visible_chars] + mask_char * (len(data) - visible_chars * 2) + data[-visible_chars:]
