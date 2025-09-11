"""
Performance optimization utilities for Stock Analytics Engine.
Provides caching, connection pooling, memory optimization, and cold start reduction.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc

from .lambda_utils import AWSClients
from .config import get_config, FeatureFlags
from .error_handling import StructuredLogger

config = get_config()
logger = StructuredLogger(__name__)

T = TypeVar('T')


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def clear_unused_objects() -> int:
        """Force garbage collection and return number of objects collected."""
        return gc.collect()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, int]:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss // 1024 // 1024,  # Resident Set Size
            'vms_mb': memory_info.vms // 1024 // 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def optimize_json_parsing(data: str) -> Dict[str, Any]:
        """Optimized JSON parsing with memory efficiency."""
        try:
            # Use ujson if available for better performance
            import ujson
            return ujson.loads(data)
        except ImportError:
            return json.loads(data)
    
    @staticmethod
    def batch_process_items(items: List[T], batch_size: int = 100) -> List[List[T]]:
        """Split items into batches for memory-efficient processing."""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class ConnectionPool:
    """Connection pool for AWS services to reduce cold start impact."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._clients = {}
        self._resources = {}
        self._lock = threading.Lock()
        self._initialized = True
    
    def get_client(self, service_name: str, region: str = None) -> Any:
        """Get cached AWS client."""
        cache_key = f"{service_name}_{region or config.get_aws_region()}"
        
        if cache_key not in self._clients:
            with self._lock:
                if cache_key not in self._clients:
                    self._clients[cache_key] = AWSClients.get_client(service_name, region)
        
        return self._clients[cache_key]
    
    def get_resource(self, service_name: str, region: str = None) -> Any:
        """Get cached AWS resource."""
        cache_key = f"{service_name}_{region or config.get_aws_region()}"
        
        if cache_key not in self._resources:
            with self._lock:
                if cache_key not in self._resources:
                    self._resources[cache_key] = AWSClients.get_resource(service_name, region)
        
        return self._resources[cache_key]
    
    def warm_up(self, services: List[str]) -> None:
        """Pre-warm connections for specified services."""
        for service in services:
            try:
                if service in ['dynamodb', 's3', 'lambda', 'cloudwatch']:
                    self.get_client(service)
                    if service == 'dynamodb':
                        self.get_resource(service)
                logger.log_debug(f"Warmed up {service} connection")
            except Exception as e:
                logger.log_warning(f"Failed to warm up {service}: {str(e)}")


class InMemoryCache:
    """High-performance in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() > entry['expires_at']:
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Evict if at max size
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            expires_at = time.time() + (ttl or self.default_ttl)
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            expired_count = sum(
                1 for entry in self._cache.values() 
                if now > entry['expires_at']
            )
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_entries': expired_count,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class AsyncProcessor:
    """Asynchronous processing utilities for improved performance."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch_async(self, items: List[T], 
                           process_func: Callable[[T], Any],
                           timeout: float = 30.0) -> List[Any]:
        """
        Process items asynchronously in batches.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            timeout: Timeout for all operations
        
        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []
        
        # Submit all tasks
        future_to_index = {}
        for i, item in enumerate(items):
            future = self.executor.submit(process_func, item)
            future_to_index[future] = i
        
        # Collect results
        results = [None] * len(items)
        completed_count = 0
        
        try:
            for future in as_completed(future_to_index.keys(), timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed_count += 1
                except Exception as e:
                    logger.log_warning(f"Task {index} failed: {str(e)}")
                    results[index] = None
        
        except TimeoutError:
            logger.log_warning(f"Batch processing timed out. Completed {completed_count}/{len(items)} tasks")
        
        return results
    
    def __del__(self):
        """Clean up executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global instances
connection_pool = ConnectionPool()
memory_cache = InMemoryCache(default_ttl=config.cache.default_ttl)


# Performance decorators
def cache_result(ttl: int = 300, key_func: Callable = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            memory_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def optimize_cold_start(warm_services: List[str] = None):
    """
    Decorator to optimize Lambda cold starts.
    
    Args:
        warm_services: List of AWS services to pre-warm
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Warm up connections on first call
            if warm_services and not getattr(wrapper, '_warmed_up', False):
                connection_pool.warm_up(warm_services)
                wrapper._warmed_up = True
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def batch_process(batch_size: int = 100, parallel: bool = True):
    """
    Decorator to process items in batches for better performance.
    
    Args:
        batch_size: Size of each batch
        parallel: Whether to process batches in parallel
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[T], *args, **kwargs):
            if not items:
                return []
            
            batches = MemoryOptimizer.batch_process_items(items, batch_size)
            
            if parallel and len(batches) > 1:
                processor = AsyncProcessor()
                batch_results = processor.process_batch_async(
                    batches,
                    lambda batch: func(batch, *args, **kwargs)
                )
                
                # Flatten results
                results = []
                for batch_result in batch_results:
                    if batch_result:
                        results.extend(batch_result)
                
                return results
            else:
                # Sequential processing
                results = []
                for batch in batches:
                    batch_result = func(batch, *args, **kwargs)
                    if batch_result:
                        results.extend(batch_result)
                
                return results
        
        return wrapper
    return decorator


# Performance monitoring utilities
class PerformanceProfiler:
    """Profile function performance and resource usage."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function call."""
        start_time = time.time()
        start_memory = MemoryOptimizer.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = MemoryOptimizer.get_memory_usage()
        
        profile_data = {
            'function_name': func.__name__,
            'duration_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
            'success': success,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store profile data
        if func.__name__ not in self.profiles:
            self.profiles[func.__name__] = []
        
        self.profiles[func.__name__].append(profile_data)
        
        # Keep only last 100 profiles per function
        if len(self.profiles[func.__name__]) > 100:
            self.profiles[func.__name__] = self.profiles[func.__name__][-100:]
        
        return profile_data
    
    def get_performance_summary(self, function_name: str) -> Dict[str, Any]:
        """Get performance summary for a function."""
        if function_name not in self.profiles:
            return {}
        
        profiles = self.profiles[function_name]
        durations = [p['duration_ms'] for p in profiles if p['success']]
        
        if not durations:
            return {'error': 'No successful executions found'}
        
        return {
            'function_name': function_name,
            'total_calls': len(profiles),
            'successful_calls': len(durations),
            'success_rate': len(durations) / len(profiles),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0
        }


# Global profiler instance
profiler = PerformanceProfiler()


# Utility functions
def optimize_lambda_memory() -> None:
    """Optimize Lambda memory usage."""
    # Clear unused objects
    collected = MemoryOptimizer.clear_unused_objects()
    
    # Clear cache if memory usage is high
    memory_usage = MemoryOptimizer.get_memory_usage()
    if memory_usage['percent'] > 80:
        memory_cache.clear()
        logger.log_info(f"Cleared cache due to high memory usage: {memory_usage['percent']}%")
    
    logger.log_debug(f"Memory optimization: collected {collected} objects, usage: {memory_usage}")


def preload_dependencies() -> None:
    """Preload common dependencies to reduce cold start time."""
    try:
        # Import commonly used modules
        import boto3
        import json
        import datetime
        import decimal
        
        # Warm up connection pool
        connection_pool.warm_up(['dynamodb', 's3', 'lambda'])
        
        logger.log_debug("Dependencies preloaded successfully")
        
    except Exception as e:
        logger.log_warning(f"Failed to preload dependencies: {str(e)}")


# Initialize performance optimizations
if FeatureFlags.is_caching_enabled():
    preload_dependencies()
