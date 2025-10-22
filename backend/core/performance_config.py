"""
Performance configuration for optimized parallel search execution.
This module contains settings to maximize async performance.
"""

import asyncio
import logging

# Async performance settings
ASYNC_CONCURRENCY_LIMIT = 10  # Maximum concurrent async operations
ASYNC_TIMEOUT = 45.0  # Timeout for individual async operations
LLM_TIMEOUT = 120.0  # Timeout for LLM API calls (longer for complex queries)
SEARCH_BATCH_SIZE = 20  # Maximum results per search operation

# Connection pool settings for OpenAI API
OPENAI_MAX_CONNECTIONS = 20
OPENAI_MAX_KEEPALIVE_CONNECTIONS = 10
OPENAI_KEEPALIVE_EXPIRY = 30.0

# Search engine performance settings
VECTOR_SEARCH_TIMEOUT = 15.0
KEYWORD_SEARCH_TIMEOUT = 10.0
RRF_BATCH_SIZE = 100  # Process RRF in batches for large result sets

# Logging configuration for performance monitoring
PERFORMANCE_LOGGING = True
DETAILED_TIMING = True

def configure_async_performance():
    """
    Configure asyncio for optimal performance with multiple concurrent operations.
    """
    # Set asyncio policy for better performance on Linux
    if hasattr(asyncio, 'set_event_loop_policy'):
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logging.info("Using uvloop for enhanced async performance")
        except ImportError:
            logging.info("uvloop not available, using default asyncio policy")
    
    # Configure loop for high concurrency
    loop = asyncio.get_event_loop()
    if hasattr(loop, 'set_task_factory'):
        # Custom task factory could be set here for debugging if needed
        pass

def get_optimized_semaphore():
    """
    Get a semaphore configured for optimal concurrent search operations.
    """
    return asyncio.Semaphore(ASYNC_CONCURRENCY_LIMIT)

class PerformanceTimer:
    """
    Context manager for timing operations with detailed logging.
    """
    def __init__(self, operation_name: str, log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = asyncio.get_event_loop().time()
        if DETAILED_TIMING:
            logging.log(self.log_level, f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = asyncio.get_event_loop().time() - self.start_time
        if PERFORMANCE_LOGGING:
            logging.log(self.log_level, f"{self.operation_name} completed in {duration:.3f}s")

# Performance monitoring decorators
def time_async_operation(operation_name: str):
    """
    Decorator to time async operations.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with PerformanceTimer(f"{operation_name}({func.__name__})"):
                return await func(*args, **kwargs)
        return wrapper
    return decorator