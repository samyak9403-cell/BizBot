"""
Cache Manager for BizBot Backend

Provides in-memory caching with TTL (Time To Live) and size limits.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    In-memory cache with TTL and size limits.
    
    Attributes:
        ttl: Time to live for cache entries in seconds
        max_size: Maximum number of entries in the cache
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            ttl: Time to live for cache entries in seconds (default: 3600 = 1 hour)
            max_size: Maximum number of entries in the cache (default: 1000)
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"CacheManager initialized with TTL={ttl}s, max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self._cache:
            logger.debug(f"Cache miss: {key}")
            return None
        
        entry = self._cache[key]
        current_time = time.time()
        
        # Check if entry has expired
        if current_time - entry['timestamp'] > self.ttl:
            logger.debug(f"Cache expired: {key}")
            del self._cache[key]
            return None
        
        logger.debug(f"Cache hit: {key}")
        return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Enforce cache size limit by removing oldest entries
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.debug(f"Cache set: {key} (cache size: {len(self._cache)})")
    
    def invalidate(self, pattern: str) -> None:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys (substring match)
        """
        keys_to_remove = [key for key in self._cache.keys() if pattern in key]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        if keys_to_remove:
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")
    
    def size(self) -> int:
        """
        Get current cache size.
        
        Returns:
            Number of entries in the cache
        """
        return len(self._cache)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry to make room for new entries."""
        if not self._cache:
            return
        
        # Find the oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
        del self._cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")
    
    def _generate_key(self, data: Dict) -> str:
        """
        Generate cache key from data.
        
        Args:
            data: Dictionary to generate key from
            
        Returns:
            Hash-based cache key
        """
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        hash_object = hashlib.sha256(sorted_data.encode())
        return hash_object.hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for entry in self._cache.values()
            if current_time - entry['timestamp'] > self.ttl
        )
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'expired_entries': expired_count
        }
