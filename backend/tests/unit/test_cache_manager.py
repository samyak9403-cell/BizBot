"""
Unit tests for CacheManager
"""

import time
import pytest
from src.cache_manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager class"""
    
    def test_cache_initialization(self):
        """Test cache manager initialization with default values"""
        cache = CacheManager()
        assert cache.ttl == 3600
        assert cache.max_size == 1000
        assert cache.size() == 0
    
    def test_cache_initialization_custom_values(self):
        """Test cache manager initialization with custom values"""
        cache = CacheManager(ttl=1800, max_size=500)
        assert cache.ttl == 1800
        assert cache.max_size == 500
    
    def test_set_and_get(self):
        """Test basic set and get operations"""
        cache = CacheManager()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist"""
        cache = CacheManager()
        assert cache.get("nonexistent") is None
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = CacheManager()
        result = cache.get("missing_key")
        assert result is None
    
    def test_cache_hit(self):
        """Test cache hit returns correct value"""
        cache = CacheManager()
        cache.set("test_key", {"data": "test_value"})
        result = cache.get("test_key")
        assert result == {"data": "test_value"}
    
    def test_ttl_expiration(self):
        """Test that entries expire after TTL"""
        cache = CacheManager(ttl=1)  # 1 second TTL
        cache.set("key1", "value1")
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert cache.get("key1") is None
    
    def test_cache_size_limit(self):
        """Test that cache respects max_size limit"""
        cache = CacheManager(max_size=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.size() == 3
        
        # Adding a 4th item should evict the oldest
        cache.set("key4", "value4")
        
        assert cache.size() == 3
        assert cache.get("key1") is None  # Oldest should be evicted
        assert cache.get("key4") == "value4"  # Newest should be present
    
    def test_invalidate_by_pattern(self):
        """Test invalidating cache entries by pattern"""
        cache = CacheManager()
        
        cache.set("user:123:profile", {"name": "Alice"})
        cache.set("user:123:settings", {"theme": "dark"})
        cache.set("user:456:profile", {"name": "Bob"})
        
        # Invalidate all entries for user 123
        cache.invalidate("user:123")
        
        assert cache.get("user:123:profile") is None
        assert cache.get("user:123:settings") is None
        assert cache.get("user:456:profile") == {"name": "Bob"}
    
    def test_clear_cache(self):
        """Test clearing all cache entries"""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.size() == 3
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("key1") is None
    
    def test_generate_key_from_dict(self):
        """Test generating cache key from dictionary"""
        cache = CacheManager()
        
        data1 = {"user": "alice", "query": "test"}
        data2 = {"query": "test", "user": "alice"}  # Same data, different order
        data3 = {"user": "bob", "query": "test"}
        
        key1 = cache._generate_key(data1)
        key2 = cache._generate_key(data2)
        key3 = cache._generate_key(data3)
        
        # Same data should produce same key regardless of order
        assert key1 == key2
        # Different data should produce different key
        assert key1 != key3
    
    def test_cache_stats(self):
        """Test getting cache statistics"""
        cache = CacheManager(ttl=3600, max_size=100)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['max_size'] == 100
        assert stats['ttl'] == 3600
        assert 'expired_entries' in stats
    
    def test_update_existing_key(self):
        """Test updating an existing cache entry"""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"
    
    def test_cache_with_complex_values(self):
        """Test caching complex data structures"""
        cache = CacheManager()
        
        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "string": "test"
        }
        
        cache.set("complex", complex_value)
        result = cache.get("complex")
        
        assert result == complex_value
    
    def test_eviction_order(self):
        """Test that oldest entries are evicted first"""
        cache = CacheManager(max_size=2)
        
        cache.set("key1", "value1")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        cache.set("key2", "value2")
        
        assert cache.size() == 2
        
        # Adding key3 should evict key1 (oldest)
        cache.set("key3", "value3")
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_invalidate_empty_cache(self):
        """Test invalidating on empty cache doesn't cause errors"""
        cache = CacheManager()
        cache.invalidate("pattern")  # Should not raise error
        assert cache.size() == 0
    
    def test_invalidate_no_matches(self):
        """Test invalidating with pattern that matches nothing"""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.invalidate("nonexistent_pattern")
        
        # All entries should still be present
        assert cache.size() == 2
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
