"""Unit tests for configuration management."""

import os
import pytest
from src.config import Config, ConfigurationError


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ConfigurationError."""
        # Save original value
        original_key = os.environ.get("MISTRAL_API_KEY")
        
        try:
            # Set empty API key
            os.environ["MISTRAL_API_KEY"] = ""
            
            # Should raise error
            with pytest.raises(ConfigurationError) as exc_info:
                Config.from_env()
            
            assert "MISTRAL_API_KEY is required" in str(exc_info.value)
        finally:
            # Restore original value
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
    
    def test_invalid_api_key_format_raises_error(self):
        """Test that invalid API key format raises ConfigurationError."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        
        try:
            # Set invalid API key (too short)
            os.environ["MISTRAL_API_KEY"] = "short"
            
            with pytest.raises(ConfigurationError) as exc_info:
                Config.from_env()
            
            assert "invalid format" in str(exc_info.value)
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
    
    def test_invalid_numeric_ranges_raise_errors(self):
        """Test that invalid numeric values raise ConfigurationError."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        original_port = os.environ.get("FLASK_PORT")
        
        try:
            # Set valid API key
            os.environ["MISTRAL_API_KEY"] = "valid_test_key_12345"
            
            # Test invalid port
            os.environ["FLASK_PORT"] = "99999"
            
            with pytest.raises(ConfigurationError) as exc_info:
                Config.from_env()
            
            assert "FLASK_PORT" in str(exc_info.value)
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
            
            if original_port:
                os.environ["FLASK_PORT"] = original_port
            elif "FLASK_PORT" in os.environ:
                del os.environ["FLASK_PORT"]
    
    def test_valid_config_loads_successfully(self):
        """Test that valid configuration loads without errors."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        
        try:
            # Set valid API key
            os.environ["MISTRAL_API_KEY"] = "valid_test_key_12345"
            
            # Should not raise error
            config = Config.from_env()
            
            # Verify some values
            assert config.MISTRAL_API_KEY == "valid_test_key_12345"
            assert config.MISTRAL_MODEL == "mistral-large-latest"
            assert config.TOP_K_DOCUMENTS == 5
            assert config.CHUNK_SIZE == 512
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
    
    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that chunk overlap must be less than chunk size."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        original_size = os.environ.get("CHUNK_SIZE")
        original_overlap = os.environ.get("CHUNK_OVERLAP")
        
        try:
            os.environ["MISTRAL_API_KEY"] = "valid_test_key_12345"
            os.environ["CHUNK_SIZE"] = "100"
            os.environ["CHUNK_OVERLAP"] = "150"
            
            with pytest.raises(ConfigurationError) as exc_info:
                Config.from_env()
            
            assert "CHUNK_OVERLAP" in str(exc_info.value)
            assert "less than CHUNK_SIZE" in str(exc_info.value)
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
            
            if original_size:
                os.environ["CHUNK_SIZE"] = original_size
            elif "CHUNK_SIZE" in os.environ:
                del os.environ["CHUNK_SIZE"]
            
            if original_overlap:
                os.environ["CHUNK_OVERLAP"] = original_overlap
            elif "CHUNK_OVERLAP" in os.environ:
                del os.environ["CHUNK_OVERLAP"]
    
    def test_api_key_not_exposed_in_repr(self):
        """Test that API key is masked in string representation."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        
        try:
            os.environ["MISTRAL_API_KEY"] = "valid_test_key_12345678"
            
            config = Config.from_env()
            repr_str = repr(config)
            
            # Should not contain full API key
            assert "valid_test_key_12345678" not in repr_str
            # Should contain masked version
            assert "..." in repr_str
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]


class TestConfigDefaults:
    """Test configuration default values."""
    
    def test_default_values_are_sensible(self):
        """Test that default configuration values are sensible."""
        original_key = os.environ.get("MISTRAL_API_KEY")
        
        try:
            os.environ["MISTRAL_API_KEY"] = "valid_test_key_12345"
            
            config = Config.from_env()
            
            # Check defaults
            assert config.MISTRAL_MODEL == "mistral-large-latest"
            assert config.FAISS_INDEX_PATH == "data/faiss_index"
            assert config.EMBEDDING_DIMENSION == 1024
            assert config.TOP_K_DOCUMENTS == 5
            assert config.CHUNK_SIZE == 512
            assert config.CHUNK_OVERLAP == 50
            assert config.FLASK_HOST == "0.0.0.0"
            assert config.FLASK_PORT == 5000
            assert config.CACHE_TTL == 3600
            assert config.CACHE_MAX_SIZE == 1000
            assert config.LOG_LEVEL == "INFO"
            assert config.MAX_RETRIES == 3
            assert config.DEFAULT_TEMPERATURE == 0.7
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            elif "MISTRAL_API_KEY" in os.environ:
                del os.environ["MISTRAL_API_KEY"]
