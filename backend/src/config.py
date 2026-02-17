"""Configuration management for BizBot backend."""

import os
import re
from typing import List, Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """Configuration manager for BizBot backend.
    
    Loads configuration from environment variables with sensible defaults.
    Validates all configuration values on initialization.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Load .env file if it exists
        load_dotenv()
        
        # API Configuration
        self.MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
        self.MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        
        # FAISS Configuration
        self.FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        self.EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        
        # RAG Configuration
        self.TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # API Server Configuration
        self.FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
        self.FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
        self.FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
        cors_origins_str = os.getenv("CORS_ORIGINS", "*")
        self.CORS_ORIGINS: List[str] = [origin.strip() for origin in cors_origins_str.split(",")]
        
        # Cache Configuration
        self.CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
        self.CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        
        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE: str = os.getenv("LOG_FILE", "logs/bizbot.log")
        
        # Retry Configuration
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
        
        # LLM Generation Configuration
        self.DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
        self.RESERVED_TOKENS: int = int(os.getenv("RESERVED_TOKENS", "1000"))
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create Config instance from environment variables.
        
        Returns:
            Config: Configured instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        config = cls()
        config.validate()
        return config
    
    def validate(self) -> None:
        """Validate all configuration values.
        
        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        errors = []
        
        # Validate API key presence
        if not self.MISTRAL_API_KEY:
            errors.append("MISTRAL_API_KEY is required but not set")
        
        # Validate API key format (basic check)
        if self.MISTRAL_API_KEY and not self._is_valid_api_key_format(self.MISTRAL_API_KEY):
            errors.append("MISTRAL_API_KEY has invalid format")
        
        # Validate numeric ranges
        if self.EMBEDDING_DIMENSION <= 0:
            errors.append(f"EMBEDDING_DIMENSION must be positive, got {self.EMBEDDING_DIMENSION}")
        
        if self.TOP_K_DOCUMENTS <= 0:
            errors.append(f"TOP_K_DOCUMENTS must be positive, got {self.TOP_K_DOCUMENTS}")
        
        if self.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got {self.CHUNK_SIZE}")
        
        if self.CHUNK_OVERLAP < 0:
            errors.append(f"CHUNK_OVERLAP must be non-negative, got {self.CHUNK_OVERLAP}")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})")
        
        if self.FLASK_PORT <= 0 or self.FLASK_PORT > 65535:
            errors.append(f"FLASK_PORT must be between 1 and 65535, got {self.FLASK_PORT}")
        
        if self.CACHE_TTL <= 0:
            errors.append(f"CACHE_TTL must be positive, got {self.CACHE_TTL}")
        
        if self.CACHE_MAX_SIZE <= 0:
            errors.append(f"CACHE_MAX_SIZE must be positive, got {self.CACHE_MAX_SIZE}")
        
        if self.MAX_RETRIES < 0:
            errors.append(f"MAX_RETRIES must be non-negative, got {self.MAX_RETRIES}")
        
        if self.RETRY_BASE_DELAY <= 0:
            errors.append(f"RETRY_BASE_DELAY must be positive, got {self.RETRY_BASE_DELAY}")
        
        if self.DEFAULT_TEMPERATURE < 0 or self.DEFAULT_TEMPERATURE > 2:
            errors.append(f"DEFAULT_TEMPERATURE must be between 0 and 2, got {self.DEFAULT_TEMPERATURE}")
        
        if self.MAX_TOKENS <= 0:
            errors.append(f"MAX_TOKENS must be positive, got {self.MAX_TOKENS}")
        
        if self.RESERVED_TOKENS < 0:
            errors.append(f"RESERVED_TOKENS must be non-negative, got {self.RESERVED_TOKENS}")
        
        if self.RESERVED_TOKENS >= self.MAX_TOKENS:
            errors.append(f"RESERVED_TOKENS ({self.RESERVED_TOKENS}) must be less than MAX_TOKENS ({self.MAX_TOKENS})")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of {valid_log_levels}, got {self.LOG_LEVEL}")
        
        # If there are any errors, fail fast
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigurationError(error_message)
    
    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate API key format.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        # Basic validation: non-empty, reasonable length, alphanumeric with dashes/underscores
        if len(api_key) < 10:
            return False
        
        # Check for valid characters (alphanumeric, dashes, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of config (with API key masked)."""
        masked_key = f"{self.MISTRAL_API_KEY[:4]}...{self.MISTRAL_API_KEY[-4:]}" if len(self.MISTRAL_API_KEY) > 8 else "****"
        return (
            f"Config(\n"
            f"  MISTRAL_API_KEY={masked_key},\n"
            f"  MISTRAL_MODEL={self.MISTRAL_MODEL},\n"
            f"  FAISS_INDEX_PATH={self.FAISS_INDEX_PATH},\n"
            f"  EMBEDDING_DIMENSION={self.EMBEDDING_DIMENSION},\n"
            f"  TOP_K_DOCUMENTS={self.TOP_K_DOCUMENTS},\n"
            f"  CHUNK_SIZE={self.CHUNK_SIZE},\n"
            f"  CHUNK_OVERLAP={self.CHUNK_OVERLAP},\n"
            f"  FLASK_HOST={self.FLASK_HOST},\n"
            f"  FLASK_PORT={self.FLASK_PORT},\n"
            f"  CACHE_TTL={self.CACHE_TTL},\n"
            f"  LOG_LEVEL={self.LOG_LEVEL}\n"
            f")"
        )
