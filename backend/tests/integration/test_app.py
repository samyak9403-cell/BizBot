"""Tests for BizBot application entry point.

Tests for app initialization, component setup, graceful shutdown,
and health checks.
"""

import pytest
import signal
import sys
from unittest.mock import Mock, patch, MagicMock, call

from app import BizBotApp
from src.config import Config, ConfigurationError


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.FLASK_ENV = 'testing'
    config.FLASK_HOST = '127.0.0.1'
    config.FLASK_PORT = 5000
    config.MISTRAL_API_KEY = 'test-key'
    config.MISTRAL_MODEL = 'mistral-large-latest'
    config.FAISS_INDEX_PATH = 'data/faiss_index'
    config.EMBEDDING_DIMENSION = 1024
    config.CHUNK_SIZE = 512
    config.CHUNK_OVERLAP = 50
    config.TOP_K_DOCUMENTS = 5
    config.MAX_TOKENS = 4000
    config.RESERVED_TOKENS = 1000
    config.DEFAULT_TEMPERATURE = 0.7
    config.CACHE_TTL = 3600
    config.CACHE_MAX_SIZE = 1000
    config.validate = Mock()
    return config


@pytest.fixture
def mock_mistral_client():
    """Create mock Mistral client."""
    return Mock()


@pytest.fixture
def mock_document_processor():
    """Create mock document processor."""
    return Mock()


@pytest.fixture
def mock_faiss_retriever():
    """Create mock FAISS retriever."""
    retriever = Mock()
    retriever.load_index = Mock(return_value=True)
    retriever.size = 5
    retriever.is_empty = False
    return retriever


@pytest.fixture
def mock_cache_manager():
    """Create mock cache manager."""
    cache = Mock()
    cache.cache = {}
    return cache


@pytest.fixture
def mock_rag_pipeline():
    """Create mock RAG pipeline."""
    return Mock()


@pytest.fixture
def mock_flask_app():
    """Create mock Flask app."""
    return Mock()


# ============================================================================
# BizBotApp Initialization Tests
# ============================================================================


class TestBizBotAppInitialization:
    """Tests for BizBotApp initialization."""
    
    def test_app_creation(self):
        """Test creating BizBotApp instance."""
        app = BizBotApp()
        
        assert app.flask_app is None
        assert app.config is None
        assert app.mistral_client is None
        assert app.document_processor is None
        assert app.faiss_retriever is None
        assert app.cache_manager is None
        assert app.rag_pipeline is None
    
    def test_signal_handlers_registered(self):
        """Test that signal handlers are registered."""
        with patch('signal.signal') as mock_signal:
            app = BizBotApp()
            
            # Verify signal handlers were registered for SIGINT and SIGTERM
            assert mock_signal.call_count >= 2


class TestBizBotAppComponentInitialization:
    """Tests for component initialization during app setup."""
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_initialize_success(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test successful initialization of all components."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        # Initialize app
        app = BizBotApp()
        app.initialize()
        
        # Verify all components were initialized
        assert app.config == mock_config
        assert app.mistral_client == mock_mistral_client
        assert app.document_processor == mock_document_processor
        assert app.faiss_retriever == mock_faiss_retriever
        assert app.cache_manager == mock_cache_manager
        assert app.rag_pipeline == mock_rag_pipeline
        assert app.flask_app == mock_flask_app
        
        # Verify Config was validated
        mock_config.validate.assert_called_once()
    
    @patch('app.Config')
    def test_initialize_with_invalid_config(self, mock_config_class, mock_config):
        """Test initialization fails with invalid configuration."""
        mock_config.validate.side_effect = ConfigurationError("Invalid config")
        mock_config_class.return_value = mock_config
        
        app = BizBotApp()
        
        with pytest.raises(ConfigurationError):
            app.initialize()
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_initialize_with_no_existing_index(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test initialization when FAISS index doesn't exist."""
        # Setup: Index doesn't exist
        mock_faiss_retriever.load_index = Mock(return_value=False)
        
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        app = BizBotApp()
        app.initialize()
        
        # Verify initialization still succeeds without index
        assert app.flask_app == mock_flask_app


# ============================================================================
# BizBotApp Run Tests
# ============================================================================


class TestBizBotAppRun:
    """Tests for running the Flask application."""
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_run_success(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test running Flask app successfully."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        # Mock Flask run to simulate KeyboardInterrupt
        mock_flask_app.run.side_effect = KeyboardInterrupt()
        
        app = BizBotApp()
        app.initialize()
        
        # Run app (should handle KeyboardInterrupt gracefully)
        app.run(debug=False)
        
        # Verify Flask was called with correct parameters
        mock_flask_app.run.assert_called_once_with(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False
        )
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_run_without_initialization(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test run fails if app not initialized."""
        app = BizBotApp()
        
        with pytest.raises(RuntimeError, match="Application not initialized"):
            app.run()
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_run_with_custom_host_port(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test run with custom host and port."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        # Mock Flask run to simulate KeyboardInterrupt
        mock_flask_app.run.side_effect = KeyboardInterrupt()
        
        app = BizBotApp()
        app.initialize()
        
        # Run with custom host/port
        app.run(host='0.0.0.0', port=8000)
        
        # Verify Flask was called with custom parameters
        mock_flask_app.run.assert_called_once_with(
            host='0.0.0.0',
            port=8000,
            debug=False,
            use_reloader=False
        )


# ============================================================================
# BizBotApp Shutdown Tests
# ============================================================================


class TestBizBotAppShutdown:
    """Tests for graceful shutdown."""
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_shutdown_success(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test graceful shutdown of all components."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        mock_faiss_retriever.save_index = Mock()
        mock_cache_manager.clear = Mock()
        
        app = BizBotApp()
        app.initialize()
        
        # Perform shutdown
        app.shutdown()
        
        # Verify cleanup calls
        mock_cache_manager.clear.assert_called_once()
        mock_faiss_retriever.save_index.assert_called_once()
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_shutdown_with_errors(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test shutdown continues even if components fail."""
        # Setup mocks - cache fails but FAISS still saves
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        mock_cache_manager.clear = Mock(side_effect=Exception("Cache error"))
        mock_faiss_retriever.save_index = Mock()
        
        app = BizBotApp()
        app.initialize()
        
        # Perform shutdown - should not raise despite cache error
        app.shutdown()
        
        # Verify cache.clear was attempted
        mock_cache_manager.clear.assert_called_once()
        # Verify FAISS still saved despite cache error
        mock_faiss_retriever.save_index.assert_called_once()


# ============================================================================
# BizBotApp Health Check Tests
# ============================================================================


class TestBizBotAppHealthCheck:
    """Tests for application health checks."""
    
    def test_health_check_uninitialized_app(self):
        """Test health check on uninitialized app."""
        app = BizBotApp()
        health = app.health_check()
        
        assert health["status"] == "degraded"
        assert all(status == "missing" for status in health["components"].values())
    
    @patch('app.create_app')
    @patch('app.RAGPipeline')
    @patch('app.CacheManager')
    @patch('app.FAISSRetriever')
    @patch('app.DocumentProcessor')
    @patch('app.MistralClient')
    @patch('app.Config')
    def test_health_check_fully_initialized_app(
        self,
        mock_config_class,
        mock_mistral_class,
        mock_processor_class,
        mock_faiss_class,
        mock_cache_class,
        mock_pipeline_class,
        mock_create_app,
        mock_config,
        mock_mistral_client,
        mock_document_processor,
        mock_faiss_retriever,
        mock_cache_manager,
        mock_rag_pipeline,
        mock_flask_app
    ):
        """Test health check on fully initialized app."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_mistral_class.return_value = mock_mistral_client
        mock_processor_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_cache_class.return_value = mock_cache_manager
        mock_pipeline_class.return_value = mock_rag_pipeline
        mock_create_app.return_value = mock_flask_app
        
        mock_faiss_retriever.is_empty = False
        mock_faiss_retriever.size = 5
        mock_cache_manager.cache = {"key1": "value1"}
        
        app = BizBotApp()
        app.initialize()
        
        health = app.health_check()
        
        assert health["status"] == "healthy"
        assert all(status == "ok" for status in health["components"].values())
        assert health["faiss_index"]["loaded"] is True
        assert health["faiss_index"]["documents"] == 5
        assert health["cache"]["size"] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
