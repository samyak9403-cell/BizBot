"""BizBot Backend Application Entry Point.

Main application entry point for the BizBot backend. Initializes all components,
loads configuration, sets up the FAISS index, and starts the Flask server with
graceful shutdown handling.

Usage:
    python app.py
    
Environment Variables:
    FLASK_ENV: Flask environment (development/testing/production)
    FLASK_HOST: Server host (default: 0.0.0.0)
    FLASK_PORT: Server port (default: 5000)
    MISTRAL_API_KEY: Mistral AI API key (required)
    FAISS_INDEX_PATH: Path to FAISS index (default: data/faiss_index)
"""

import sys
import signal
import logging
from pathlib import Path
from typing import Optional

from src.config import Config, ConfigurationError
from src.mistral_client import MistralClient
from src.document_processor import DocumentProcessor
from src.faiss_retriever import FAISSRetriever
from src.cache_manager import CacheManager
from src.prompt_builder import PromptBuilder
from src.rag_pipeline import RAGPipeline
from src.api import create_app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BizBotApp:
    """Main application container for BizBot backend.
    
    Manages initialization of all components, configuration loading,
    and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize BizBot application components."""
        self.flask_app = None
        self.config = None
        self.mistral_client = None
        self.document_processor = None
        self.faiss_retriever = None
        self.cache_manager = None
        self.rag_pipeline = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self) -> None:
        """Initialize all application components.
        
        Loads configuration, initializes clients, loads FAISS index,
        and creates Flask app.
        
        Raises:
            ConfigurationError: If configuration is invalid
            FileNotFoundError: If required files are missing
            RuntimeError: If component initialization fails
        """
        try:
            logger.info("=" * 80)
            logger.info("BizBot Backend Initialization Starting")
            logger.info("=" * 80)
            
            # Step 1: Load configuration
            logger.info("Step 1/7: Loading configuration...")
            self.config = Config()
            self.config.validate()
            logger.info(f"✓ Configuration loaded (Flask env: {self.config.FLASK_ENV})")
            
            # Step 2: Initialize Mistral client
            logger.info("Step 2/7: Initializing Mistral AI client...")
            self.mistral_client = MistralClient(
                api_key=self.config.MISTRAL_API_KEY,
                model=self.config.MISTRAL_MODEL
            )
            logger.info(f"✓ Mistral client ready (model: {self.config.MISTRAL_MODEL})")
            
            # Step 3: Initialize document processor
            logger.info("Step 3/7: Initializing document processor...")
            self.document_processor = DocumentProcessor(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            logger.info(
                f"✓ Document processor ready "
                f"(chunk_size: {self.config.CHUNK_SIZE}, "
                f"overlap: {self.config.CHUNK_OVERLAP})"
            )
            
            # Step 4: Initialize FAISS retriever and load index
            logger.info("Step 4/7: Initializing FAISS retriever...")
            self.faiss_retriever = FAISSRetriever(
                mistral_client=self.mistral_client,
                index_path=self.config.FAISS_INDEX_PATH,
                embedding_dimension=self.config.EMBEDDING_DIMENSION
            )
            
            # Load existing index if available
            try:
                self.faiss_retriever.load_index()
                logger.info(
                    f"✓ FAISS index loaded from {self.config.FAISS_INDEX_PATH} "
                    f"({self.faiss_retriever.size} documents)"
                )
            except (FileNotFoundError, Exception) as e:
                logger.warning(
                    f"⚠ No existing FAISS index found at {self.config.FAISS_INDEX_PATH}. "
                    f"Run 'python build_knowledge_base.py' to create one. Error: {e}"
                )
            
            # Step 5: Initialize cache manager
            logger.info("Step 5/7: Initializing cache manager...")
            self.cache_manager = CacheManager(
                ttl=self.config.CACHE_TTL,
                max_size=self.config.CACHE_MAX_SIZE
            )
            logger.info(
                f"✓ Cache manager ready "
                f"(TTL: {self.config.CACHE_TTL}s, max_size: {self.config.CACHE_MAX_SIZE})"
            )
            
            # Step 6: Initialize prompt builder
            logger.info("Step 6/7: Initializing prompt builder...")
            self.prompt_builder = PromptBuilder(
                max_context_tokens=self.config.RESERVED_TOKENS
            )
            logger.info("✓ Prompt builder ready")
            
            # Step 7: Initialize RAG pipeline
            logger.info("Step 7/7: Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline(
                mistral_client=self.mistral_client,
                faiss_retriever=self.faiss_retriever,
                prompt_builder=self.prompt_builder,
                top_k_documents=self.config.TOP_K_DOCUMENTS
            )
            logger.info("✓ RAG pipeline ready")
            
            # Step 7: Create Flask app
            logger.info("Step 7/7: Creating Flask application...")
            self.flask_app = create_app(self.config)
            
            # Store initialized components in app context
            self.flask_app.config['RAG_PIPELINE'] = self.rag_pipeline
            self.flask_app.config['CACHE_MANAGER'] = self.cache_manager
            self.flask_app.config['FAISS_RETRIEVER'] = self.faiss_retriever
            self.flask_app.config['DOCUMENT_PROCESSOR'] = self.document_processor
            logger.info("✓ Flask application created")
            
            logger.info("=" * 80)
            logger.info("✅ BizBot Backend Initialization Complete")
            logger.info("=" * 80)
            logger.info(f"server will run on http://{self.config.FLASK_HOST}:{self.config.FLASK_PORT}")
            logger.info("Press Ctrl+C to shut down gracefully")
            
        except ConfigurationError as e:
            logger.error(f"❌ Configuration Error: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"❌ File Not Found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Initialization Error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize BizBot: {e}") from e
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: bool = False) -> None:
        """Run the Flask application.
        
        Args:
            host: Server host (defaults to config.FLASK_HOST)
            port: Server port (defaults to config.FLASK_PORT)
            debug: Enable Flask debug mode
        """
        if self.flask_app is None:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        
        host = host or self.config.FLASK_HOST
        port = port or self.config.FLASK_PORT
        
        logger.info(f"Starting Flask server on {host}:{port}")
        
        try:
            self.flask_app.run(
                host=host,
                port=port,
                debug=debug,
                use_reloader=False  # Disable reloader for CLI usage
            )
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.shutdown()
        except Exception as e:
            logger.error(f"Flask server error: {e}", exc_info=True)
            self.shutdown()
            raise
    
    def shutdown(self) -> None:
        """Perform graceful shutdown.
        
        Cleans up resources and saves state before shutdown.
        """
        logger.info("=" * 80)
        logger.info("Initiating graceful shutdown...")
        logger.info("=" * 80)
        
        try:
            # Step 1: Cache cleanup
            if self.cache_manager is not None:
                logger.info("Saving cache state...")
                try:
                    self.cache_manager.clear()
                    logger.info("✓ Cache cleared")
                except Exception as e:
                    logger.warning(f"Warning: Failed to clear cache: {e}")
            
            # Step 2: FAISS cleanup
            if self.faiss_retriever is not None:
                logger.info("Finalizing FAISS index...")
                try:
                    # Ensure index is saved if modified
                    if hasattr(self.faiss_retriever, 'save_index'):
                        self.faiss_retriever.save_index()
                        logger.info("✓ FAISS index saved")
                except Exception as e:
                    logger.warning(f"Warning: Failed to save FAISS index: {e}")
            
            # Step 3: Close connections
            logger.info("Closing connections...")
            if self.mistral_client is not None:
                try:
                    # MistralClient uses requests library which auto-closes
                    logger.info("✓ Mistral client closed")
                except Exception as e:
                    logger.warning(f"Warning: Failed to close Mistral client: {e}")
            
            # Step 4: Flush logs
            logger.info("Flushing logs...")
            for handler in logging.root.handlers:
                handler.flush()
            logger.info("✓ Logs flushed")
            
            logger.info("=" * 80)
            logger.info("✅ Graceful shutdown complete")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Signal frame
        """
        logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)
    
    def health_check(self) -> dict:
        """Perform application health check.
        
        Returns:
            dict: Health status with component information
        """
        health = {
            "status": "healthy",
            "components": {
                "config": "ok" if self.config else "missing",
                "mistral_client": "ok" if self.mistral_client else "missing",
                "document_processor": "ok" if self.document_processor else "missing",
                "faiss_retriever": "ok" if self.faiss_retriever else "missing",
                "cache_manager": "ok" if self.cache_manager else "missing",
                "rag_pipeline": "ok" if self.rag_pipeline else "missing",
                "flask_app": "ok" if self.flask_app else "missing"
            }
        }
        
        # Check for any missing components
        if any(status == "missing" for status in health["components"].values()):
            health["status"] = "degraded"
        
        # Add FAISS index status
        if self.faiss_retriever:
            health["faiss_index"] = {
                "loaded": not self.faiss_retriever.is_empty,
                "documents": self.faiss_retriever.size
            }
        
        # Add cache status
        if self.cache_manager:
            health["cache"] = {
                "size": len(self.cache_manager.cache)
            }
        
        return health


def main():
    """Application entry point."""
    try:
        # Create and initialize app
        app = BizBotApp()
        app.initialize()
        
        # Run Flask server
        app.run()
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
