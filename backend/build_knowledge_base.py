#!/usr/bin/env python3
"""CLI script for building and initializing the FAISS knowledge base.

Processes documents from data/documents/, chunks them, generates embeddings,
and builds a FAISS index for semantic search.
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

from src.config import Config, ConfigurationError
from src.mistral_client import MistralClient, MistralAPIError
from src.faiss_retriever import FAISSRetriever
from src.document_processor import DocumentProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """Build and initialize the FAISS knowledge base."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize knowledge base builder.
        
        Args:
            config: Configuration object (creates default if None)
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config or Config()
        self.config.validate()
        
        logger.info("Initializing knowledge base builder...")
        
        # Initialize Mistral client for embeddings
        try:
            self.mistral_client = MistralClient(
                api_key=self.config.MISTRAL_API_KEY,
                model=self.config.MISTRAL_MODEL
            )
            logger.info("Mistral client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {str(e)}")
            raise ConfigurationError("Mistral initialization failed") from e
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        logger.info("Document processor initialized")
        
        # Initialize FAISS retriever
        try:
            self.faiss_retriever = FAISSRetriever(
                mistral_client=self.mistral_client,
                index_path=self.config.FAISS_INDEX_PATH,
                embedding_dimension=self.config.EMBEDDING_DIMENSION
            )
            logger.info("FAISS retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS retriever: {str(e)}")
            raise ConfigurationError("FAISS initialization failed") from e
    
    def build_knowledge_base(
        self,
        documents_dir: str = "data/documents/",
        rebuild: bool = False,
        verbose: bool = False
    ) -> int:
        """Build knowledge base from documents.
        
        Args:
            documents_dir: Directory containing documents
            rebuild: If True, rebuild index; if False, update existing
            verbose: If True, print detailed progress
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            documents_path = Path(documents_dir)
            
            # Validate documents directory
            if not documents_path.exists():
                logger.error(f"Documents directory not found: {documents_dir}")
                return 1
            
            if not documents_path.is_dir():
                logger.error(f"Path is not a directory: {documents_dir}")
                return 1
            
            logger.info(f"Loading documents from {documents_dir}")
            
            # Load documents
            documents = self.document_processor.load_documents(str(documents_path))
            
            if not documents:
                logger.warning(f"No documents found in {documents_dir}")
                return 1
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Process and chunk documents
            logger.info("Processing and chunking documents...")
            chunked_documents = []
            
            for doc in documents:
                try:
                    chunks = self.document_processor.chunk_document(doc)
                    chunked_documents.extend(chunks)
                    
                    if verbose:
                        logger.info(f"Chunked {doc.metadata.source} into {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Failed to chunk {doc.metadata.source}: {str(e)}")
                    continue
            
            if not chunked_documents:
                logger.error("No chunks created from documents")
                return 1
            
            logger.info(f"Created {len(chunked_documents)} document chunks")
            
            # Build or update FAISS index
            if rebuild or self.faiss_retriever.is_empty:
                logger.info("Building new FAISS index...")
                self.faiss_retriever.build_index(chunked_documents)
            else:
                logger.info("Adding documents to existing FAISS index...")
                self.faiss_retriever.add_documents(chunked_documents)
            
            logger.info(f"FAISS index now contains {self.faiss_retriever.size} documents")
            
            # Save index
            logger.info("Saving FAISS index to disk...")
            self.faiss_retriever.save_index()
            logger.info(f"Index saved to {self.config.FAISS_INDEX_PATH}")
            
            logger.info("✓ Knowledge base built successfully!")
            return 0
            
        except MistralAPIError as e:
            logger.error(f"Mistral API error: {str(e)}")
            return 1
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {str(e)}", exc_info=True)
            return 1


def main():
    """Main entry point for CLI script."""
    parser = argparse.ArgumentParser(
        description="Build and initialize the FAISS knowledge base for BizBot"
    )
    
    parser.add_argument(
        '--documents-dir',
        default='data/documents/',
        help='Directory containing documents (default: data/documents/)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild index from scratch (default: update existing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        logger.info("Starting knowledge base builder...")
        builder = KnowledgeBaseBuilder()
        exit_code = builder.build_knowledge_base(
            documents_dir=args.documents_dir,
            rebuild=args.rebuild,
            verbose=args.verbose
        )
        
        if exit_code != 0:
            logger.error("Knowledge base build failed")
        
        return exit_code
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
