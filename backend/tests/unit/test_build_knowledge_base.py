"""Tests for knowledge base builder CLI script.

Tests document loading, chunking, and knowledge base initialization.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from build_knowledge_base import KnowledgeBaseBuilder
from src.config import Config, ConfigurationError
from src.document_processor import Document, DocumentMetadata


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.FLASK_ENV = 'testing'
    return config


@pytest.fixture
def temp_documents_dir():
    """Create temporary documents directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create sample documents
        (tmpdir_path / "test1.txt").write_text("Test document one content")
        (tmpdir_path / "test2.md").write_text("# Test document two\n\nContent here")
        (tmpdir_path / "test3.json").write_text('{"content": "Test document three"}')
        
        yield str(tmpdir_path)


@pytest.fixture
def mock_mistral_client():
    """Create mock Mistral client."""
    mock_client = Mock()
    mock_client.embed.return_value = [[0.1, 0.2, 0.3] * 341 + [0.1]]  # 1024 dimensions
    return mock_client


@pytest.fixture
def mock_faiss_retriever():
    """Create mock FAISS retriever."""
    mock_retriever = Mock()
    mock_retriever.is_empty = True
    mock_retriever.size = 10
    return mock_retriever


@pytest.fixture
def mock_document_processor():
    """Create mock document processor."""
    mock_processor = Mock()
    
    # Create mock documents
    mock_doc1 = Mock(spec=Document)
    mock_doc1.metadata = Mock(spec=DocumentMetadata)
    mock_doc1.metadata.source = "test1.txt"
    mock_doc1.content = "Test content one"
    
    mock_doc2 = Mock(spec=Document)
    mock_doc2.metadata = Mock(spec=DocumentMetadata)
    mock_doc2.metadata.source = "test2.md"
    mock_doc2.content = "Test content two"
    
    mock_processor.load_documents.return_value = [mock_doc1, mock_doc2]
    mock_processor.chunk_document.side_effect = lambda doc: [
        Mock(spec=Document, content=f"chunk1_{doc.metadata.source}"),
        Mock(spec=Document, content=f"chunk2_{doc.metadata.source}")
    ]
    
    return mock_processor


# ============================================================================
# KnowledgeBaseBuilder Tests
# ============================================================================


class TestKnowledgeBaseBuilderInitialization:
    """Tests for KnowledgeBaseBuilder initialization."""
    
    @patch('build_knowledge_base.MistralClient')
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    def test_builder_initialization(self, mock_dp, mock_faiss, mock_mistral, config):
        """Test successful builder initialization."""
        builder = KnowledgeBaseBuilder(config)
        
        assert builder.config == config
        mock_mistral.assert_called_once_with(
            api_key=config.MISTRAL_API_KEY,
            model=config.MISTRAL_MODEL
        )
    
    @patch('build_knowledge_base.MistralClient')
    def test_builder_initialization_with_invalid_config(self, mock_mistral):
        """Test builder initialization fails with invalid config."""
        config = Mock()
        config.validate.side_effect = ConfigurationError("Invalid config")
        
        with pytest.raises(ConfigurationError):
            KnowledgeBaseBuilder(config)


class TestKnowledgeBaseBuilding:
    """Tests for knowledge base building process."""
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_build_with_valid_documents_directory(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        temp_documents_dir,
        mock_document_processor,
        mock_faiss_retriever,
        mock_mistral_client,
        config
    ):
        """Test building knowledge base with valid documents."""
        mock_dp_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_mistral.return_value = mock_mistral_client
        
        builder = KnowledgeBaseBuilder(config)
        result = builder.build_knowledge_base(
            documents_dir=temp_documents_dir,
            rebuild=True
        )
        
        assert result == 0
        mock_document_processor.load_documents.assert_called_once()
        mock_faiss_retriever.build_index.assert_called_once()
        mock_faiss_retriever.save_index.assert_called_once()
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_build_with_nonexistent_directory(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        config
    ):
        """Test building with nonexistent documents directory."""
        builder = KnowledgeBaseBuilder(config)
        result = builder.build_knowledge_base(
            documents_dir="/nonexistent/path"
        )
        
        assert result == 1
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_build_with_no_documents(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        mock_document_processor,
        config
    ):
        """Test building with empty documents directory."""
        mock_document_processor.load_documents.return_value = []
        mock_dp_class.return_value = mock_document_processor
        mock_faiss_class.return_value = Mock()
        mock_mistral.return_value = Mock()
        
        builder = KnowledgeBaseBuilder(config)
        result = builder.build_knowledge_base(
            documents_dir="data/documents/"
        )
        
        assert result == 1
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_build_update_existing_index(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        temp_documents_dir,
        mock_document_processor,
        mock_faiss_retriever,
        mock_mistral_client,
        config
    ):
        """Test updating existing FAISS index."""
        mock_faiss_retriever.is_empty = False
        mock_dp_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_mistral.return_value = mock_mistral_client
        
        builder = KnowledgeBaseBuilder(config)
        result = builder.build_knowledge_base(
            documents_dir=temp_documents_dir,
            rebuild=False
        )
        
        assert result == 0
        mock_faiss_retriever.add_documents.assert_called_once()
        mock_faiss_retriever.build_index.assert_not_called()


class TestDocumentProcessing:
    """Tests for document processing during knowledge base building."""
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_chunk_documents_properly(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        temp_documents_dir,
        mock_document_processor,
        mock_faiss_retriever,
        mock_mistral_client,
        config
    ):
        """Test that documents are properly chunked."""
        mock_dp_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_mistral.return_value = mock_mistral_client
        
        builder = KnowledgeBaseBuilder(config)
        builder.build_knowledge_base(
            documents_dir=temp_documents_dir,
            rebuild=True
        )
        
        # Verify chunking was called for each document
        assert mock_document_processor.chunk_document.call_count == 2
    
    @patch('build_knowledge_base.FAISSRetriever')
    @patch('build_knowledge_base.DocumentProcessor')
    @patch('build_knowledge_base.MistralClient')
    def test_skip_failed_document_chunks(
        self,
        mock_mistral,
        mock_dp_class,
        mock_faiss_class,
        mock_document_processor,
        mock_faiss_retriever,
        mock_mistral_client,
        config
    ):
        """Test that failed document chunks are skipped."""
        # Mock processor that fails on first document, succeeds on second
        mock_doc1 = Mock()
        mock_doc1.metadata.source = "fail.txt"
        
        mock_doc2 = Mock()
        mock_doc2.metadata.source = "success.txt"
        
        mock_document_processor.load_documents.return_value = [mock_doc1, mock_doc2]
        
        mock_chunk_doc2 = Mock()
        mock_chunk_doc2.content = "chunk content"
        
        mock_document_processor.chunk_document.side_effect = [
            Exception("Chunk error"),  # Fails on first
            [mock_chunk_doc2]  # Succeeds on second
        ]
        
        mock_dp_class.return_value = mock_document_processor
        mock_faiss_class.return_value = mock_faiss_retriever
        mock_mistral.return_value = mock_mistral_client
        
        # Should still complete, but skip the failed document
        with patch('build_knowledge_base.logger'):
            builder = KnowledgeBaseBuilder(config)
            result = builder.build_knowledge_base(
                documents_dir="data/documents/",
                rebuild=True,
                verbose=False
            )
        
        # Should have tried to build index despite one failure
        assert result == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
