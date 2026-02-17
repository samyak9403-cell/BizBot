"""Unit tests for FAISSRetriever."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import faiss

from src.faiss_retriever import FAISSRetriever, FAISSRetrieverError
from src.mistral_client import MistralClient
from src.document_processor import Document, DocumentMetadata


@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client."""
    client = Mock(spec=MistralClient)
    
    # Mock embed method to return normalized vectors
    def mock_embed(texts):
        if isinstance(texts, str):
            # Single text - return 1D array
            vec = np.random.randn(1024).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec
        else:
            # Multiple texts - return 2D array
            vecs = np.random.randn(len(texts), 1024).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
            return vecs
    
    client.embed.side_effect = mock_embed
    return client


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(5):
        metadata = DocumentMetadata(
            source=f"doc_{i}.txt",
            category="business",
            industry="technology" if i % 2 == 0 else "finance"
        )
        doc = Document(
            content=f"This is document {i} about business ideas.",
            metadata=metadata
        )
        docs.append(doc)
    return docs


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestFAISSRetrieverInitialization:
    """Test FAISSRetriever initialization."""
    
    def test_init_with_valid_parameters(self, mock_mistral_client, temp_index_dir):
        """Test initialization with valid parameters."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir,
            embedding_dimension=1024
        )
        
        assert retriever.mistral_client == mock_mistral_client
        assert retriever.index_path == Path(temp_index_dir)
        assert retriever.embedding_dimension == 1024
        assert retriever.index is None
        assert retriever.documents == []
    
    def test_init_with_invalid_dimension_raises_error(self, mock_mistral_client):
        """Test initialization with invalid embedding dimension."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            FAISSRetriever(
                mistral_client=mock_mistral_client,
                index_path="data/test",
                embedding_dimension=0
            )
    
    def test_init_creates_index_directory(self, mock_mistral_client, temp_index_dir):
        """Test that initialization creates index directory if it doesn't exist."""
        index_path = Path(temp_index_dir) / "new_index"
        assert not index_path.exists()
        
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=str(index_path)
        )
        
        assert index_path.exists()
        assert index_path.is_dir()


class TestFAISSRetrieverBuildIndex:
    """Test index building functionality."""
    
    def test_build_index_with_documents(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test building index from documents."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        assert retriever.index is not None
        assert retriever.index.ntotal == len(sample_documents)
        assert len(retriever.documents) == len(sample_documents)
        
        # Verify embeddings were generated
        mock_mistral_client.embed.assert_called_once()
        
        # Verify documents have embeddings
        for doc in retriever.documents:
            assert doc.embedding is not None
            assert len(doc.embedding) == 1024
    
    def test_build_index_with_empty_list_raises_error(self, mock_mistral_client, temp_index_dir):
        """Test building index with empty document list raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        with pytest.raises(ValueError, match="Cannot build index from empty document list"):
            retriever.build_index([])
    
    def test_build_index_preserves_metadata(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test that building index preserves document metadata."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        for original, stored in zip(sample_documents, retriever.documents):
            assert stored.content == original.content
            assert stored.metadata.source == original.metadata.source
            assert stored.metadata.category == original.metadata.category
            assert stored.metadata.industry == original.metadata.industry


class TestFAISSRetrieverAddDocuments:
    """Test adding documents to existing index."""
    
    def test_add_documents_to_existing_index(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test adding documents to an existing index."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        # Build initial index
        initial_docs = sample_documents[:3]
        retriever.build_index(initial_docs)
        initial_size = retriever.index.ntotal
        
        # Add more documents
        new_docs = sample_documents[3:]
        retriever.add_documents(new_docs)
        
        assert retriever.index.ntotal == initial_size + len(new_docs)
        assert len(retriever.documents) == len(sample_documents)
    
    def test_add_documents_without_index_raises_error(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test adding documents without building index first raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        with pytest.raises(FAISSRetrieverError, match="Index not initialized"):
            retriever.add_documents(sample_documents)
    
    def test_add_empty_document_list_raises_error(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test adding empty document list raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        with pytest.raises(ValueError, match="Cannot add empty document list"):
            retriever.add_documents([])


class TestFAISSRetrieverSearch:
    """Test semantic search functionality."""
    
    def test_search_returns_top_k_results(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search returns exactly K results."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        results = retriever.search("business ideas", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
    
    def test_search_returns_all_when_k_exceeds_index_size(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search returns all documents when K exceeds index size."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        results = retriever.search("business ideas", top_k=100)
        
        assert len(results) == len(sample_documents)
    
    def test_search_results_ordered_by_similarity(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search results are ordered by similarity score (descending)."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        results = retriever.search("business ideas", top_k=5)
        
        scores = [score for doc, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_similarity_scores_in_valid_range(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test similarity scores are in range [0, 1]."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        results = retriever.search("business ideas", top_k=5)
        
        for doc, score in results:
            assert 0.0 <= score <= 1.0
    
    def test_search_with_empty_query_raises_error(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search with empty query raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.search("", top_k=3)
    
    def test_search_with_invalid_top_k_raises_error(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search with invalid top_k raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.search("business ideas", top_k=0)
    
    def test_search_without_index_raises_error(self, mock_mistral_client, temp_index_dir):
        """Test search without building index raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        with pytest.raises(FAISSRetrieverError, match="Index is empty"):
            retriever.search("business ideas", top_k=3)
    
    def test_search_with_metadata_filter(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test search with metadata filtering."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        
        # Filter by industry
        results = retriever.search(
            "business ideas",
            top_k=5,
            filter_metadata={"industry": "technology"}
        )
        
        # All results should have technology industry
        for doc, score in results:
            assert doc.metadata.industry == "technology"


class TestFAISSRetrieverPersistence:
    """Test index persistence (save/load)."""
    
    def test_save_index(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test saving index to disk."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        retriever.build_index(sample_documents)
        retriever.save_index()
        
        # Check files exist
        index_file = Path(temp_index_dir) / "faiss.index"
        metadata_file = Path(temp_index_dir) / "metadata.pkl"
        
        assert index_file.exists()
        assert metadata_file.exists()
    
    def test_save_without_index_raises_error(self, mock_mistral_client, temp_index_dir):
        """Test saving without building index raises error."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        with pytest.raises(FAISSRetrieverError, match="No index to save"):
            retriever.save_index()
    
    def test_load_index(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test loading index from disk."""
        # Build and save index
        retriever1 = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        retriever1.build_index(sample_documents)
        retriever1.save_index()
        
        # Load index in new retriever
        retriever2 = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        retriever2.load_index()
        
        assert retriever2.index is not None
        assert retriever2.index.ntotal == len(sample_documents)
        assert len(retriever2.documents) == len(sample_documents)
    
    def test_load_nonexistent_index_raises_error(self, mock_mistral_client, temp_index_dir):
        """Test loading nonexistent index raises FileNotFoundError."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        with pytest.raises(FileNotFoundError, match="Index file not found"):
            retriever.load_index()
    
    def test_save_load_round_trip_preserves_search_results(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test that save/load round trip preserves search functionality."""
        # Build and save index
        retriever1 = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        retriever1.build_index(sample_documents)
        
        # Save index
        retriever1.save_index()
        
        # Load index in new retriever
        retriever2 = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        retriever2.load_index()
        
        # Verify index properties are preserved
        assert retriever2.index.ntotal == retriever1.index.ntotal
        assert len(retriever2.documents) == len(retriever1.documents)
        
        # Verify documents are preserved
        for doc1, doc2 in zip(retriever1.documents, retriever2.documents):
            assert doc1.content == doc2.content
            assert doc1.metadata.source == doc2.metadata.source
            # Verify embeddings are preserved
            assert doc1.embedding == doc2.embedding


class TestFAISSRetrieverProperties:
    """Test retriever properties."""
    
    def test_is_empty_property(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test is_empty property."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        assert retriever.is_empty is True
        
        retriever.build_index(sample_documents)
        
        assert retriever.is_empty is False
    
    def test_size_property(self, mock_mistral_client, sample_documents, temp_index_dir):
        """Test size property."""
        retriever = FAISSRetriever(
            mistral_client=mock_mistral_client,
            index_path=temp_index_dir
        )
        
        assert retriever.size == 0
        
        retriever.build_index(sample_documents)
        
        assert retriever.size == len(sample_documents)
