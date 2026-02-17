"""Integration tests for FAISSRetriever with MistralClient."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from src.config import Config
from src.mistral_client import MistralClient
from src.faiss_retriever import FAISSRetriever
from src.document_processor import Document, DocumentMetadata


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(3):
        metadata = DocumentMetadata(
            source=f"integration_doc_{i}.txt",
            category="business",
            industry="technology"
        )
        doc = Document(
            content=f"Integration test document {i} about business and technology.",
            metadata=metadata
        )
        docs.append(doc)
    return docs


class TestFAISSRetrieverMistralIntegration:
    """Test FAISSRetriever integration with MistralClient."""
    
    @patch('src.mistral_client.Mistral')
    @patch.dict('os.environ', {
        'MISTRAL_API_KEY': 'test_integration_key_12345',
        'FAISS_INDEX_PATH': 'data/test_faiss_index'
    })
    def test_retriever_with_mistral_client_from_config(
        self,
        mock_mistral_class,
        sample_documents,
        temp_index_dir
    ):
        """Test FAISSRetriever works with MistralClient created from Config."""
        # Setup mock Mistral client
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        # Mock embedding responses
        import numpy as np
        
        def mock_embed_create(model, inputs):
            # Return mock embeddings
            embeddings = []
            for _ in inputs:
                vec = np.random.randn(1024).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                embeddings.append(Mock(embedding=vec.tolist()))
            
            response = Mock()
            response.data = embeddings
            return response
        
        mock_client.embeddings.create.side_effect = mock_embed_create
        
        # Create config and clients
        config = Config()
        config.MISTRAL_API_KEY = "test_integration_key_12345"
        
        mistral_client = MistralClient.from_config(config)
        
        retriever = FAISSRetriever(
            mistral_client=mistral_client,
            index_path=temp_index_dir,
            embedding_dimension=1024
        )
        
        # Build index
        retriever.build_index(sample_documents)
        
        # Verify index was built
        assert retriever.index is not None
        assert retriever.index.ntotal == len(sample_documents)
        
        # Verify embeddings were generated via Mistral client
        assert mock_client.embeddings.create.called
    
    @patch('src.mistral_client.Mistral')
    def test_retriever_search_with_mistral_embeddings(
        self,
        mock_mistral_class,
        sample_documents,
        temp_index_dir
    ):
        """Test search functionality with Mistral-generated embeddings."""
        # Setup mock Mistral client
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        import numpy as np
        
        # Store embeddings for consistency
        stored_embeddings = {}
        
        def mock_embed_create(model, inputs):
            embeddings = []
            for text in inputs:
                # Use consistent embeddings for same text
                if text not in stored_embeddings:
                    vec = np.random.randn(1024).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    stored_embeddings[text] = vec
                else:
                    vec = stored_embeddings[text]
                
                embeddings.append(Mock(embedding=vec.tolist()))
            
            response = Mock()
            response.data = embeddings
            return response
        
        mock_client.embeddings.create.side_effect = mock_embed_create
        
        # Create retriever
        mistral_client = MistralClient(api_key="test_key_12345")
        retriever = FAISSRetriever(
            mistral_client=mistral_client,
            index_path=temp_index_dir
        )
        
        # Build index
        retriever.build_index(sample_documents)
        
        # Perform search
        results = retriever.search("business technology", top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(0.0 <= score <= 1.0 for doc, score in results)
        
        # Verify search called embedding generation
        assert mock_client.embeddings.create.call_count >= 2  # Once for build, once for search
    
    @patch('src.mistral_client.Mistral')
    def test_retriever_persistence_integration(
        self,
        mock_mistral_class,
        sample_documents,
        temp_index_dir
    ):
        """Test save/load functionality with Mistral client."""
        # Setup mock Mistral client
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        import numpy as np
        
        def mock_embed_create(model, inputs):
            embeddings = []
            for _ in inputs:
                vec = np.random.randn(1024).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                embeddings.append(Mock(embedding=vec.tolist()))
            
            response = Mock()
            response.data = embeddings
            return response
        
        mock_client.embeddings.create.side_effect = mock_embed_create
        
        # Create first retriever and build index
        mistral_client1 = MistralClient(api_key="test_key_12345")
        retriever1 = FAISSRetriever(
            mistral_client=mistral_client1,
            index_path=temp_index_dir
        )
        retriever1.build_index(sample_documents)
        retriever1.save_index()
        
        # Create second retriever and load index
        mistral_client2 = MistralClient(api_key="test_key_12345")
        retriever2 = FAISSRetriever(
            mistral_client=mistral_client2,
            index_path=temp_index_dir
        )
        retriever2.load_index()
        
        # Verify loaded index
        assert retriever2.index is not None
        assert retriever2.index.ntotal == len(sample_documents)
        assert len(retriever2.documents) == len(sample_documents)
        
        # Verify documents are preserved
        for doc1, doc2 in zip(retriever1.documents, retriever2.documents):
            assert doc1.content == doc2.content
            assert doc1.metadata.source == doc2.metadata.source
