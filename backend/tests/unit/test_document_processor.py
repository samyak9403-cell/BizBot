"""Unit tests for document processing."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from src.document_processor import (
    Document,
    DocumentMetadata,
    DocumentProcessor
)


class TestDocumentMetadata:
    """Test DocumentMetadata validation."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = DocumentMetadata(
            source="test.txt",
            category="business",
            date="2024-01-15",
            industry="technology",
            tags=["startup", "saas"]
        )
        
        assert metadata.source == "test.txt"
        assert metadata.category == "business"
        assert metadata.date == "2024-01-15"
        assert metadata.industry == "technology"
        assert metadata.tags == ["startup", "saas"]
    
    def test_minimal_metadata(self):
        """Test metadata with only required fields."""
        metadata = DocumentMetadata(
            source="test.txt",
            category="general"
        )
        
        assert metadata.source == "test.txt"
        assert metadata.category == "general"
        assert metadata.date is None
        assert metadata.industry is None
        assert metadata.tags == []
    
    def test_empty_source_fails(self):
        """Test that empty source is rejected."""
        with pytest.raises(ValueError):
            DocumentMetadata(source="", category="test")
    
    def test_empty_category_fails(self):
        """Test that empty category is rejected."""
        with pytest.raises(ValueError):
            DocumentMetadata(source="test.txt", category="")
    
    def test_invalid_date_format_fails(self):
        """Test that invalid date format is rejected."""
        with pytest.raises(ValueError, match="Date must be in ISO format"):
            DocumentMetadata(
                source="test.txt",
                category="test",
                date="01/15/2024"  # Wrong format
            )
    
    def test_valid_iso_date_formats(self):
        """Test various valid ISO date formats."""
        # Date only
        metadata1 = DocumentMetadata(
            source="test.txt",
            category="test",
            date="2024-01-15"
        )
        assert metadata1.date == "2024-01-15"
        
        # Full ISO 8601
        metadata2 = DocumentMetadata(
            source="test.txt",
            category="test",
            date="2024-01-15T10:30:00"
        )
        assert metadata2.date == "2024-01-15T10:30:00"
    
    def test_empty_tag_fails(self):
        """Test that empty tags are rejected."""
        with pytest.raises(ValueError, match="non-empty strings"):
            DocumentMetadata(
                source="test.txt",
                category="test",
                tags=["valid", ""]
            )


class TestDocument:
    """Test Document validation."""
    
    def test_valid_document(self):
        """Test creating valid document."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(
            content="This is test content.",
            metadata=metadata
        )
        
        assert doc.content == "This is test content."
        assert doc.metadata.source == "test.txt"
        assert doc.embedding is None
        assert doc.chunk_id is None
    
    def test_document_with_embedding(self):
        """Test document with embedding."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        embedding = [0.1, 0.2, 0.3, 0.4]
        
        doc = Document(
            content="Test content",
            metadata=metadata,
            embedding=embedding
        )
        
        assert doc.embedding == embedding
    
    def test_document_with_chunk_id(self):
        """Test document with chunk ID."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(
            content="Test content",
            metadata=metadata,
            chunk_id=5
        )
        
        assert doc.chunk_id == 5
    
    def test_empty_content_fails(self):
        """Test that empty content is rejected."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        with pytest.raises(ValueError):
            Document(content="", metadata=metadata)
    
    def test_whitespace_only_content_fails(self):
        """Test that whitespace-only content is rejected."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            Document(content="   \n\t  ", metadata=metadata)
    
    def test_negative_chunk_id_fails(self):
        """Test that negative chunk ID is rejected."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        with pytest.raises(ValueError):
            Document(
                content="Test",
                metadata=metadata,
                chunk_id=-1
            )
    
    def test_empty_embedding_fails(self):
        """Test that empty embedding list is rejected."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            Document(
                content="Test",
                metadata=metadata,
                embedding=[]
            )
    
    def test_invalid_embedding_values_fail(self):
        """Test that non-numeric embedding values are rejected."""
        from pydantic import ValidationError
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        with pytest.raises(ValidationError):
            Document(
                content="Test",
                metadata=metadata,
                embedding=[0.1, "invalid", 0.3]
            )
    
    def test_get_embedding_array(self):
        """Test converting embedding to numpy array."""
        import numpy as np
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        embedding = [0.1, 0.2, 0.3]
        
        doc = Document(
            content="Test",
            metadata=metadata,
            embedding=embedding
        )
        
        arr = doc.get_embedding_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert list(arr) == embedding
    
    def test_get_embedding_array_none(self):
        """Test getting embedding array when no embedding."""
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(content="Test", metadata=metadata)
        
        assert doc.get_embedding_array() is None
    
    def test_set_embedding_array(self):
        """Test setting embedding from numpy array."""
        import numpy as np
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(content="Test", metadata=metadata)
        
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        doc.set_embedding_array(arr)
        
        # Check that embedding is set (with float32 precision)
        assert doc.embedding is not None
        assert len(doc.embedding) == 3
        assert all(isinstance(x, float) for x in doc.embedding)


class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50
    
    def test_initialization_defaults(self):
        """Test processor with default values."""
        processor = DocumentProcessor()
        
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50
    
    def test_invalid_chunk_size_fails(self):
        """Test that invalid chunk size is rejected."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentProcessor(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentProcessor(chunk_size=-10)
    
    def test_invalid_chunk_overlap_fails(self):
        """Test that invalid chunk overlap is rejected."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            DocumentProcessor(chunk_overlap=-5)
    
    def test_overlap_exceeds_size_fails(self):
        """Test that overlap >= size is rejected."""
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            DocumentProcessor(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            DocumentProcessor(chunk_size=100, chunk_overlap=150)
    
    def test_extract_metadata_basic(self):
        """Test basic metadata extraction."""
        processor = DocumentProcessor()
        
        # Create a temporary file path (doesn't need to exist for this test)
        metadata = processor.extract_metadata("documents/business/startup-guide.txt")
        
        assert metadata.source == "startup-guide.txt"
        assert metadata.category == "business"
        assert metadata.tags == ["startup", "guide"]
    
    def test_extract_metadata_with_industry(self):
        """Test metadata extraction with industry inference."""
        processor = DocumentProcessor()
        
        metadata = processor.extract_metadata("docs/technology/ai-trends.md")
        
        assert metadata.source == "ai-trends.md"
        assert metadata.industry == "technology"
    
    def test_tokenize(self):
        """Test text tokenization."""
        processor = DocumentProcessor()
        
        text = "Hello, world! This is a test."
        tokens = processor._tokenize(text)
        
        assert "Hello" in tokens
        assert "world" in tokens
        assert "," in tokens
        assert "!" in tokens
    
    def test_detokenize(self):
        """Test token reconstruction."""
        processor = DocumentProcessor()
        
        tokens = ["Hello", ",", "world", "!", "This", "is", "a", "test", "."]
        text = processor._detokenize(tokens)
        
        # Should reconstruct with proper spacing
        assert "Hello," in text
        assert "world!" in text
        assert "test." in text
    
    def test_chunk_document_small(self):
        """Test chunking document that fits in one chunk."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(
            content="This is a small document that fits in one chunk.",
            metadata=metadata
        )
        
        chunks = processor.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].content == doc.content
        assert chunks[0].chunk_id == 0
        assert chunks[0].metadata.source == "test.txt"
    
    def test_chunk_document_large(self):
        """Test chunking large document into multiple chunks."""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
        
        # Create a document with many words
        content = " ".join([f"word{i}" for i in range(50)])
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(content=content, metadata=metadata)
        
        chunks = processor.chunk_document(doc)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have an ID
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i
            assert chunk.metadata.source == "test.txt"
    
    def test_chunk_document_overlap(self):
        """Test that consecutive chunks have overlap."""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=3)
        
        # Create content with identifiable words
        words = [f"word{i:02d}" for i in range(30)]
        content = " ".join(words)
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        doc = Document(content=content, metadata=metadata)
        
        chunks = processor.chunk_document(doc)
        
        # Check that consecutive chunks share some content
        if len(chunks) > 1:
            # Get tokens from first two chunks
            tokens1 = processor._tokenize(chunks[0].content)
            tokens2 = processor._tokenize(chunks[1].content)
            
            # Last few tokens of chunk 0 should appear in chunk 1
            # (accounting for overlap)
            overlap_tokens = tokens1[-(processor.chunk_overlap):]
            
            # Check if any overlap tokens appear in chunk 2
            has_overlap = any(token in tokens2 for token in overlap_tokens)
            assert has_overlap, "Consecutive chunks should have overlapping content"
    
    def test_chunk_document_empty_fails(self):
        """Test that chunking empty document fails."""
        from pydantic import ValidationError
        
        processor = DocumentProcessor()
        
        metadata = DocumentMetadata(source="test.txt", category="test")
        
        # Document creation itself will fail with whitespace-only content
        with pytest.raises(ValidationError):
            doc = Document(content="   ", metadata=metadata)
            processor.chunk_document(doc)
    
    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents."""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
        
        metadata1 = DocumentMetadata(source="doc1.txt", category="test")
        metadata2 = DocumentMetadata(source="doc2.txt", category="test")
        
        doc1 = Document(content="Short doc", metadata=metadata1)
        doc2 = Document(
            content=" ".join([f"word{i}" for i in range(30)]),
            metadata=metadata2
        )
        
        chunks = processor.chunk_documents([doc1, doc2])
        
        # Should have chunks from both documents
        assert len(chunks) > 1
        
        # Check that both sources appear
        sources = {chunk.metadata.source for chunk in chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources


class TestDocumentLoading:
    """Test document loading from files."""
    
    def test_load_documents_directory_not_found(self):
        """Test loading from non-existent directory."""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_documents("nonexistent_directory")
    
    def test_load_documents_not_a_directory(self, tmp_path):
        """Test loading from a file instead of directory."""
        # Create a file
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="not a directory"):
            processor.load_documents(str(file_path))
    
    def test_load_documents_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="No supported files found"):
            processor.load_documents(str(tmp_path))
    
    def test_load_text_file(self, tmp_path):
        """Test loading text file."""
        # Create a text file
        file_path = tmp_path / "test.txt"
        file_path.write_text("This is test content.")
        
        processor = DocumentProcessor()
        documents = processor.load_documents(str(tmp_path))
        
        assert len(documents) == 1
        assert documents[0].content == "This is test content."
        assert documents[0].metadata.source == "test.txt"
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading markdown file."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Heading\n\nThis is markdown content.")
        
        processor = DocumentProcessor()
        documents = processor.load_documents(str(tmp_path))
        
        assert len(documents) == 1
        assert "# Heading" in documents[0].content
        assert documents[0].metadata.source == "test.md"
    
    def test_load_json_file_with_content(self, tmp_path):
        """Test loading JSON file with content field."""
        file_path = tmp_path / "test.json"
        data = {"content": "This is JSON content", "other": "data"}
        file_path.write_text(json.dumps(data))
        
        processor = DocumentProcessor()
        documents = processor.load_documents(str(tmp_path))
        
        assert len(documents) == 1
        assert documents[0].content == "This is JSON content"
    
    def test_load_json_file_without_content(self, tmp_path):
        """Test loading JSON file without content field."""
        file_path = tmp_path / "test.json"
        data = {"title": "Test", "description": "A test document"}
        file_path.write_text(json.dumps(data))
        
        processor = DocumentProcessor()
        documents = processor.load_documents(str(tmp_path))
        
        assert len(documents) == 1
        # Should contain the JSON structure
        assert "title" in documents[0].content
        assert "Test" in documents[0].content
    
    def test_load_multiple_files(self, tmp_path):
        """Test loading multiple files."""
        # Create multiple files
        (tmp_path / "doc1.txt").write_text("Document 1")
        (tmp_path / "doc2.md").write_text("Document 2")
        (tmp_path / "doc3.json").write_text('{"content": "Document 3"}')
        
        processor = DocumentProcessor()
        documents = processor.load_documents(str(tmp_path))
        
        assert len(documents) == 3
        
        # Check all documents loaded
        sources = {doc.metadata.source for doc in documents}
        assert "doc1.txt" in sources
        assert "doc2.md" in sources
        assert "doc3.json" in sources
