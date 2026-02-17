"""Document processing for knowledge base ingestion.

Handles loading, chunking, and metadata extraction for business documents.
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata for knowledge base documents.
    
    Stores information about document source, category, and other attributes
    that help with retrieval and filtering. Supports both traditional documents
    and CSV-based structured data.
    """
    
    source: str = Field(..., description="Document source/filename", min_length=1)
    category: str = Field("general", description="Document category", min_length=1)
    date: Optional[str] = Field(None, description="Document date (ISO format)")
    industry: Optional[str] = Field(None, description="Related industry")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    id: Optional[str] = Field(None, description="Unique document identifier (e.g., from CSV)")
    type: Optional[str] = Field(None, description="Document type (e.g., 'csv', 'txt', 'pdf')")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata fields")
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate date is in ISO format if provided."""
        if v is not None and v != "":
            try:
                # Try parsing as ISO format
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Date must be in ISO format (YYYY-MM-DD or ISO 8601), got: {v}")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure all tags are non-empty strings."""
        if v is not None:
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("All tags must be non-empty strings")
        return v


class Document(BaseModel):
    """Knowledge base document with content and metadata.
    
    Represents a document or document chunk with its text content,
    metadata, and optional embedding vector.
    """
    
    content: str = Field(..., description="Document text content", min_length=1)
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    chunk_id: Optional[int] = Field(None, description="Chunk identifier", ge=0)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Document content cannot be empty or whitespace only")
        return v
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimensions if provided."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty list")
            # Check all values are floats
            for val in v:
                if not isinstance(val, (int, float)):
                    raise ValueError("All embedding values must be numeric")
        return v
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Get embedding as numpy array.
        
        Returns:
            Embedding as numpy array, or None if no embedding
        """
        if self.embedding is None:
            return None
        return np.array(self.embedding, dtype=np.float32)
    
    def set_embedding_array(self, embedding: np.ndarray) -> None:
        """Set embedding from numpy array.
        
        Args:
            embedding: Numpy array of embedding values
        """
        self.embedding = embedding.tolist()
    
    model_config = {"arbitrary_types_allowed": True}




class DocumentProcessor:
    """Process documents for knowledge base ingestion.
    
    Handles loading documents from various formats (txt, pdf, json, md),
    extracting metadata, and preparing documents for indexing.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            
        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(
            f"Initialized DocumentProcessor: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from directory.
        
        Supports: .txt, .pdf, .json, .md, .csv files
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of loaded documents
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If directory is empty or contains no supported files
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        documents = []
        
        # Load traditional documents (.txt, .pdf, .json, .md)
        supported_extensions = {'.txt', '.pdf', '.json', '.md'}
        files = []
        for ext in supported_extensions:
            files.extend(dir_path.glob(f"**/*{ext}"))
        
        for file_path in files:
            try:
                doc = self._load_single_document(file_path)
                documents.append(doc)
                logger.debug(f"Loaded document: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {str(e)}")
        
        # Load CSV files using CSVDocumentLoader
        csv_files = list(dir_path.glob("*.csv"))
        if csv_files:
            try:
                from src.csv_loader import CSVDocumentLoader, CSVConfig
                csv_config = CSVConfig(data_dir=str(dir_path))
                csv_loader = CSVDocumentLoader(csv_config)
                csv_docs = csv_loader.load_all_csvs()
                documents.extend(csv_docs)
                logger.info(f"Loaded {len(csv_docs)} document(s) from {len(csv_files)} CSV file(s)")
            except ImportError:
                logger.warning("CSV loader not available, skipping CSV files")
            except Exception as e:
                logger.warning(f"Failed to load CSV files: {str(e)}")
        
        if not documents:
            raise ValueError(
                f"No supported files found in {directory}. "
                f"Supported formats: {', '.join(supported_extensions)}, .csv"
            )
        
        logger.info(f"Found {len(files)} traditional document(s) and {len(csv_files)} CSV file(s)")
        logger.info(f"Successfully loaded {len(documents)} document(s) total")
        return documents
    
    def _load_single_document(self, file_path: Path) -> Document:
        """Load a single document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Loaded document
            
        Raises:
            ValueError: If file format is unsupported or content is invalid
        """
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            content = self._load_text_file(file_path)
        elif extension == '.md':
            content = self._load_text_file(file_path)
        elif extension == '.json':
            content = self._load_json_file(file_path)
        elif extension == '.pdf':
            content = self._load_pdf_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Extract metadata
        metadata = self.extract_metadata(str(file_path))
        
        # Create document
        return Document(content=content, metadata=metadata)
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load text or markdown file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content.strip()
    
    def _load_json_file(self, file_path: Path) -> str:
        """Load JSON file and convert to text.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            JSON content as formatted string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to readable text format
        if isinstance(data, dict):
            # If JSON has a 'content' or 'text' field, use that
            if 'content' in data:
                return str(data['content']).strip()
            elif 'text' in data:
                return str(data['text']).strip()
            else:
                # Otherwise, format the entire JSON
                return json.dumps(data, indent=2)
        elif isinstance(data, list):
            # Join list items
            return '\n\n'.join(str(item) for item in data)
        else:
            return str(data).strip()
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF file and extract text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            ImportError: If PyPDF2 is not installed
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF processing. "
                "Install it with: pip install PyPDF2"
            )
        
        reader = PdfReader(str(file_path))
        
        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text.strip())
        
        return '\n\n'.join(text_parts)
    
    def extract_metadata(self, filepath: str) -> DocumentMetadata:
        """Extract metadata from file path and attributes.
        
        Args:
            filepath: Path to file
            
        Returns:
            Extracted metadata
        """
        file_path = Path(filepath)
        
        # Extract basic metadata
        source = file_path.name
        
        # Try to infer category from directory structure
        # If file is in a subdirectory, use that as category
        if len(file_path.parts) > 1:
            category = file_path.parts[-2]  # Parent directory name
        else:
            category = "general"
        
        # Get file modification date
        try:
            mtime = os.path.getmtime(filepath)
            date = datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            date = None
        
        # Try to infer industry from filename or path
        industry = None
        filepath_lower = filepath.lower()
        industries = ['technology', 'healthcare', 'finance', 'retail', 'education', 
                     'manufacturing', 'real estate', 'food', 'entertainment']
        for ind in industries:
            if ind in filepath_lower:
                industry = ind
                break
        
        # Extract tags from filename (words separated by - or _)
        filename_without_ext = file_path.stem
        tags = []
        for separator in ['-', '_']:
            if separator in filename_without_ext:
                tags = [tag.strip() for tag in filename_without_ext.split(separator)]
                break
        
        return DocumentMetadata(
            source=source,
            category=category,
            date=date,
            industry=industry,
            tags=tags if tags else []
        )

    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks with overlap.
        
        Chunks documents to respect token limits while preserving context
        through overlapping content between consecutive chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks, each with preserved metadata
            
        Raises:
            ValueError: If document content is empty
        """
        content = document.content.strip()
        
        if not content:
            raise ValueError("Cannot chunk document with empty content")
        
        # Tokenize content (simple word-based tokenization)
        # In production, use a proper tokenizer like tiktoken
        tokens = self._tokenize(content)
        
        if len(tokens) <= self.chunk_size:
            # Document fits in single chunk
            logger.debug(
                f"Document fits in single chunk: {len(tokens)} tokens "
                f"<= {self.chunk_size} chunk_size"
            )
            chunk = Document(
                content=content,
                metadata=document.metadata.model_copy(deep=True),
                chunk_id=0
            )
            return [chunk]
        
        # Split into overlapping chunks
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Convert tokens back to text
            chunk_text = self._detokenize(chunk_tokens)
            
            # Create chunk document with preserved metadata
            chunk = Document(
                content=chunk_text,
                metadata=document.metadata.model_copy(deep=True),
                chunk_id=chunk_id
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            # If this is the last chunk, we're done
            if end_idx >= len(tokens):
                break
            
            # Move forward by (chunk_size - overlap) tokens
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_id += 1
        
        logger.debug(
            f"Split document into {len(chunks)} chunks: "
            f"original_tokens={len(tokens)}, chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Simple word-based tokenization. In production, use a proper
        tokenizer like tiktoken that matches the LLM's tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (words)
        """
        # Split on whitespace and punctuation
        import re
        # Keep words and punctuation as separate tokens
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed text
        """
        # Simple reconstruction: join with spaces, handle punctuation
        text = ""
        for i, token in enumerate(tokens):
            if i == 0:
                text = token
            elif token in {'.', ',', '!', '?', ':', ';', ')', ']', '}'}:
                # No space before punctuation
                text += token
            elif i > 0 and tokens[i-1] in {'(', '[', '{'}:
                # No space after opening brackets
                text += token
            else:
                text += " " + token
        
        return text
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    f"Failed to chunk document from {doc.metadata.source}: {str(e)}"
                )
                # Continue with other documents
        
        logger.info(
            f"Chunked {len(documents)} document(s) into {len(all_chunks)} chunk(s)"
        )
        
        return all_chunks
