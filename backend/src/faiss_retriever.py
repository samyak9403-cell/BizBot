"""FAISS-based semantic search retriever for knowledge base.

Manages FAISS vector index for efficient similarity search over document embeddings.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss

from .mistral_client import MistralClient
from .document_processor import Document


logger = logging.getLogger(__name__)


class FAISSRetrieverError(Exception):
    """Raised when FAISS retriever operations fail."""
    pass


class FAISSRetriever:
    """FAISS-based retriever for semantic search over documents.
    
    Manages a FAISS index for efficient similarity search, handles document
    embedding generation, and provides persistence for the index and metadata.
    """
    
    def __init__(
        self,
        mistral_client: MistralClient,
        index_path: str = "data/faiss_index",
        embedding_dimension: int = 1024
    ):
        """Initialize FAISS retriever with embedding model.
        
        Args:
            mistral_client: Mistral client for generating embeddings
            index_path: Directory path for storing index files
            embedding_dimension: Dimension of embedding vectors
            
        Raises:
            ValueError: If embedding_dimension is invalid
        """
        if embedding_dimension <= 0:
            raise ValueError(f"embedding_dimension must be positive, got {embedding_dimension}")
        
        self.mistral_client = mistral_client
        self.index_path = Path(index_path)
        self.embedding_dimension = embedding_dimension
        
        # Initialize FAISS index (using L2 distance for cosine similarity)
        # We'll normalize vectors, so L2 distance = 2(1 - cosine_similarity)
        self.index: Optional[faiss.IndexFlatL2] = None
        
        # Store document metadata separately (FAISS only stores vectors)
        self.documents: List[Document] = []
        
        # Create index directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized FAISSRetriever: index_path={index_path}, "
            f"embedding_dimension={embedding_dimension}"
        )
    
    def _create_index(self) -> faiss.IndexFlatL2:
        """Create a new FAISS index.
        
        Returns:
            New FAISS index
        """
        index = faiss.IndexFlatL2(self.embedding_dimension)
        logger.debug(f"Created new FAISS index with dimension {self.embedding_dimension}")
        return index
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using Mistral.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
            
        Raises:
            FAISSRetrieverError: If embedding generation fails
        """
        try:
            embedding = self.mistral_client.embed(text)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise FAISSRetrieverError(f"Embedding generation failed: {str(e)}") from e
    
    def _embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for multiple documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Array of normalized embeddings, shape (num_docs, embedding_dim)
            
        Raises:
            FAISSRetrieverError: If embedding generation fails
        """
        if not documents:
            raise ValueError("Cannot embed empty document list")
        
        try:
            # Extract text content
            texts = [doc.content for doc in documents]
            
            # Process embeddings in batches to avoid token limit issues
            batch_size = 8  # Process 8 documents at a time (very conservative for large docs)
            all_embeddings = []
            
            logger.debug(f"Generating embeddings for {len(texts)} documents (batch size: {batch_size})")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch_texts)} texts)")
                
                try:
                    batch_embeddings = self.mistral_client.embed(batch_texts)
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to embed batch {i // batch_size + 1}: {str(e)}")
                    raise
            
            # Combine all batches
            embeddings = np.vstack(all_embeddings)
            
            # Normalize each embedding for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            
            logger.info(f"Successfully generated embeddings for {len(embeddings)} documents")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {str(e)}")
            raise FAISSRetrieverError(f"Document embedding failed: {str(e)}") from e


    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS index from documents.
        
        Generates embeddings for all documents and adds them to the index.
        Stores document metadata separately for retrieval.
        
        Args:
            documents: List of business documents to index
            
        Raises:
            ValueError: If documents list is empty
            FAISSRetrieverError: If index building fails
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list")
        
        logger.info(f"Building FAISS index from {len(documents)} documents")
        
        try:
            # Create new index
            self.index = self._create_index()
            
            # Generate embeddings for all documents
            embeddings = self._embed_documents(documents)
            
            # Verify embedding dimensions
            if embeddings.shape[1] != self.embedding_dimension:
                raise FAISSRetrieverError(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {embeddings.shape[1]}"
                )
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store documents with their embeddings
            self.documents = []
            for i, doc in enumerate(documents):
                # Create a copy and add embedding
                doc_copy = doc.model_copy(deep=True)
                doc_copy.set_embedding_array(embeddings[i])
                self.documents.append(doc_copy)
            
            logger.info(
                f"Successfully built FAISS index: {len(documents)} documents, "
                f"index size: {self.index.ntotal}"
            )
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}", exc_info=True)
            raise FAISSRetrieverError(f"Index building failed: {str(e)}") from e
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing index.
        
        Args:
            documents: List of documents to add
            
        Raises:
            ValueError: If documents list is empty
            FAISSRetrieverError: If index doesn't exist or adding fails
        """
        if not documents:
            raise ValueError("Cannot add empty document list")
        
        if self.index is None:
            raise FAISSRetrieverError(
                "Index not initialized. Call build_index() first or load_index()"
            )
        
        logger.info(f"Adding {len(documents)} documents to existing index")
        
        try:
            # Generate embeddings for new documents
            embeddings = self._embed_documents(documents)
            
            # Verify embedding dimensions
            if embeddings.shape[1] != self.embedding_dimension:
                raise FAISSRetrieverError(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {embeddings.shape[1]}"
                )
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store documents with their embeddings
            for i, doc in enumerate(documents):
                doc_copy = doc.model_copy(deep=True)
                doc_copy.set_embedding_array(embeddings[i])
                self.documents.append(doc_copy)
            
            logger.info(
                f"Successfully added {len(documents)} documents. "
                f"Total index size: {self.index.ntotal}"
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {str(e)}", exc_info=True)
            raise FAISSRetrieverError(f"Adding documents failed: {str(e)}") from e


    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for relevant documents.
        
        Performs semantic search by converting the query to an embedding
        and finding the most similar documents in the index.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (not implemented yet)
            
        Returns:
            List of (document, similarity_score) tuples, ordered by relevance
            Similarity scores are cosine similarities in range [0, 1]
            
        Raises:
            ValueError: If query is empty or top_k is invalid
            FAISSRetrieverError: If index doesn't exist or search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if self.index is None or self.index.ntotal == 0:
            raise FAISSRetrieverError(
                "Index is empty. Call build_index() first or load_index()"
            )
        
        # Limit top_k to available documents
        actual_k = min(top_k, self.index.ntotal)
        
        logger.debug(f"Searching for top {actual_k} documents matching query")
        
        try:
            # Convert query to embedding
            query_embedding = self._embed_text(query)
            
            # Reshape for FAISS (needs 2D array)
            query_vector = query_embedding.reshape(1, -1)
            
            # Search index
            # distances are L2 distances, we'll convert to cosine similarity
            distances, indices = self.index.search(query_vector, actual_k)
            
            # Convert L2 distances to cosine similarities
            # Since vectors are normalized: L2_dist^2 = 2(1 - cosine_sim)
            # Therefore: cosine_sim = 1 - (L2_dist^2 / 2)
            similarities = 1 - (distances[0] ** 2 / 2)
            
            # Clip to [0, 1] range (numerical errors might cause slight violations)
            similarities = np.clip(similarities, 0.0, 1.0)
            
            # Build results list
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append((doc, float(similarity)))
            
            # Apply metadata filters if provided
            if filter_metadata:
                results = self._apply_metadata_filters(results, filter_metadata)
            
            logger.info(
                f"Search completed: found {len(results)} results, "
                f"top score: {results[0][1]:.4f}" if results else "no results"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise FAISSRetrieverError(f"Search failed: {str(e)}") from e
    
    def _apply_metadata_filters(
        self,
        results: List[Tuple[Document, float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """Apply metadata filters to search results.
        
        Args:
            results: List of (document, score) tuples
            filters: Dictionary of metadata field -> value filters
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for doc, score in results:
            match = True
            
            for field, value in filters.items():
                # Get metadata field value
                metadata_value = getattr(doc.metadata, field, None)
                
                # Check if it matches the filter
                if metadata_value != value:
                    match = False
                    break
            
            if match:
                filtered.append((doc, score))
        
        logger.debug(
            f"Applied metadata filters: {len(results)} -> {len(filtered)} results"
        )
        
        return filtered


    def save_index(self, path: Optional[str] = None) -> None:
        """Persist FAISS index to disk.
        
        Saves both the FAISS index and document metadata to separate files.
        
        Args:
            path: Optional custom path for saving. If None, uses self.index_path
            
        Raises:
            FAISSRetrieverError: If index doesn't exist or saving fails
        """
        if self.index is None:
            raise FAISSRetrieverError("No index to save. Build or load an index first.")
        
        save_path = Path(path) if path else self.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        index_file = save_path / "faiss.index"
        metadata_file = save_path / "metadata.pkl"
        
        logger.info(f"Saving FAISS index to {save_path}")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_file))
            logger.debug(f"Saved FAISS index to {index_file}")
            
            # Save document metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.debug(f"Saved metadata for {len(self.documents)} documents to {metadata_file}")
            
            logger.info(
                f"Successfully saved index: {self.index.ntotal} vectors, "
                f"{len(self.documents)} documents"
            )
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}", exc_info=True)
            raise FAISSRetrieverError(f"Index save failed: {str(e)}") from e
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load FAISS index from disk.
        
        Loads both the FAISS index and document metadata from files.
        Handles corrupted index files by raising appropriate errors.
        
        Args:
            path: Optional custom path for loading. If None, uses self.index_path
            
        Raises:
            FileNotFoundError: If index files don't exist
            FAISSRetrieverError: If index is corrupted or loading fails
        """
        load_path = Path(path) if path else self.index_path
        index_file = load_path / "faiss.index"
        metadata_file = load_path / "metadata.pkl"
        
        # Check if files exist
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_file}. "
                f"Build an index first with build_index()"
            )
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}. "
                f"Index may be corrupted."
            )
        
        logger.info(f"Loading FAISS index from {load_path}")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            logger.debug(f"Loaded FAISS index from {index_file}")
            
            # Verify index dimension matches configuration
            if self.index.d != self.embedding_dimension:
                raise FAISSRetrieverError(
                    f"Index dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {self.index.d}. Index may be corrupted or from different configuration."
                )
            
            # Load document metadata
            with open(metadata_file, 'rb') as f:
                self.documents = pickle.load(f)
            logger.debug(f"Loaded metadata for {len(self.documents)} documents")
            
            # Verify consistency between index and metadata
            if self.index.ntotal != len(self.documents):
                logger.warning(
                    f"Index size mismatch: FAISS index has {self.index.ntotal} vectors "
                    f"but metadata has {len(self.documents)} documents. "
                    f"Index may be corrupted."
                )
                raise FAISSRetrieverError(
                    "Index corrupted: mismatch between index size and metadata count"
                )
            
            logger.info(
                f"Successfully loaded index: {self.index.ntotal} vectors, "
                f"{len(self.documents)} documents"
            )
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except FAISSRetrieverError:
            # Re-raise our custom errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}", exc_info=True)
            
            # Check if it's a corruption issue
            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                raise FAISSRetrieverError(
                    f"Index file is corrupted: {str(e)}. "
                    f"Rebuild the index with build_index()"
                ) from e
            else:
                raise FAISSRetrieverError(f"Index load failed: {str(e)}") from e
    
    @property
    def is_empty(self) -> bool:
        """Check if index is empty.
        
        Returns:
            True if index doesn't exist or has no vectors
        """
        return self.index is None or self.index.ntotal == 0
    
    @property
    def size(self) -> int:
        """Get number of vectors in index.
        
        Returns:
            Number of vectors in index, or 0 if index doesn't exist
        """
        return self.index.ntotal if self.index is not None else 0
    
    def __repr__(self) -> str:
        """String representation of retriever."""
        return (
            f"FAISSRetriever(\n"
            f"  index_path={self.index_path},\n"
            f"  embedding_dimension={self.embedding_dimension},\n"
            f"  index_size={self.size},\n"
            f"  num_documents={len(self.documents)}\n"
            f")"
        )
