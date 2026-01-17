"""
KAIROS Storage Backends - Abstract Base Class

Defines the interface that all storage backends must implement.
Allows users to swap between NumPy (default), ChromaDB, or custom backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BaseStore(ABC):
    """
    Abstract base class for KAIROS storage backends.
    
    Implement this interface to create custom storage backends
    (e.g., FAISS, Pinecone, Weaviate, Qdrant, etc.)
    
    Example:
        class MyCustomStore(BaseStore):
            def store(self, token_id, vector, metadata):
                # Your implementation
                pass
            ...
    """
    
    @abstractmethod
    def store(
        self, 
        token_id: str, 
        latent_vector: np.ndarray, 
        metadata: Optional[Dict] = None,
        float_vector: Optional[np.ndarray] = None
    ) -> bool:
        """
        Store a memory token.
        
        Args:
            token_id: Unique identifier for this token
            latent_vector: The vector representation to store
            metadata: Optional metadata dict
            float_vector: Optional float vector for similarity (if latent is binary)
            
        Returns:
            True if successfully stored
        """
        pass
    
    @abstractmethod
    def retrieve(self, token_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Retrieve a specific token by ID.
        
        Args:
            token_id: Token identifier
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        pass
    
    @abstractmethod
    def retrieve_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve most similar tokens.
        
        Args:
            query_vector: Query vector in same latent space
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (token_id, similarity_score, metadata) tuples
        """
        pass
    
    @abstractmethod
    def delete(self, token_id: str) -> bool:
        """
        Delete a token.
        
        Args:
            token_id: Token to delete
            
        Returns:
            True if successfully deleted
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dict with keys like 'total_tokens', 'total_size_bytes', etc.
        """
        pass
    
    @abstractmethod
    def get_all_tokens(self) -> Dict[str, Dict]:
        """
        Get all tokens with their metadata.
        
        Returns:
            Dict mapping token_id to {'vector': np.ndarray, 'metadata': dict}
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored tokens."""
        pass
    
    def persist(self) -> None:
        """
        Persist data to disk (optional).
        
        Override if your backend needs explicit persistence.
        Default is no-op (for backends with auto-persistence).
        """
        pass
