"""
KAIROS ChromaDB Store Backend

ChromaDB adapter for KAIROS memory storage.
Provides persistent vector storage with ChromaDB's built-in embedding support.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json

from .base import BaseStore
from ..utils import logger

# Optional ChromaDB import
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaStore(BaseStore):
    """
    ChromaDB-based vector store.
    
    Features:
    - Built-in persistence
    - Metadata filtering
    - Scalable to millions of vectors
    
    Requires: pip install chromadb
    """
    
    def __init__(
        self, 
        storage_path: str = "./data/kairos_chroma",
        collection_name: str = "kairos_memories"
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            storage_path: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.storage_path = storage_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=storage_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized with {self.collection.count()} existing vectors")
    
    def store(
        self, 
        token_id: str, 
        latent_vector: np.ndarray, 
        metadata: Optional[Dict] = None,
        float_vector: Optional[np.ndarray] = None
    ) -> bool:
        try:
            # Use float vector if provided (for binary latent vectors)
            vector = float_vector if float_vector is not None else latent_vector
            
            # Ensure vector is float and 1D
            vector = np.array(vector, dtype=np.float32).flatten().tolist()
            
            # Prepare metadata (ChromaDB requires string/int/float/bool values)
            meta = self._sanitize_metadata(metadata or {})
            meta['stored_at'] = time.time()
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=[token_id],
                embeddings=[vector],
                metadatas=[meta]
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store {token_id}: {e}", exc_info=True)
            return False
    
    def retrieve(self, token_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        try:
            result = self.collection.get(
                ids=[token_id],
                include=["embeddings", "metadatas"]
            )
            
            if not result['ids']:
                return None
            
            vector = np.array(result['embeddings'][0], dtype=np.float32)
            metadata = result['metadatas'][0] if result['metadatas'] else {}
            
            return (vector, metadata)
        except Exception as e:
            logger.error(f"Failed to retrieve {token_id}: {e}", exc_info=True)
            return None
    
    def retrieve_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        try:
            # Prepare query vector
            query = np.array(query_vector, dtype=np.float32).flatten().tolist()
            
            # Build where clause for filters
            where = self._build_where_clause(filters) if filters else None
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query],
                n_results=top_k,
                where=where,
                include=["distances", "metadatas"]
            )
            
            # Convert results
            output = []
            if results['ids'] and results['ids'][0]:
                for i, token_id in enumerate(results['ids'][0]):
                    # ChromaDB returns distances, convert to similarity
                    # For cosine distance: similarity = 1 - distance
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1.0 - distance
                    
                    # Apply importance weighting
                    meta = results['metadatas'][0][i] if results['metadatas'] else {}
                    importance = meta.get('importance', 0.5)
                    weighted_score = similarity * (0.5 + importance)
                    
                    output.append((token_id, weighted_score, meta))
            
            # Sort by weighted score
            output.sort(key=lambda x: x[1], reverse=True)
            return output
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return []
    
    def delete(self, token_id: str) -> bool:
        try:
            self.collection.delete(ids=[token_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete {token_id}: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict:
        count = self.collection.count()
        return {
            'total_tokens': count,
            'total_size_bytes': 0,  # ChromaDB doesn't expose this easily
            'total_size_mb': 0,
            'avg_token_size': 0,
            'backend': 'chroma',
            'collection_name': self.collection_name
        }
    
    def get_all_tokens(self) -> Dict[str, Dict]:
        try:
            # Get all items from collection
            result = self.collection.get(
                include=["embeddings", "metadatas"]
            )
            
            output = {}
            for i, token_id in enumerate(result['ids']):
                output[token_id] = {
                    'vector': np.array(result['embeddings'][i], dtype=np.float32),
                    'metadata': result['metadatas'][i] if result['metadatas'] else {}
                }
            
            return output
        except Exception as e:
            logger.error(f"Failed to get all tokens: {e}", exc_info=True)
            return {}
    
    def clear(self) -> None:
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to clear: {e}", exc_info=True)
    
    def persist(self) -> None:
        # ChromaDB with PersistentClient auto-persists
        pass
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata values to ChromaDB-compatible types."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = json.dumps(value)
            elif value is None:
                continue
            else:
                sanitized[key] = str(value)
        return sanitized
    
    def _build_where_clause(self, filters: Dict) -> Optional[Dict]:
        """Build ChromaDB where clause from filters."""
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {"$and": conditions}
