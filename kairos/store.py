"""
Latent Token Store - KAIROS Memory Component
Efficient storage and retrieval of compressed memory tokens.

FAISS-like vector storage with numpy/MLX acceleration.

Note: This is a legacy implementation. The current system uses
the backend architecture (NumpyStore, ChromaStore) which implement BaseStore.
This file is kept for backward compatibility.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path
from .utils import logger


class LatentTokenStore:
    """
    High-performance storage for compressed latent memory tokens.
    
    Features:
    - Fast similarity search in continuous latent space
    - Metadata indexing for filtering
    - Persistence to disk
    - Multi-dimensional vector support (384d legacy, 408d multi-dim)
    """
    
    # Vector dimensions
    LEGACY_DIM = 384
    MULTIDIM_DIM = 408
    
    def __init__(self, storage_path: str = "./data/kairos_latent"):
        """
        Initialize latent token store.
        
        Args:
            storage_path: Directory for persisting tokens
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage (will be persisted)
        self.tokens: Dict[str, np.ndarray] = {}  # token_id -> latent_vector
        self.metadata: Dict[str, Dict] = {}  # token_id -> metadata
        self.index: Dict[str, List[str]] = {}  # metadata_key -> [token_ids]
        
        # OPTIMIZATION: Cached normalized matrix for fast retrieval
        self._cached_matrix: Optional[np.ndarray] = None
        self._cached_token_ids: Optional[List[str]] = None
        self._cache_dirty: bool = True
        
        # Load existing tokens if available
        self._load_from_disk()
    
    def store(self, token_id: str, latent_vector: np.ndarray, metadata: Optional[Dict] = None,
              float_vector: Optional[np.ndarray] = None) -> bool:
        """
        Store a compressed memory token.
        
        Args:
            token_id: Unique identifier for this token
            latent_vector: Compressed latent representation (binary or float)
            metadata: Optional metadata (timestamp, type, tags, etc.)
            float_vector: Optional float vector for similarity (if latent is binary)
            
        Returns:
            True if successfully stored
        """
        try:
            # Determine if binary
            is_binary = latent_vector.dtype == np.uint8 and len(latent_vector) == 48
            
            if is_binary and float_vector is not None:
                # Store float for in-memory search, binary for disk
                self.tokens[token_id] = float_vector.copy()
                binary_for_disk = latent_vector
            else:
                # Store as-is
                self.tokens[token_id] = latent_vector.copy()
                binary_for_disk = None
            
            # Store metadata with timestamp
            meta = metadata or {}
            meta['stored_at'] = time.time()
            meta['vector_shape'] = self.tokens[token_id].shape
            meta['is_binary_storage'] = is_binary
            self.metadata[token_id] = meta
            
            # Update indexes
            self._update_indexes(token_id, meta)
            
            # Persist to disk (binary vectors are 32x smaller!)
            self._save_token(token_id, binary_vector=binary_for_disk)
            
            # Invalidate cache
            self._cache_dirty = True
            
            return True
        except Exception as e:
            logger.error(f"Failed to store token {token_id}: {e}", exc_info=True)
            return False
    
    def retrieve(self, token_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Retrieve a specific token by ID.
        
        Returns:
            Tuple of (latent_vector, metadata) or None if not found
        """
        if token_id not in self.tokens:
            return None
        
        return (self.tokens[token_id].copy(), self.metadata[token_id].copy())
    
    def retrieve_similar(self, query_vector: np.ndarray, top_k: int = 5, 
                        filters: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve most similar tokens using cosine similarity.
        
        This is the core CLaRa-inspired retrieval: operating in continuous latent space.
        
        Args:
            query_vector: Query representation in same latent space
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {'type': 'episodic'})
            
        Returns:
            List of (token_id, similarity_score, metadata) tuples, sorted by relevance
        """
        if not self.tokens:
            return []
        
        # Apply filters first
        candidate_ids = self._apply_filters(filters) if filters else list(self.tokens.keys())
        
        if not candidate_ids:
            return []
        
        try:
            # OPTIMIZATION: Use cached normalized matrix if available and not dirty
            if not self._cache_dirty and self._cached_matrix is not None and self._cached_token_ids is not None:
                # Use cache (fast path)
                normalized_matrix = self._cached_matrix
                token_ids_order = self._cached_token_ids
                
                # Build importance array from cache order
                importance_array = np.array([self.metadata[tid].get('importance', 0.5) for tid in token_ids_order])
            else:
                # Rebuild cache (slow path, only on first query or after store)
                token_ids_order = list(self.tokens.keys())
                
                # Handle mixed dimensions: find max dimension
                max_dim = max(len(self.tokens[tid]) for tid in token_ids_order)
                
                # Pad all vectors to max dimension for consistent matrix operations
                padded_tokens = []
                for tid in token_ids_order:
                    vec = self.tokens[tid]
                    if len(vec) < max_dim:
                        # Pad with neutral values
                        padded = np.zeros(max_dim, dtype=np.float32)
                        padded[:len(vec)] = vec
                        if max_dim == self.MULTIDIM_DIM and len(vec) == self.LEGACY_DIM:
                            padded[390] = 1.0  # neutral emotion category flag
                        padded_tokens.append(padded)
                    else:
                        padded_tokens.append(vec)
                
                token_matrix = np.stack(padded_tokens)
                
                # Sanitize matrix
                token_matrix = np.nan_to_num(token_matrix, nan=0.0, posinf=1000.0, neginf=-1000.0)
                
                # Normalize matrix
                matrix_norms = np.linalg.norm(token_matrix, axis=1, keepdims=True) + 1e-8
                normalized_matrix = token_matrix / matrix_norms
                
                # Cache it
                self._cached_matrix = normalized_matrix
                self._cached_token_ids = token_ids_order
                self._cache_dirty = False
                
                # Build importance array
                importance_array = np.array([self.metadata[tid].get('importance', 0.5) for tid in token_ids_order])
            
            # Align query dimension with stored vectors
            if len(query_vector) < normalized_matrix.shape[1]:
                # Pad query to match stored vectors
                padded_query = np.zeros(normalized_matrix.shape[1], dtype=np.float32)
                padded_query[:len(query_vector)] = query_vector
                padded_query[390] = 1.0 if len(query_vector) == self.LEGACY_DIM else padded_query[390]
                query_vector = padded_query
            elif len(query_vector) > normalized_matrix.shape[1]:
                # Truncate query to match (semantic portion)
                query_vector = query_vector[:normalized_matrix.shape[1]]
            
            # Normalize query
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            
            # FULLY VECTORIZED: Compute cosine similarity for ALL tokens
            cosine_scores = normalized_matrix @ query_norm
            
            # FULLY VECTORIZED: Apply importance weighting
            # weighted_score = similarity * (0.5 + importance)
            weighted_scores = cosine_scores * (0.5 + importance_array)
            
            # FULLY VECTORIZED: Get top-k indices using argpartition (O(n) instead of O(n log n))
            if len(weighted_scores) <= top_k:
                top_indices = np.argsort(weighted_scores)[::-1]
            else:
                # Partial sort: get top_k largest
                top_indices = np.argpartition(weighted_scores, -top_k)[-top_k:]
                # Sort those top_k by score
                top_indices = top_indices[np.argsort(weighted_scores[top_indices])[::-1]]
            
            # Build result list
            results = []
            for idx in top_indices:
                token_id = token_ids_order[idx]
                score = float(weighted_scores[idx])
                # Clip to valid range
                score = max(-1.0, min(1.0, score))
                results.append((token_id, score, self.metadata[token_id]))
            
            return results
                
        except Exception as e:
            logger.warning(f"Vectorization failed ({e}), falling back to iterative loop")
            # Fallback to iterative loop
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            similarities = []
            for token_id in candidate_ids:
                token_vector = self.tokens[token_id]
                token_vector = np.nan_to_num(token_vector, nan=0.0)
                token_norm = token_vector / (np.linalg.norm(token_vector) + 1e-8)
                similarity = float(np.dot(query_norm, token_norm))
                importance = self.metadata[token_id].get('importance', 0.5)
                weighted_score = similarity * (0.5 + importance)
                similarities.append((token_id, weighted_score, self.metadata[token_id]))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    def delete(self, token_id: str) -> bool:
        """Delete a token and its metadata."""
        if token_id not in self.tokens:
            return False
        
        # Remove from storage
        del self.tokens[token_id]
        del self.metadata[token_id]
        
        # Remove from indexes
        self._remove_from_indexes(token_id)
        
        # Remove file
        token_file = self.storage_path / f"{token_id}.npz"
        if token_file.exists():
            token_file.unlink()
        
        # Invalidate cache
        self._cache_dirty = True
        
        return True
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        total_tokens = len(self.tokens)
        total_size = sum(v.nbytes for v in self.tokens.values())
        
        return {
            'total_tokens': total_tokens,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_token_size': round(total_size / total_tokens, 2) if total_tokens > 0 else 0
        }

    def get_all_tokens(self) -> Dict[str, Dict]:
        """
        Get all tokens with their metadata.
        Used for emotional retrieval and full-scan queries.
        """
        result = {}
        for token_id in self.tokens.keys():
            result[token_id] = {
                'vector': self.tokens[token_id],
                'metadata': self.metadata.get(token_id, {})
            }
        return result
    
    def clear(self):
        """Clear all tokens from memory and disk."""
        for token_id in list(self.tokens.keys()):
            self.delete(token_id)
        
        self._cache_dirty = True
    
    def _apply_filters(self, filters: Dict) -> List[str]:
        """Apply metadata filters to get candidate token IDs."""
        # Start with all tokens
        candidates = set(self.tokens.keys())
        
        # Apply each filter
        for key, value in filters.items():
            if key in self.index:
                # Use index for fast lookup
                matching_ids = set(self.index.get(f"{key}:{value}", []))
                candidates &= matching_ids
            else:
                # Scan metadata
                matching_ids = {
                    tid for tid, meta in self.metadata.items()
                    if meta.get(key) == value
                }
                candidates &= matching_ids
        
        return list(candidates)
    
    def _update_indexes(self, token_id: str, metadata: Dict):
        """Update metadata indexes for fast filtering."""
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                index_key = f"{key}:{value}"
                if index_key not in self.index:
                    self.index[index_key] = []
                if token_id not in self.index[index_key]:
                    self.index[index_key].append(token_id)
    
    def _remove_from_indexes(self, token_id: str):
        """Remove token from all indexes."""
        for index_list in self.index.values():
            if token_id in index_list:
                index_list.remove(token_id)
    
    def _save_token(self, token_id: str, binary_vector: Optional[np.ndarray] = None):
        """Save a single token to disk. Uses binary vector if provided (32x smaller)."""
        token_file = self.storage_path / f"{token_id}.npz"
        
        # Use binary vector for disk storage if available (32x compression)
        vector_to_save = binary_vector if binary_vector is not None else self.tokens[token_id]
        
        np.savez_compressed(
            token_file,
            vector=vector_to_save,
            metadata=json.dumps(self.metadata[token_id])
        )
    
    def _load_from_disk(self):
        """Load all tokens from disk on initialization."""
        if not self.storage_path.exists():
            return
        
        for token_file in self.storage_path.glob("*.npz"):
            try:
                data = np.load(token_file, allow_pickle=True)
                token_id = token_file.stem
                
                self.tokens[token_id] = data['vector']
                self.metadata[token_id] = json.loads(str(data['metadata']))
                self._update_indexes(token_id, self.metadata[token_id])
            except Exception as e:
                logger.warning(f"Failed to load token {token_file}: {e}")
    
    def persist_all(self):
        """Force persist all tokens to disk."""
        for token_id in self.tokens.keys():
            self._save_token(token_id)
