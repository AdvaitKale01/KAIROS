"""
KAIROS NumPy Store Backend

Default storage backend using NumPy for in-memory vector operations.
FAISS-like performance with pure NumPy (no external dependencies).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path

from .base import BaseStore
from ..utils import logger


class NumpyStore(BaseStore):
    """
    NumPy-based vector store (default backend).
    
    Features:
    - Fast cosine similarity search
    - Metadata indexing for filtering
    - Disk persistence with compression
    - Cached matrix operations for speed
    """
    
    LEGACY_DIM = 384
    MULTIDIM_DIM = 408
    
    def __init__(self, storage_path: str = "./data/kairos_latent"):
        """
        Initialize NumPy store.
        
        Args:
            storage_path: Directory for persisting tokens
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.tokens: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict] = {}
        self.index: Dict[str, List[str]] = {}
        
        # Cached normalized matrix for fast retrieval
        self._cached_matrix: Optional[np.ndarray] = None
        self._cached_token_ids: Optional[List[str]] = None
        self._cache_dirty: bool = True
        
        # Load existing tokens
        self._load_from_disk()
    
    def store(
        self, 
        token_id: str, 
        latent_vector: np.ndarray, 
        metadata: Optional[Dict] = None,
        float_vector: Optional[np.ndarray] = None
    ) -> bool:
        try:
            is_binary = latent_vector.dtype == np.uint8 and len(latent_vector) == 48
            
            if is_binary and float_vector is not None:
                self.tokens[token_id] = float_vector.copy()
                binary_for_disk = latent_vector
            else:
                self.tokens[token_id] = latent_vector.copy()
                binary_for_disk = None
            
            meta = metadata or {}
            meta['stored_at'] = time.time()
            meta['vector_shape'] = self.tokens[token_id].shape
            meta['is_binary_storage'] = is_binary
            self._metadata[token_id] = meta
            
            self._update_indexes(token_id, meta)
            self._save_token(token_id, binary_vector=binary_for_disk)
            self._cache_dirty = True
            
            return True
        except Exception as e:
            logger.error(f"Failed to store {token_id}: {e}", exc_info=True)
            return False
    
    def retrieve(self, token_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        if token_id not in self.tokens:
            return None
        return (self.tokens[token_id].copy(), self._metadata[token_id].copy())
    
    def retrieve_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        if not self.tokens:
            return []
        
        candidate_ids = self._apply_filters(filters) if filters else list(self.tokens.keys())
        if not candidate_ids:
            return []
        
        try:
            # Filter to only candidate tokens
            candidate_set = set(candidate_ids)
            
            # Use cached matrix if available, but filter to candidates
            if not self._cache_dirty and self._cached_matrix is not None:
                # Get indices of candidate tokens in cached order
                token_ids_order = self._cached_token_ids
                candidate_indices = [i for i, tid in enumerate(token_ids_order) if tid in candidate_set]
                
                if not candidate_indices:
                    return []
                
                # Extract candidate rows from cached matrix
                normalized_matrix = self._cached_matrix[candidate_indices]
                candidate_token_ids = [token_ids_order[i] for i in candidate_indices]
                importance_array = np.array([
                    self._metadata[tid].get('importance', 0.5) for tid in candidate_token_ids
                ])
            else:
                # Rebuild cache with only candidate tokens
                candidate_token_ids = [tid for tid in self.tokens.keys() if tid in candidate_set]
                if not candidate_token_ids:
                    return []
                
                max_dim = max(len(self.tokens[tid]) for tid in candidate_token_ids)
                
                padded_tokens = []
                for tid in candidate_token_ids:
                    vec = self.tokens[tid]
                    if len(vec) < max_dim:
                        padded = np.zeros(max_dim, dtype=np.float32)
                        padded[:len(vec)] = vec
                        if max_dim == self.MULTIDIM_DIM and len(vec) == self.LEGACY_DIM:
                            padded[390] = 1.0
                        padded_tokens.append(padded)
                    else:
                        padded_tokens.append(vec)
                
                token_matrix = np.stack(padded_tokens)
                token_matrix = np.nan_to_num(token_matrix, nan=0.0, posinf=1000.0, neginf=-1000.0)
                
                matrix_norms = np.linalg.norm(token_matrix, axis=1, keepdims=True) + 1e-8
                normalized_matrix = token_matrix / matrix_norms
                
                # Update cache with all tokens (for future queries)
                all_token_ids = list(self.tokens.keys())
                all_max_dim = max(len(self.tokens[tid]) for tid in all_token_ids)
                all_padded_tokens = []
                for tid in all_token_ids:
                    vec = self.tokens[tid]
                    if len(vec) < all_max_dim:
                        padded = np.zeros(all_max_dim, dtype=np.float32)
                        padded[:len(vec)] = vec
                        if all_max_dim == self.MULTIDIM_DIM and len(vec) == self.LEGACY_DIM:
                            padded[390] = 1.0
                        all_padded_tokens.append(padded)
                    else:
                        all_padded_tokens.append(vec)
                
                all_token_matrix = np.stack(all_padded_tokens)
                all_token_matrix = np.nan_to_num(all_token_matrix, nan=0.0, posinf=1000.0, neginf=-1000.0)
                all_matrix_norms = np.linalg.norm(all_token_matrix, axis=1, keepdims=True) + 1e-8
                self._cached_matrix = all_token_matrix / all_matrix_norms
                self._cached_token_ids = all_token_ids
                self._cache_dirty = False
                
                importance_array = np.array([
                    self._metadata[tid].get('importance', 0.5) for tid in candidate_token_ids
                ])
            
            # Align query dimension to match stored vectors
            stored_dim = normalized_matrix.shape[1]
            query_dim = len(query_vector)
            
            if query_dim < stored_dim:
                # Pad query to match stored dimension
                padded_query = np.zeros(stored_dim, dtype=np.float32)
                padded_query[:query_dim] = query_vector
                # If padding from 384d to 408d, set neutral emotion flag
                if stored_dim == self.MULTIDIM_DIM and query_dim == self.LEGACY_DIM:
                    padded_query[390] = 1.0  # neutral emotion category flag
                query_vector = padded_query
            elif query_dim > stored_dim:
                # Truncate query to match stored dimension (keep semantic part)
                query_vector = query_vector[:stored_dim]
            
            # For multi-dimensional vectors (408d), use only semantic part (first 384d) for similarity
            # Emotional and temporal dimensions are for filtering, not similarity
            if stored_dim == self.MULTIDIM_DIM and query_dim >= self.LEGACY_DIM:
                # Extract semantic portions (first 384 dimensions)
                semantic_query = query_vector[:self.LEGACY_DIM]
                semantic_matrix = normalized_matrix[:, :self.LEGACY_DIM]
                
                # Normalize semantic query (handle zero vectors)
                query_norm_val = np.linalg.norm(semantic_query)
                if query_norm_val < 1e-8:
                    query_norm = semantic_query  # Zero vector, can't normalize
                else:
                    query_norm = semantic_query / query_norm_val
                
                # Normalize semantic matrix rows (handle zero vectors)
                semantic_norms = np.linalg.norm(semantic_matrix, axis=1, keepdims=True)
                semantic_norms = np.maximum(semantic_norms, 1e-8)  # Prevent division by zero
                semantic_matrix_norm = semantic_matrix / semantic_norms
                
                # Compute cosine similarity on semantic dimensions only
                cosine_scores = semantic_matrix_norm @ query_norm
            else:
                # For legacy 384d vectors, use full vector
                query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
                cosine_scores = normalized_matrix @ query_norm
            
            # Apply importance weighting: weighted_score = similarity * (0.5 + importance)
            # This gives importance values between 0.5 and 1.5 a multiplier effect
            weighted_scores = cosine_scores * (0.5 + importance_array)
            
            # Get top-k from candidates only
            num_candidates = len(weighted_scores)
            if num_candidates <= top_k:
                top_indices = np.argsort(weighted_scores)[::-1]
            else:
                top_indices = np.argpartition(weighted_scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(weighted_scores[top_indices])[::-1]]
            
            results = []
            seen_token_ids = set()  # Prevent duplicates
            for idx in top_indices:
                token_id = candidate_token_ids[idx]
                if token_id in seen_token_ids:
                    continue  # Skip duplicates
                seen_token_ids.add(token_id)
                
                # Use raw cosine score (before importance weighting) for similarity
                # The importance weighting is for ranking, but similarity should be pure cosine
                cosine_score = float(cosine_scores[idx])
                weighted_score = float(weighted_scores[idx])
                
                # Return weighted score but also store raw cosine for debugging
                results.append((token_id, weighted_score, self._metadata[token_id]))
            
            return results
                
        except Exception as e:
            logger.warning(f"Vectorization failed: {e}, falling back to iterative search")
            return self._fallback_search(query_vector, candidate_ids, top_k)
    
    def _fallback_search(self, query_vector, candidate_ids, top_k):
        """Fallback iterative search."""
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        similarities = []
        for token_id in candidate_ids:
            token_vector = np.nan_to_num(self.tokens[token_id], nan=0.0)
            token_norm = token_vector / (np.linalg.norm(token_vector) + 1e-8)
            similarity = float(np.dot(query_norm, token_norm))
            importance = self._metadata[token_id].get('importance', 0.5)
            similarities.append((token_id, similarity * (0.5 + importance), self._metadata[token_id]))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def delete(self, token_id: str) -> bool:
        if token_id not in self.tokens:
            return False
        
        del self.tokens[token_id]
        del self._metadata[token_id]
        self._remove_from_indexes(token_id)
        
        token_file = self.storage_path / f"{token_id}.npz"
        if token_file.exists():
            token_file.unlink()
        
        self._cache_dirty = True
        return True
    
    def get_stats(self) -> Dict:
        total_tokens = len(self.tokens)
        total_size = sum(v.nbytes for v in self.tokens.values())
        return {
            'total_tokens': total_tokens,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_token_size': round(total_size / total_tokens, 2) if total_tokens > 0 else 0,
            'backend': 'numpy'
        }
    
    def get_all_tokens(self) -> Dict[str, Dict]:
        return {
            token_id: {'vector': self.tokens[token_id], 'metadata': self._metadata.get(token_id, {})}
            for token_id in self.tokens
        }
    
    def clear(self) -> None:
        for token_id in list(self.tokens.keys()):
            self.delete(token_id)
        self._cache_dirty = True
    
    def persist(self) -> None:
        for token_id in self.tokens:
            self._save_token(token_id)
    
    # Internal helpers
    def _apply_filters(self, filters: Dict) -> List[str]:
        candidates = set(self.tokens.keys())
        for key, value in filters.items():
            matching = set(self.index.get(f"{key}:{value}", []))
            if not matching:
                matching = {tid for tid, meta in self._metadata.items() if meta.get(key) == value}
            candidates &= matching
        return list(candidates)
    
    def _update_indexes(self, token_id: str, metadata: Dict):
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                index_key = f"{key}:{value}"
                if index_key not in self.index:
                    self.index[index_key] = []
                if token_id not in self.index[index_key]:
                    self.index[index_key].append(token_id)
    
    def _remove_from_indexes(self, token_id: str):
        for index_list in self.index.values():
            if token_id in index_list:
                index_list.remove(token_id)
    
    def _save_token(self, token_id: str, binary_vector: Optional[np.ndarray] = None):
        token_file = self.storage_path / f"{token_id}.npz"
        vector = binary_vector if binary_vector is not None else self.tokens[token_id]
        np.savez_compressed(token_file, vector=vector, metadata=json.dumps(self._metadata[token_id]))
    
    def _load_from_disk(self):
        if not self.storage_path.exists():
            return
        for token_file in self.storage_path.glob("*.npz"):
            try:
                data = np.load(token_file, allow_pickle=True)
                token_id = token_file.stem
                self.tokens[token_id] = data['vector']
                self._metadata[token_id] = json.loads(str(data['metadata']))
                self._update_indexes(token_id, self._metadata[token_id])
            except Exception as e:
                logger.warning(f"Failed to load {token_file}: {e}")
