"""
Query Encoder - KAIROS Memory Component
Encodes user queries into the same latent space as compressed memories.

Key CLaRa principle: Joint optimization with generator for better retrieval.

Multi-dimensional encoding with emotional + temporal context.
"""
import numpy as np
from typing import Dict, List, Optional
import hashlib
import time

# GPU thread safety (local)
from .utils import gpu_lock, logger

# Sentence transformer for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Multi-dimensional encoder
from .compressor import MultiDimensionalEncoder


class QueryEncoder:
    """
    Encode queries into latent space for similarity-based retrieval.
    
    This shares the same continuous latent space as the LatentCompressor,
    enabling efficient semantic search without full-text matching.
    
    Features:
    - 384d semantic from sentence transformer
    - 8d emotional context (inferred from query)
    - 16d temporal context (query-time encoding)
    """
    
    # Temporal words that signal time-based queries
    TEMPORAL_KEYWORDS = {
        'yesterday': -24, 'today': 0, 'earlier': -6, 'this morning': -12,
        'last week': -168, 'last night': -12, 'recently': -24,
        'a while ago': -72, 'last month': -720
    }
    
    # Emotional query keywords
    EMOTIONAL_KEYWORDS = {
        'happy': {'pleasure': 0.8, 'arousal': 0.5, 'category': 'positive'},
        'sad': {'pleasure': -0.8, 'arousal': -0.3, 'category': 'negative'},
        'angry': {'pleasure': -0.6, 'arousal': 0.8, 'category': 'negative'},
        'excited': {'pleasure': 0.7, 'arousal': 0.9, 'category': 'positive'},
        'scared': {'pleasure': -0.5, 'arousal': 0.7, 'category': 'negative'},
        'calm': {'pleasure': 0.3, 'arousal': -0.5, 'category': 'positive'},
        'anxious': {'pleasure': -0.4, 'arousal': 0.6, 'category': 'negative'},
        'stressed': {'pleasure': -0.5, 'arousal': 0.7, 'category': 'negative'},
    }
    
    def __init__(self, embedding_dim: int = 384, multidim_dim: int = 408):
        """
        Initialize query encoder.
        
        Args:
            embedding_dim: Dimension of semantic latent space
            multidim_dim: Dimension of full multi-dimensional space (408)
        """
        self.embedding_dim = embedding_dim
        self.multidim_dim = multidim_dim
        self.feedback_scores: Dict[str, float] = {}  # Track retrieval quality
        
        # Load same model as compressor for consistent latent space
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.debug("Loaded sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.encoder = None
            logger.warning("sentence-transformers not available. Using fallback hash encoding")
        
        # Multi-dimensional encoder for emotional + temporal
        self.multidim_encoder = MultiDimensionalEncoder()
        
    def encode_query(self, query: str, context: Optional[str] = None) -> np.ndarray:
        """
        Encode a query into latent space using REAL semantic embeddings.
        
        Args:
            query: User query or prompt
            context: Optional conversation context
            
        Returns:
            Latent vector representing the query (384d)
        """
        # Combine query with context if available
        full_query = f"{context}\n{query}" if context else query
        
        # Use sentence transformer if available (REAL semantic encoding)
        if self.encoder:
            # GPU lock prevents Metal command buffer corruption
            with gpu_lock:
                query_vector = self.encoder.encode(
                    full_query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        else:
            # Fallback: Hash-based
            query_hash = hashlib.sha256(full_query.encode()).digest()
            query_vector = np.frombuffer(query_hash, dtype=np.float32)[:self.embedding_dim]
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        return query_vector
    
    def encode_with_intent(self, query: str, intent: str) -> np.ndarray:
        """
        Encode query with retrieval intent for better targeting.
        
        Args:
            query: User query
            intent: Retrieval intent ('factual', 'episodic', 'creative', etc.)
            
        Returns:
            Intent-aware latent vector
        """
        # Augment query with intent signal
        augmented = f"[{intent.upper()}] {query}"
        return self.encode_query(augmented)
    
    def encode_multidim_query(self, query: str, 
                               current_emotion: Optional[Dict] = None,
                               context: Optional[str] = None) -> np.ndarray:
        """
        Encode query into multi-dimensional latent space (408d).
        
        Automatically infers emotional and temporal context from the query text.
        
        Args:
            query: User query
            current_emotion: Current emotional state (if known)
            context: Optional conversation context
            
        Returns:
            408-dimensional query vector
        """
        # Get semantic embedding (384d)
        semantic_vec = self.encode_query(query, context)
        
        # Infer emotional context from query (or use provided)
        emotion_state = self._infer_query_emotion(query, current_emotion)
        
        # Infer temporal context from query
        query_timestamp = self._infer_query_time(query)
        
        # Combine into multi-dimensional vector
        multidim_vec = self.multidim_encoder.encode_multidim(
            semantic_vec, emotion_state, query_timestamp
        )
        
        return multidim_vec
    
    def _infer_query_emotion(self, query: str, 
                              current_emotion: Optional[Dict] = None) -> Dict:
        """
        Infer emotional context from query keywords.
        
        If the query asks about emotions, uses those emotions.
        Otherwise falls back to current emotional state or neutral.
        """
        query_lower = query.lower()
        
        # Check for emotional keywords in query
        for keyword, emotion_vals in self.EMOTIONAL_KEYWORDS.items():
            if keyword in query_lower:
                return {
                    'pleasure': emotion_vals.get('pleasure', 0.0),
                    'arousal': emotion_vals.get('arousal', 0.0),
                    'dominance': 0.0,
                    'intensity': 0.7,
                    'category': emotion_vals.get('category', 'neutral')
                }
        
        # Fall back to current emotion or neutral
        if current_emotion:
            return current_emotion
        
        return {'pleasure': 0.0, 'arousal': 0.0, 'dominance': 0.0, 
                'intensity': 0.5, 'category': 'neutral'}
    
    def _infer_query_time(self, query: str) -> float:
        """
        Infer temporal context from query keywords.
        
        E.g., "What did we discuss yesterday?" -> returns timestamp from 24h ago
        """
        query_lower = query.lower()
        
        for keyword, hours_offset in self.TEMPORAL_KEYWORDS.items():
            if keyword in query_lower:
                return time.time() + (hours_offset * 3600)
        
        # Default to current time
        return time.time()
    
    def record_feedback(self, query_id: str, retrieval_quality: float):
        """
        Record feedback on retrieval quality for learning.
        
        CLaRa principle: Generator teaches retriever what's important.
        
        Args:
            query_id: Identifier for this query
            retrieval_quality: Score 0-1 indicating how useful the retrieval was
        """
        self.feedback_scores[query_id] = retrieval_quality
    
    def get_average_quality(self) -> float:
        """Get average retrieval quality across all queries."""
        if not self.feedback_scores:
            return 0.0
        return sum(self.feedback_scores.values()) / len(self.feedback_scores)
    
    def optimize_from_feedback(self):
        """
        Optimize encoding based on accumulated feedback.
        
        Placeholder for future learning loop implementation.
        In production, this would fine-tune the encoding model.
        """
        avg_quality = self.get_average_quality()
        
        if avg_quality < 0.7:
            # Quality is low - encoder needs adjustment
            logger.warning(f"Average quality: {avg_quality:.2f}. Optimization needed.")
            # TODO: Implement encoder fine-tuning
        
        return avg_quality
