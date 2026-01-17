"""
KAIROS Memory - Main Orchestrator
Knowledge-Augmented Intelligent Retrieval and Organizational System

Standalone, importable memory system for GenAI applications.
Combines CLaRa-inspired latent compression with efficient vector retrieval.
"""
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import numpy as np

from .compressor import LatentCompressor
from .encoder import QueryEncoder
from .feedback import GeneratorFeedback
from .backends import BaseStore, get_backend
from .utils import logger


class KAIROSMemory:
    """
    KAIROS memory architecture for any GenAI application.
    
    Features:
    - Working memory: Current conversation context
    - Short-term buffer: Recent exchanges
    - Multi-dimensional encoding: Semantic + Emotional + Temporal (408d)
    - Generator feedback loop: Improves retrieval over time
    - Pluggable storage backends: NumPy (default), ChromaDB, or custom
    
    Usage:
        from kairos import KAIROSMemory
        
        # Default NumPy backend
        memory = KAIROSMemory()
        
        # ChromaDB backend
        memory = KAIROSMemory(backend="chroma")
        
        # Custom backend
        memory = KAIROSMemory(backend=MyCustomStore())
    """
    
    def __init__(
        self, 
        storage_path: str = "./data/kairos",
        backend: Union[str, BaseStore] = "numpy",
        use_multidim: bool = True,
        enable_feedback: bool = True,
        working_memory_limit: int = 20,
        short_term_limit: int = 50,
        embedding_model: Optional[Any] = None
    ):
        """
        Initialize KAIROS memory system.
        
        Args:
            storage_path: Base directory for persisting memory data
            backend: Storage backend - "numpy" (default), "chroma", or BaseStore instance
            use_multidim: If True, uses 408d multi-dimensional encoding (default)
            enable_feedback: Enable generator feedback loop for improving retrieval
            working_memory_limit: Max items in working memory
            short_term_limit: Max items in short-term buffer
            embedding_model: Optional external embedding model (e.g., SentenceTransformer).
                             Must implement an `encode(text)` method. If None, uses a 
                             deterministic hash-based fallback (non-semantic).
        """
        self.storage_path = storage_path
        self.use_multidim = use_multidim
        self.enable_feedback = enable_feedback
        self.backend_type = backend if isinstance(backend, str) else "custom"
        
        # Traditional layers
        self.working_memory = deque(maxlen=working_memory_limit)
        self.short_term_memory = deque(maxlen=short_term_limit)
        
        # KAIROS components
        self.compressor = LatentCompressor(compression_ratio=16, embedding_model=embedding_model)
        self.query_encoder = QueryEncoder(embedding_model=embedding_model)
        
        # Storage backend (pluggable)
        self.latent_store = get_backend(backend, storage_path)
        
        # Generator feedback
        if enable_feedback:
            self.feedback = GeneratorFeedback(feedback_path=f"{storage_path}/feedback")
            logger.info("Generator feedback loop enabled")
        else:
            self.feedback = None
        
        # Statistics with benchmark timing metrics
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'compression_failures': 0,
            'feedback_sessions': 0,
            'store_latencies': [],
            'retrieve_latencies': [],
            'total_chars_stored': 0,
            'total_bytes_stored': 0,
            'similarity_scores': []
        }
        
        logger.info(f"Initialized (backend={self.backend_type}, multidim={use_multidim})")
    
    def add_to_working(self, role: str, content: str):
        """
        Add message to working memory.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        self.working_memory.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
    
    def get_working_memory(self) -> List[Dict]:
        """Get current working memory as a list."""
        return list(self.working_memory)
    
    def clear_working_memory(self):
        """Clear working memory."""
        self.working_memory.clear()
    
    def consolidate_exchange(
        self, 
        user_msg: str, 
        assistant_msg: str, 
        importance: float = 0.5, 
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Compress and store a conversation exchange.
        
        Args:
            user_msg: User message
            assistant_msg: Assistant response
            importance: Importance score 0-1 (higher = more important)
            metadata: Optional extra metadata (e.g. emotion, tags)
            
        Returns:
            Token ID if stored successfully, None otherwise
        """
        # Input validation
        if not user_msg or not isinstance(user_msg, str):
            logger.warning("consolidate_exchange: user_msg must be a non-empty string")
            return None
        if not assistant_msg or not isinstance(assistant_msg, str):
            logger.warning("consolidate_exchange: assistant_msg must be a non-empty string")
            return None
        if not isinstance(importance, (int, float)) or not (0 <= importance <= 1):
            logger.warning(f"consolidate_exchange: importance must be between 0 and 1, got {importance}")
            importance = max(0.0, min(1.0, float(importance)))
        
        store_start = time.perf_counter()
        
        # Create exchange text
        exchange_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        
        # Build metadata
        base_metadata = {
            'type': 'exchange',
            'timestamp': time.time(),
            'user_msg_preview': user_msg[:100],
            'assistant_msg_preview': assistant_msg[:100],
            'importance': importance,
            'exchange_length': len(exchange_text)
        }
        
        # Merge extra metadata if provided
        if metadata:
            base_metadata.update(metadata)
        
        try:
            if self.use_multidim:
                # Use 408d multi-dimensional encoding
                compressed = self.compressor.compress_multidim(
                    exchange_text,
                    emotion_state=base_metadata,
                    timestamp=base_metadata.get('timestamp'),
                    metadata=base_metadata
                )
                
                token_id = f"mem_{int(time.time() * 1000)}"
                compressed['metadata']['is_multidim'] = True
                
            else:
                # Use standard 384d encoding
                compressed = self.compressor.compress(exchange_text, base_metadata)
                token_id = f"mem_{int(time.time() * 1000)}"
            
            # Store in latent space
            success = self.latent_store.store(
                token_id,
                compressed['latent_vector'],
                compressed['metadata'],
                float_vector=compressed.get('float_vector')
            )
            
            if success:
                self.stats['stores'] += 1
                store_latency = (time.perf_counter() - store_start) * 1000
                self.stats['store_latencies'].append(store_latency)
                self.stats['total_chars_stored'] += len(exchange_text)
                self.stats['total_bytes_stored'] += len(exchange_text.encode('utf-8'))
                
                return token_id
                
        except Exception as e:
            logger.error(f"Compression failed: {e}", exc_info=True)
            self.stats['compression_failures'] += 1
        
        return None
    
    def retrieve_relevant(
        self, 
        query: str, 
        top_k: int = 5, 
        intent: Optional[str] = None
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Retrieve relevant memories from latent space.
        
        Args:
            query: User query or context
            top_k: Number of results to return
            intent: Optional retrieval intent ('factual', 'episodic', etc.)
            
        Returns:
            Tuple of (list of memory dicts, feedback session ID)
        """
        # Input validation
        if not query or not isinstance(query, str):
            logger.warning("retrieve_relevant: query must be a non-empty string")
            return [], None
        if not isinstance(top_k, int) or top_k < 1:
            logger.warning(f"retrieve_relevant: top_k must be a positive integer, got {top_k}")
            top_k = max(1, int(top_k))
        
        retrieve_start = time.perf_counter()
        
        # Encode query
        if self.use_multidim:
            query_vector = self.query_encoder.encode_multidim_query(query)
        elif intent:
            query_vector = self.query_encoder.encode_with_intent(query, intent)
        else:
            query_vector = self.query_encoder.encode_query(query)
        
        try:
            # Retrieve from latent store
            results = self.latent_store.retrieve_similar(query_vector, top_k)
            
            # Re-rank using generator feedback
            if self.feedback:
                results = self.feedback.re_rank_results(results)
            
            # Start feedback session
            session_id = None
            if self.feedback:
                session_id = self.feedback.start_retrieval_session(query, results)
            
            self.stats['retrievals'] += 1
            
            # Track timing
            retrieve_latency = (time.perf_counter() - retrieve_start) * 1000
            self.stats['retrieve_latencies'].append(retrieve_latency)
            
            # Format results
            formatted = []
            for token_id, weighted_score, metadata in results:
                # weighted_score includes importance weighting, but we want raw cosine for similarity
                # Extract raw cosine from weighted: weighted = cosine * (0.5 + importance)
                # So: cosine = weighted / (0.5 + importance)
                importance = metadata.get('importance', 0.5)
                raw_cosine = weighted_score / (0.5 + importance) if (0.5 + importance) > 0 else weighted_score
                raw_cosine = max(-1.0, min(1.0, raw_cosine))  # Clamp to valid cosine range
                
                self.stats['similarity_scores'].append(raw_cosine)
                formatted.append({
                    'token_id': token_id,
                    'similarity': raw_cosine,  # Use raw cosine similarity, not weighted score
                    'metadata': metadata,
                    'content': metadata.get('user_msg_preview', '')
                })
            
            return formatted, session_id
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return [], None
    
    def record_usage(self, session_id: str, token_id: str, usage_score: float):
        """
        Record how much a retrieved memory was used in generation.
        
        This improves future retrieval by learning from feedback.
        
        Args:
            session_id: Feedback session ID from retrieve_relevant()
            token_id: Token ID of the memory
            usage_score: 0-1 score (0=unused, 1=heavily used)
        """
        if self.feedback and session_id:
            self.feedback.record_token_usage(session_id, token_id, usage_score)
    
    def complete_feedback_session(self, session_id: str, quality: float = 1.0):
        """
        Complete a feedback session after generation.
        
        Args:
            session_id: Feedback session ID from retrieve_relevant()
            quality: Overall generation quality (0-1)
        """
        if self.feedback and session_id:
            self.feedback.complete_session(session_id, quality)
            self.stats['feedback_sessions'] += 1
    
    def retrieve_by_emotion(self, emotion: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve memories tagged with a specific emotion.
        
        Args:
            emotion: Emotion to filter by (e.g., "JOY", "ANGER")
            top_k: Maximum results
            
        Returns:
            List of matching memory dicts
        """
        try:
            all_tokens = self.latent_store.get_all_tokens()
            
            matching = []
            for token_id, data in all_tokens.items():
                meta = data.get('metadata', {})
                if meta.get('emotion', '').upper() == emotion.upper():
                    matching.append({
                        'token_id': token_id,
                        'content': meta.get('user_msg_preview', 'Unknown'),
                        'emotion': meta.get('emotion'),
                        'intensity': meta.get('intensity', 0),
                        'timestamp': meta.get('timestamp', 0)
                    })
            
            # Sort by intensity
            matching.sort(key=lambda x: x['intensity'], reverse=True)
            
            return matching[:top_k]
            
        except Exception as e:
            logger.error(f"Emotional retrieval failed: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict:
        """Get comprehensive KAIROS statistics."""
        store_stats = self.latent_store.get_stats()
        
        store_lats = self.stats['store_latencies']
        retrieve_lats = self.stats['retrieve_latencies']
        sim_scores = self.stats['similarity_scores']
        
        return {
            # Core metrics
            'stores': self.stats['stores'],
            'retrievals': self.stats['retrievals'],
            'total_documents': store_stats['total_tokens'],
            'total_chars_stored': self.stats['total_chars_stored'],
            'total_bytes_stored': self.stats['total_bytes_stored'],
            
            # Latency metrics
            'store_latency_ms': {
                'mean': float(np.mean(store_lats)) if store_lats else 0,
                'p50': float(np.percentile(store_lats, 50)) if store_lats else 0,
                'p95': float(np.percentile(store_lats, 95)) if store_lats else 0,
            },
            'retrieve_latency_ms': {
                'mean': float(np.mean(retrieve_lats)) if retrieve_lats else 0,
                'p50': float(np.percentile(retrieve_lats, 50)) if retrieve_lats else 0,
                'p95': float(np.percentile(retrieve_lats, 95)) if retrieve_lats else 0,
            },
            
            # Quality metrics
            'avg_similarity_score': float(np.mean(sim_scores)) if sim_scores else 0,
            'feedback_sessions': self.stats['feedback_sessions'],
            
            # Storage
            'storage_size_mb': store_stats['total_size_mb'],
            'backend': store_stats.get('backend', 'unknown'),
            'multidim_enabled': self.use_multidim,
            'feedback_enabled': self.enable_feedback
        }
    
    def clear_all(self):
        """Clear all stored memories."""
        self.latent_store.clear()
        self.working_memory.clear()
        self.short_term_memory.clear()
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'compression_failures': 0,
            'feedback_sessions': 0,
            'store_latencies': [],
            'retrieve_latencies': [],
            'total_chars_stored': 0,
            'total_bytes_stored': 0,
            'similarity_scores': []
        }
        logger.info("All memories cleared")
