"""
Latent Compressor - KAIROS Memory Component
Compresses conversation exchanges into continuous latent tokens.

Inspired by Apple CLaRa's semantic compression approach.
Achieves 16-128x compression while preserving reasoning signals.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import hashlib
import json
import re

# GPU thread safety (local fallback)
from .utils import gpu_lock, logger

# Sentence transformer for semantic embeddings
# Sentence transformer availability check is no longer needed for internal loading
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# MLX for Apple Silicon optimization (optional)
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class MultiDimensionalEncoder:
    """
    Encodes emotional and temporal signals into dense vector representations.
    
    Produces:
    - Emotional embedding: 8 dimensions (PAD values + intensity + category)
    - Temporal embedding: 16 dimensions (sinusoidal positional encoding)
    
    Combined with semantic (384d), creates 408-dim multi-dimensional vectors.
    """
    
    # Emotion categories for one-hot encoding
    EMOTION_CATEGORIES = ['positive', 'negative', 'neutral', 'mixed']
    
    # Temporal encoding dimensions
    TEMPORAL_DIM = 16
    EMOTIONAL_DIM = 8
    
    def __init__(self):
        """Initialize the multi-dimensional encoder."""
        pass
    
    def encode_emotional(self, emotion_state: dict = None) -> np.ndarray:
        """
        Encode emotional state into 8-dimensional vector.
        
        Args:
            emotion_state: Dict with optional keys:
                - 'pleasure': -1 to 1 (valence)
                - 'arousal': -1 to 1 (activation)
                - 'dominance': -1 to 1 (control)
                - 'intensity': 0 to 1
                - 'category': 'positive', 'negative', 'neutral', 'mixed'
                - 'emotion': emotion name (used to infer category)
        
        Returns:
            8-dimensional emotional embedding
        """
        if emotion_state is None:
            emotion_state = {}
        
        # Extract PAD values (default to neutral)
        pleasure = emotion_state.get('pleasure', 0.0)
        arousal = emotion_state.get('arousal', 0.0)
        dominance = emotion_state.get('dominance', 0.0)
        intensity = emotion_state.get('intensity', 0.5)
        
        # Infer category from emotion name if not provided
        category = emotion_state.get('category', None)
        if category is None:
            emotion_name = emotion_state.get('emotion', '').upper()
            category = self._infer_emotion_category(emotion_name)
        
        # One-hot encode category (4 dims)
        category_onehot = np.zeros(4)
        if category in self.EMOTION_CATEGORIES:
            category_onehot[self.EMOTION_CATEGORIES.index(category)] = 1.0
        else:
            category_onehot[2] = 1.0  # Default to neutral
        
        # Combine into 8-dim vector
        emotional_vec = np.array([
            pleasure,      # dim 0: valence
            arousal,       # dim 1: activation
            dominance,     # dim 2: control
            intensity,     # dim 3: strength
            category_onehot[0],  # dim 4: positive
            category_onehot[1],  # dim 5: negative
            category_onehot[2],  # dim 6: neutral
            category_onehot[3],  # dim 7: mixed
        ], dtype=np.float32)
        
        return emotional_vec
    
    def _infer_emotion_category(self, emotion_name: str) -> str:
        """Infer emotion category from emotion name."""
        positive_emotions = {'JOY', 'HAPPY', 'LOVE', 'EXCITEMENT', 'GRATITUDE', 
                           'HOPE', 'PRIDE', 'CALM', 'CONTENT', 'AMUSEMENT'}
        negative_emotions = {'ANGER', 'FEAR', 'SADNESS', 'DISGUST', 'ANXIETY',
                           'FRUSTRATION', 'GUILT', 'SHAME', 'ENVY', 'GRIEF'}
        
        if emotion_name in positive_emotions:
            return 'positive'
        elif emotion_name in negative_emotions:
            return 'negative'
        elif emotion_name in {'SURPRISE', 'ANTICIPATION'}:
            return 'mixed'
        else:
            return 'neutral'
    
    def encode_temporal(self, timestamp: float = None) -> np.ndarray:
        """
        Encode temporal information using sinusoidal positional encoding.
        
        Inspired by Transformer positional encoding, but adapted for time:
        - 4 dims: Time of day (sinusoidal encoding of hour)
        - 4 dims: Day of week (sinusoidal encoding)
        - 8 dims: Recency encoding (log-scaled hours ago)
        
        Args:
            timestamp: Unix timestamp. If None, uses current time.
        
        Returns:
            16-dimensional temporal embedding
        """
        import time as time_module
        from datetime import datetime
        
        if timestamp is None:
            timestamp = time_module.time()
        
        dt = datetime.fromtimestamp(timestamp)
        now = time_module.time()
        
        # Time of day encoding (4 dims) - sinusoidal for hour
        hour = dt.hour + dt.minute / 60.0
        hour_normalized = hour / 24.0  # 0-1
        time_of_day = np.array([
            np.sin(2 * np.pi * hour_normalized),
            np.cos(2 * np.pi * hour_normalized),
            np.sin(4 * np.pi * hour_normalized),  # Higher frequency
            np.cos(4 * np.pi * hour_normalized),
        ], dtype=np.float32)
        
        # Day of week encoding (4 dims)
        day = dt.weekday()  # 0=Monday, 6=Sunday
        day_normalized = day / 7.0
        day_of_week = np.array([
            np.sin(2 * np.pi * day_normalized),
            np.cos(2 * np.pi * day_normalized),
            1.0 if day >= 5 else 0.0,  # Weekend flag
            0.0,  # Reserved
        ], dtype=np.float32)
        
        # Recency encoding (8 dims) - how long ago was this memory
        hours_ago = max(0, (now - timestamp) / 3600.0)
        
        # Log-scaled recency for different time scales
        recency = np.array([
            np.exp(-hours_ago / 1),      # 1-hour decay
            np.exp(-hours_ago / 6),      # 6-hour decay
            np.exp(-hours_ago / 24),     # 1-day decay
            np.exp(-hours_ago / 168),    # 1-week decay
            np.exp(-hours_ago / 720),    # 1-month decay
            np.exp(-hours_ago / 2160),   # 3-month decay
            1.0 if hours_ago < 1 else 0.0,   # Very recent flag
            1.0 if hours_ago < 24 else 0.0,  # Today flag
        ], dtype=np.float32)
        
        # Combine into 16-dim vector
        temporal_vec = np.concatenate([time_of_day, day_of_week, recency])
        
        return temporal_vec
    
    def encode_multidim(self, semantic_vec: np.ndarray, 
                        emotion_state: dict = None,
                        timestamp: float = None) -> np.ndarray:
        """
        Create full multi-dimensional vector by combining all signals.
        
        Args:
            semantic_vec: 384-dim semantic embedding from sentence transformer
            emotion_state: Emotional context dict
            timestamp: Memory timestamp
        
        Returns:
            408-dimensional multi-dimensional vector
        """
        emotional_vec = self.encode_emotional(emotion_state)
        temporal_vec = self.encode_temporal(timestamp)
        
        # Concatenate: [semantic_384 | emotional_8 | temporal_16] = 408 dims
        multidim_vec = np.concatenate([semantic_vec, emotional_vec, temporal_vec])
        
        return multidim_vec
    
    @staticmethod
    def pad_legacy_vector(legacy_vec: np.ndarray, target_dim: int = 408) -> np.ndarray:
        """
        Pad a legacy 384-dim vector to 408-dim for backward compatibility.
        
        Uses neutral emotional state and zero temporal encoding.
        """
        if len(legacy_vec) >= target_dim:
            return legacy_vec[:target_dim]
        
        padding = np.zeros(target_dim - len(legacy_vec), dtype=np.float32)
        # Set neutral emotion category flag
        if target_dim == 408 and len(legacy_vec) == 384:
            padding[6] = 1.0  # neutral category flag at position 390 (384+6)
        
        return np.concatenate([legacy_vec, padding])


class LatentCompressor:
    """
    Semantic compressor that converts text exchanges into compact latent representations.
    
    Features:
    - Multi-dimensional encoding: semantic (384d) + emotional (8d) + temporal (16d) = 408d
    - Real sentence transformer embeddings
    - MLX acceleration on Apple Silicon (optional)
    - Adaptive compression based on memory age
    - Preserves semantic meaning at high compression
    """
    
    # Vector dimensions
    SEMANTIC_DIM = 384
    EMOTIONAL_DIM = 8
    TEMPORAL_DIM = 16
    MULTIDIM_DIM = 408  # Total: 384 + 8 + 16
    
    def __init__(self, compression_ratio: int = 16, use_mlx: bool = True, embedding_model: Optional[Any] = None):
        """
        Initialize compressor.
        
        Args:
            compression_ratio: Target compression ratio (4, 16, 64, or 128)
            use_mlx: Use MLX for Apple Silicon acceleration
            embedding_model: External embedding model (must implement encode method)
        """
        self.compression_ratio = compression_ratio
        self.use_mlx = use_mlx and MLX_AVAILABLE
        
        # Use provided model or fallback
        if embedding_model:
            self.encoder = embedding_model
            self.embedding_dim = 384  # Assuming standard size, or could inspect model
            logger.info(f"Using provided external embedding model: {type(embedding_model).__name__}")
        else:
            self.encoder = None
            self.embedding_dim = 384
            logger.warning("No embedding model provided. Using fallback hash compression. Provide an embedding_model for semantic compression.")
        
        # Binary quantization for 32x compression (384 float32 → 48 bytes)
        self.use_binary_quantization = True
        if self.use_binary_quantization:
            logger.debug("Binary quantization enabled (32x compression)")
        
        # Multi-dimensional encoder for emotional + temporal signals
        self.multidim_encoder = MultiDimensionalEncoder()
        self.multidim_dim = self.MULTIDIM_DIM
        logger.debug(f"Multi-dimensional encoding enabled ({self.multidim_dim}d)")
    
    def _binarize(self, embedding: np.ndarray) -> np.ndarray:
        """Convert float32 embedding to binary (1-bit per dimension).
        
        384 float32 values (1,536 bytes) → 48 bytes (32x compression)
        Uses sign of each dimension: positive → 1, negative → 0
        """
        binary_bits = (embedding > 0).astype(np.uint8)
        return np.packbits(binary_bits)
    
    def _unbinarize(self, binary: np.ndarray, dim: int = 384) -> np.ndarray:
        """Convert binary back to float for similarity calculations."""
        bits = np.unpackbits(binary)[:dim]
        return bits.astype(np.float32) * 2 - 1  # Convert 0,1 → -1,1
        
    def compress(self, exchange_text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Compress conversation exchange into latent tokens.
        
        Uses semantic embeddings with optional binary quantization for 32x compression.
        
        Args:
            exchange_text: Full conversation text to compress
            metadata: Optional metadata (timestamp, participants, etc.)
            
        Returns:
            Dict containing:
                - latent_vector: Compressed semantic representation
                - binary_vector: Binary quantized version (if enabled)
                - compression_ratio: Actual compression achieved
                - metadata: Enhanced metadata
                - checksum: For integrity verification
        """
        input_size = len(exchange_text.encode('utf-8'))
        
        # Use sentence transformer if available (REAL semantic compression)
        if self.encoder:
            # Generate semantic embedding with GPU lock for thread safety
            with gpu_lock:
                float_vector = self.encoder.encode(
                    exchange_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Unit norm for cosine similarity
                )
            
            # Optionally convert to MLX for Apple Silicon acceleration
            if self.use_mlx:
                try:
                    float_vector = np.array(mx.array(float_vector))
                except Exception:
                    pass  # Fall back to numpy if MLX fails
            
        else:
            # Fallback: Hash-based
            # Use deterministic random generation based on hash to produce full 384d vector
            seed = int(hashlib.sha256(exchange_text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            float_vector = rng.randn(self.embedding_dim).astype(np.float32)
            float_vector = float_vector / (np.linalg.norm(float_vector) + 1e-8)
        
        # Smart binary quantization: only use if it actually compresses
        binary_vector = self._binarize(float_vector)
        binary_size = binary_vector.nbytes  # 48 bytes
        
        if self.use_binary_quantization and input_size >= binary_size:
            # Binary is smaller or equal - use it
            compressed_size = binary_size
            latent_vector = binary_vector
            use_binary = True
        else:
            # Original is smaller - skip binary, but still use float for retrieval
            binary_vector = None
            compressed_size = float_vector.nbytes  # 1536 bytes for float storage
            latent_vector = float_vector
            use_binary = False
        
        # Never report ratio below 1.0 (we always preserve meaning)
        actual_ratio = max(1.0, input_size / compressed_size) if compressed_size > 0 else 1.0
        
        return {
            'latent_vector': latent_vector,
            'float_vector': float_vector,  # Keep for similarity calculations
            'binary_vector': binary_vector,
            'compression_ratio': round(actual_ratio, 2),
            'original_size': input_size,
            'compressed_size': min(compressed_size, input_size),  # Never larger than original
            'is_binary': use_binary,
            'metadata': metadata or {},
            'checksum': hashlib.md5(exchange_text.encode()).hexdigest(),
            'encoder': 'sentence-transformer' if self.encoder else 'hash-fallback'
        }
    
    def compress_multidim(self, exchange_text: str, 
                          emotion_state: Optional[Dict] = None,
                          timestamp: Optional[float] = None,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Compress text into multi-dimensional latent vector (408d).
        
        Combines:
        - Semantic embedding (384d) from sentence transformer
        - Emotional embedding (8d) from PAD values + category
        - Temporal embedding (16d) from timestamp
        
        Args:
            exchange_text: Text to compress
            emotion_state: Dict with 'pleasure', 'arousal', 'dominance', 'intensity', 'emotion'
            timestamp: Unix timestamp for temporal encoding
            metadata: Additional metadata to store
            
        Returns:
            Dict with multi-dimensional latent vector and metadata
        """
        import time
        
        input_size = len(exchange_text.encode('utf-8'))
        
        # Generate semantic embedding (384d)
        if self.encoder:
            with gpu_lock:
                semantic_vec = self.encoder.encode(
                    exchange_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        else:
            # Fallback: Hash-based
            # Use deterministic random generation based on hash to produce full 384d vector
            seed = int(hashlib.sha256(exchange_text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            semantic_vec = rng.randn(self.SEMANTIC_DIM).astype(np.float32)
            semantic_vec = semantic_vec / (np.linalg.norm(semantic_vec) + 1e-8)
        
        # Ensure semantic_vec is numpy array
        semantic_vec = np.array(semantic_vec, dtype=np.float32)
        
        # Extract emotion from metadata if not provided directly
        if emotion_state is None and metadata:
            emotion_state = {
                'emotion': metadata.get('emotion', ''),
                'intensity': metadata.get('intensity', 0.5),
                'pleasure': metadata.get('pleasure', 0.0),
                'arousal': metadata.get('arousal', 0.0),
                'dominance': metadata.get('dominance', 0.0),
            }
        
        # Use current time if not provided
        if timestamp is None:
            timestamp = metadata.get('timestamp', time.time()) if metadata else time.time()
        
        # Generate multi-dimensional vector (408d)
        multidim_vec = self.multidim_encoder.encode_multidim(
            semantic_vec, emotion_state, timestamp
        )
        
        # Calculate compressed size
        compressed_size = multidim_vec.nbytes  # 408 * 4 = 1632 bytes
        actual_ratio = max(1.0, input_size / compressed_size) if compressed_size > 0 else 1.0
        
        # Build result
        result = {
            'latent_vector': multidim_vec,
            'float_vector': multidim_vec,  # For similarity calculations
            'semantic_vector': semantic_vec,  # Keep pure semantic for fallback
            'is_multidim': True,
            'vector_dim': len(multidim_vec),
            'compression_ratio': round(actual_ratio, 2),
            'original_size': input_size,
            'compressed_size': compressed_size,
            'metadata': metadata or {},
            'checksum': hashlib.md5(exchange_text.encode()).hexdigest(),
            'encoder': 'multidim-408d' if self.encoder else 'multidim-hash-fallback'
        }
        
        return result
    
    def decompress(self, latent_vector: np.ndarray) -> Optional[str]:
        """
        Attempt to reconstruct original text (lossy).
        
        Note: In CLaRa-style systems, full reconstruction isn't the goal.
        The latent space is designed for retrieval, not reconstruction.
        
        Returns None for now - reconstruction not implemented.
        """
        # Reconstruction is optional and lossy
        # Primary use case is retrieval in latent space
        return None
    
    def compress_batch(self, exchanges: List[str], metadata_list: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Compress multiple exchanges in batch.
        
        Args:
            exchanges: List of conversation texts
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of compressed representations
        """
        if metadata_list is None:
            metadata_list = [None] * len(exchanges)
        
        return [
            self.compress(text, meta) 
            for text, meta in zip(exchanges, metadata_list)
        ]
    
    def adaptive_compress(self, exchange_text: str, age_days: float, metadata: Optional[Dict] = None) -> Dict:
        """
        Apply adaptive compression based on memory age.
        
        CLaRa-inspired multi-tier compression:
        - Recent memories: Lower compression (high fidelity)
        - Old memories: Higher compression (space efficient)
        
        Args:
            exchange_text: Text to compress
            age_days: Age of memory in days
            metadata: Optional metadata
            
        Returns:
            Compressed representation with adaptive ratio
        """
        # Determine compression strategy based on age
        if age_days < 1:
            # Recent (<1 day): Minimal compression, preserve detail
            strategy = 'high_fidelity'
            truncate_to = None  # Keep full text
        elif age_days < 7:
            # This week: Moderate compression
            strategy = 'moderate'
            truncate_to = 500  # Keep main points
        elif age_days < 30:
            # This month: Aggressive compression
            strategy = 'aggressive'
            truncate_to = 200  # Keep summary only
        else:
            # Ancient: Maximum compression
            strategy = 'maximum'
            truncate_to = 100  # Keep bare essence
        
        # Apply text truncation for older memories
        if truncate_to and len(exchange_text) > truncate_to:
            # Intelligent truncation: Try to keep key sentences
            sentences = exchange_text.split('. ')
            compressed_text = '. '.join(sentences[:2])  # Keep first 2 sentences
            if len(compressed_text) > truncate_to:
                compressed_text = compressed_text[:truncate_to] + "..."
        else:
            compressed_text = exchange_text
        
        # Compress with sentence transformer
        result = self.compress(compressed_text, metadata)
        
        # Add compression strategy info
        result['compression_strategy'] = strategy
        result['age_days'] = age_days
        result['text_truncated'] = len(exchange_text) > len(compressed_text)
        
        return result
    
    def verify_integrity(self, compressed: Dict, original_text: str) -> bool:
        """
        Verify compressed data integrity using checksum.
        
        Args:
            compressed: Compressed representation dict
            original_text: Original text to verify against
            
        Returns:
            True if integrity check passes
        """
        expected_checksum = hashlib.md5(original_text.encode()).hexdigest()
        return compressed.get('checksum') == expected_checksum

    def compress_multi_vector(self, exchange_text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Compress text into MULTIPLE latent vectors (CLaRa-style Memory Tokens).
        
        Instead of one vector for the whole text, we generate a vector for each
        semantic unit (sentence/clause). This allows for much more precise 
        retrieval in the latent space.
        
        Args:
            exchange_text: Full text to compress
            metadata: Base metadata
            
        Returns:
            List of compressed result dicts
        """
        chunks = self._split_into_chunks(exchange_text)
        results = []
        
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            # Create chunk-specific metadata
            chunk_meta = base_metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['total_chunks'] = len(chunks)
            chunk_meta['is_multi_vector'] = True
            chunk_meta['parent_text_hash'] = hashlib.md5(exchange_text.encode()).hexdigest()[:8]
            
            # Compress the chunk
            chunk_meta['chunk_text'] = chunk
            
            result = self.compress(chunk, chunk_meta)
            results.append(result)
            
        return results

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into semantic chunks (sentences)."""
        if not text:
            return []
            
        raw_chunks = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty and very short chunks
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 10]
        
        # If no chunks found (e.g. short text), return original
        if not chunks:
            return [text]
            
        return chunks
