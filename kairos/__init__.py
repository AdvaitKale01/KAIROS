"""
KAIROS - Knowledge-Augmented Intelligent Retrieval and Organizational System

A standalone, importable memory system for GenAI applications.
Provides CLaRa-inspired latent compression with efficient vector retrieval.

Quick Start:
    from kairos import KAIROSMemory
    
    # Default NumPy backend
    memory = KAIROSMemory()
    
    # ChromaDB backend
    memory = KAIROSMemory(backend="chroma")
    
    # Custom backend
    memory = KAIROSMemory(backend=MyCustomStore())
"""

from .memory import KAIROSMemory
from .compressor import LatentCompressor, MultiDimensionalEncoder
from .encoder import QueryEncoder
from .feedback import GeneratorFeedback

# Storage backends
from .backends import BaseStore, NumpyStore, get_backend
from .backends import CHROMA_AVAILABLE
if CHROMA_AVAILABLE:
    from .backends import ChromaStore

__version__ = "1.1.0"

__all__ = [
    # Main class
    'KAIROSMemory',
    
    # Core components
    'LatentCompressor',
    'MultiDimensionalEncoder',
    'QueryEncoder',
    'GeneratorFeedback',
    
    # Storage backends
    'BaseStore',
    'NumpyStore',
    'ChromaStore',
    'get_backend',
    'CHROMA_AVAILABLE',
]
