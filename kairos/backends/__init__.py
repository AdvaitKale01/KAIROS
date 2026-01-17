"""
KAIROS Storage Backends

Pluggable storage backends for KAIROS memory.
Default: NumpyStore (FAISS-like, no dependencies)
Optional: ChromaStore (requires chromadb)
Custom: Implement BaseStore interface
"""
from typing import Union, Optional

from .base import BaseStore
from .numpy_store import NumpyStore

# Optional ChromaDB backend
try:
    from .chroma import ChromaStore
    CHROMA_AVAILABLE = True
except ImportError:
    ChromaStore = None
    CHROMA_AVAILABLE = False


def get_backend(
    backend: Union[str, BaseStore] = "numpy",
    storage_path: str = "./data/kairos",
    **kwargs
) -> BaseStore:
    """
    Factory function to get a storage backend.
    
    Args:
        backend: Backend name ("numpy", "chroma") or BaseStore instance
        storage_path: Base path for storage
        **kwargs: Additional backend-specific arguments
        
    Returns:
        BaseStore instance
        
    Examples:
        store = get_backend("numpy")           # Default
        store = get_backend("chroma")          # ChromaDB
        store = get_backend(MyCustomStore())   # Custom
    """
    if isinstance(backend, BaseStore):
        return backend
    
    backend_name = backend.lower()
    
    if backend_name == "numpy":
        return NumpyStore(storage_path=f"{storage_path}/latent", **kwargs)
    
    elif backend_name == "chroma":
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb\n"
                "Or use the default NumPy backend: KAIROSMemory(backend='numpy')"
            )
        return ChromaStore(storage_path=f"{storage_path}/chroma", **kwargs)
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: 'numpy', 'chroma', or pass a BaseStore instance."
        )


__all__ = [
    'BaseStore',
    'NumpyStore', 
    'ChromaStore',
    'get_backend',
    'CHROMA_AVAILABLE',
]
