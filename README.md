# KAIROS Memory

**Knowledge-Augmented Intelligent Retrieval and Organizational System**

A standalone, importable memory system for GenAI applications. Provides CLaRa-inspired latent compression with efficient vector retrieval.

## Features

- 🧠 **Multi-dimensional encoding** - Semantic (384d) + Emotional (8d) + Temporal (16d) = 408d vectors
- ⚡ **Pluggable backends** - NumPy (default), ChromaDB, or custom
- 🔄 **Generator feedback loop** - Learns from usage patterns to improve retrieval
- 💾 **Persistent storage** - Memories are automatically saved to disk
- 🍎 **Apple Silicon optimized** - Optional MLX acceleration on M-series chips

## Installation

```bash
# Minimal installation (no semantic model included)
pip install -e .

# With optional semantic model support (recommends sentence-transformers)
pip install -e ".[transformers]"

# With ChromaDB support
pip install -e ".[chroma]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

```python
from kairos import KAIROSMemory
from sentence_transformers import SentenceTransformer

# 1. Load your own embedding model (or any class with an encode() method)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Initialize KAIROS with the model
memory = KAIROSMemory(embedding_model=embedding_model)

# Store a conversation exchange
memory.consolidate_exchange(
    user_msg="What is quantum computing?",
    assistant_msg="Quantum computing uses qubits...",
    importance=0.8
)

# Retrieve relevant memories
results, session_id = memory.retrieve_relevant("Tell me about quantum")

for mem in results:
    print(f"Similarity: {mem['similarity']:.2f}")
    print(f"Content: {mem['content']}")
```

## Storage Backends

### NumPy (Default)
FAISS-like vector storage using pure NumPy. No external dependencies.

```python
memory = KAIROSMemory()  # Default
memory = KAIROSMemory(backend="numpy")  # Explicit
```

### ChromaDB
Scalable vector database with built-in persistence.

```bash
pip install chromadb
```

```python
memory = KAIROSMemory(backend="chroma")
```

### Custom Backend
Implement `BaseStore` for your own storage (Pinecone, Weaviate, etc.):

```python
from kairos import BaseStore, KAIROSMemory

class PineconeStore(BaseStore):
    def store(self, token_id, vector, metadata, float_vector=None):
        # Your implementation
        pass
    
    def retrieve_similar(self, query_vector, top_k=5, filters=None):
        # Your implementation
        pass
    
    # ... implement other methods

memory = KAIROSMemory(backend=PineconeStore())
```

## API Reference

### KAIROSMemory

```python
memory = KAIROSMemory(
    storage_path="./data/kairos",  # Where to store data
    backend="numpy",                # "numpy", "chroma", or BaseStore
    use_multidim=True,              # 408d encoding (recommended)
    enable_feedback=True,           # Learn from usage patterns
)
```

| Method | Description |
|--------|-------------|
| `consolidate_exchange(user_msg, assistant_msg, importance)` | Store a conversation |
| `retrieve_relevant(query, top_k, intent)` | Find similar memories |
| `add_to_working(role, content)` | Add to working memory |
| `get_working_memory()` | Get current conversation |
| `retrieve_by_emotion(emotion, top_k)` | Filter by emotion tag |
| `get_stats()` | Get performance metrics |
| `clear_all()` | Clear all memories |

### BaseStore Interface

Required methods for custom backends:

| Method | Description |
|--------|-------------|
| `store(token_id, vector, metadata)` | Store a vector |
| `retrieve(token_id)` | Get by ID |
| `retrieve_similar(query_vector, top_k)` | Similarity search |
| `delete(token_id)` | Delete a vector |
| `get_stats()` | Storage statistics |
| `get_all_tokens()` | Get all vectors |
| `clear()` | Clear storage |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   KAIROSMemory                       │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Compressor  │  │  Backend    │  │   Encoder   │ │
│  │  (408d)     │──│ (Pluggable) │──│   (Query)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                          │                          │
│         ┌────────────────┼────────────────┐        │
│         │                │                │        │
│     NumpyStore      ChromaStore     CustomStore    │
│     (default)       (optional)       (yours)       │
│     NumpyStore      ChromaStore     CustomStore    │
│     (default)       (optional)       (yours)       │
└─────────────────────────────────────────────────────┘
```

## Embedding Models

KAIROS allows you to inject *any* embedding model that follows the `encode(text)` signature. This makes the package lightweight and flexible.

**Recommended:** `sentence-transformers/all-MiniLM-L6-v2` (fast, efficient, 384d).

If no model is provided, KAIROS falls back to a **deterministic hash encoder**. This is useful for testing or non-semantic exact matching, but **not recommended for production semantic search**.

```python
# Fallback usage (warning: no semantic understanding)
memory = KAIROSMemory(embedding_model=None)
```

## Hardware Acceleration

KAIROS supports multiple hardware backends depending on your platform:

### NVIDIA GPU (CUDA)

For NVIDIA GPUs, install PyTorch with CUDA support before loading your embedding model:

```bash
# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install sentence-transformers
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# Model will automatically use CUDA if available
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
memory = KAIROSMemory(embedding_model=model)
```

### Intel CPU (IPEX)

For optimized performance on Intel CPUs:

```bash
pip install intel-extension-for-pytorch
pip install sentence-transformers
```

```python
import intel_extension_for_pytorch as ipex
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# Apply IPEX optimizations
model = ipex.optimize(model)
memory = KAIROSMemory(embedding_model=model)
```

### Apple Silicon (MLX)

For M-series Macs, MLX acceleration is available:

```bash
pip install -e ".[mlx]"
```

MLX is used internally for vector operations when available.

## License

MIT
