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
# From this directory
pip install -e .

# With ChromaDB support
pip install -e ".[chroma]"
```

## Quick Start

```python
from kairos import KAIROSMemory

# Initialize with default NumPy backend
memory = KAIROSMemory()

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
└─────────────────────────────────────────────────────┘
```

## License

MIT
