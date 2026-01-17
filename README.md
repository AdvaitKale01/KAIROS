# KAIROS Memory
**Knowledge-Augmented Intelligent Retrieval and Organizational System**

> "RAG gives agents a library. KAIROS gives them a life."

KAIROS is not just a vector database. It is a **Cognitive Memory Layer** designed to give AI agents **true episodic memory** and **emotional context**, going far beyond static RAG retrieval.

While traditional systems stuff prompts with raw text chunks, KAIROS mimics human memory: compressing experiences into latent patterns, storing the *semantic and emotional* essence of interactions, and learning which memories actually matter over time.

## Why KAIROS?

*   🧠 **True Episodic Memory**: Stores "User said X, I replied Y" as a coherent event, not fragmented text chunks.
*   ❤️ **Emotional Intelligence**: Encodes **8-dimensional emotional state** (Joy, Trust, Fear, etc.) into every memory, allowing agents to recall *how* a conversation felt.
*   ⏳ **Dynamic Learning**: Unlike static RAG, KAIROS has a feedback loop. Using a memory strengthens it; ignoring it causes it to fade.
*   🚀 **Hardware Agnostic**: Runs on **NVIDIA CUDA**, **Apple Silicon (MLX)**, **Intel IPEX**, or pure CPU.

## Core Features

- **Latent Compression**: Compresses lengthy exchanges into compact 408d vectors (384d Semantic + 8d Emotional + 16d Temporal).
- **Pluggable Backends**: Use efficient NumPy (default) or scale with ChromaDB.
- **Generator Feedback**: System creates a closed loop where the agent's usage optimizes future retrieval.
- **Persistent & Portable**: Memories are automatically saved to disk and technically portable between models.

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

## LangChain & LangGraph Support

KAIROS provides native adapters for LangChain and LangGraph in `kairos.integrations`.

### Installation

```bash
pip install -e ".[langchain]"
```

### 1. Agents & Chat Memory
Use `KairosChatMemory` to give any LangChain agent persistent, episodic memory.

```python
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from kairos import KAIROSMemory
from kairos.integrations import KairosChatMemory

# Initialize Core
k_mem = KAIROSMemory(embedding_model=...)

# Wrap for LangChain
memory = KairosChatMemory(
    memory=k_mem,
    memory_key="history",  # Key used in PromptTemplate
    k_retrieval=3          # How many past memories to inject
)

# Run Chain
chain = ConversationChain(llm=OpenAI(), memory=memory)
chain.predict(input="My name is Alice.") 
# -> Saved to KAIROS automatically
```

### 2. Custom RAG Pipelines
Use `KairosRetriever` or `KairosVectorStore` for custom LCEL chains.

```python
from kairos.integrations import KairosRetriever

retriever = KairosRetriever(memory=k_mem, k=5)
docs = retriever.invoke("What does Alice like?")
```

## Migrating from Standard RAG

Moving from a traditional "Chunk-and-Retrieve" RAG system (like `vectorstore.add_documents(chunks)`) requires a shift in thinking. KAIROS is an **Episodic Memory System**, not just a document store.

### The Conceptual Shift

| Traditional RAG | KAIROS Memory |
|-----------------|---------------|
| **Unit of Storage** | Arbitrary Text Chunk (e.g. 500 chars) | **Interaction Event** (User query + Assistant response) |
| **Context Window** | Fills with raw text snippets | Fills with **Synthesized Memories** |
| **Retrieval Query** | Matches exact text keywords/semantics | Matches **User Intent** & **Semantic Meaning** |
| **Vector Size** | Varies (e.g. 1536d) | **Compressed (408d)**: 384d Semantic + 8d Emotion + 16d Time |

### Step-by-Step Migration Guide

#### Phase 1: Data Strategy
Don't just dump your Wiki/Docs into KAIROS.
*   **Old Way**: "Here is a 5000-word PDF split into 10 chunks."
*   **KAIROS Way**: Convert that PDF into **QA Pairs** or **Simulated Dialogues**.
    *   *Why?* KAIROS optimizes for retrieving *answers* based on *questions*, not just matching text patterns.

#### Phase 2: Code Migration

**Scenario**: You have a list of text chunks you want to store.

**❌ Old Approach (LangChain/Chroma direct):**
```python
texts = ["Chunk 1 content...", "Chunk 2 content..."]
vectorstore.add_texts(texts)
```

**✅ KAIROS Approach:**
Treat the content as something an "Assistant" would say in response to a hypothetical "User" query (or a generic prompt).

```python
texts = ["Chunk 1 content...", "Chunk 2 content..."]

for text in texts:
    memory.consolidate_exchange(
        user_msg="Context about [Topic]",  # Provide a relevant hook/trigger
        assistant_msg=text,                # The actual content goes here
        importance=0.5
    )
```

#### Phase 3: Retrieval Updates

**❌ Old Approach:**
```python
docs = vectorstore.similarity_search("query")
prompt = f"Context: {docs[0].page_content}..."
```

**✅ KAIROS Approach:**
KAIROS handles the "Context" construction for you if you use the LangChain adapter.

```python
# Just map the memory variables
result = chain.invoke({"input": "query"})
# The PromptTemplate receives "relevant_memories" automatically
```

### Why this is better?
1.  **Noise Reduction**: You retrieve complete thoughts/answers, not cut-off sentences.
2.  **Temporal Grounding**: KAIROS knows *when* information was added.
3.  **Self-Cleaning**: The feedback loop (if enabled) will "forget" chunks that are retrieved but never useful for generating good answers.

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

AGPL-3.0 - See [LICENSE](LICENSE) for details.
