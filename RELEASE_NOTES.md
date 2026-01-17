# Release Notes

## v1.1.0 - The "External Brain" Update

### 🚀 Major Changes
- **External Embedding Models**: `sentence-transformers` is no longer a hard dependency. You can now inject any embedding model (SentenceTransformer, OpenAI, etc.) into `KAIROSMemory`.
- **Hardware Acceleration Support**: Explicit documentation and examples for running on **NVIDIA GPU (CUDA)**, **Intel CPU (IPEX)**, and **Apple Silicon (MLX)**.
- **License Change**: Switched to **AGPL-3.0** to protect open-source contributions while preventing closed-source commercial exploitation.

### ✨ New Features
- **LangChain Integration**: First-class support for LangChain & LangGraph via `kairos.integrations`.
    - `KairosChatMemory`: Drop-in replacement for conversation memory.
    - `KairosVectorStore`: Use KAIROS as a standard vector database.
    - `KairosRetriever`: Lightweight retriever for LCEL chains.
- **New Examples**:
    - `examples/langchain_agent.py`: Full agent with memory example.
    - `examples/cuda_usage.py`: GPU usage.
    - `examples/chroma_usage.py`: ChromaDB backend.
- **Dependency Groups**: Added `[langchain]`, `[transformers]`, `[chroma]`, and `[mlx]`.

### 🐛 Bug Fixes
- Fixed hard-coded dependency on `sentence-transformers` inside `QueryEncoder` and `LatentCompressor`.
- Improved fallback mechanism to be dimensionally compatible (384d) even without a real model.

### 📚 Documentation
- Added `MIGRATION_GUIDE.md`.
- Updated `README.md` with new installation instructions and hardware guides.
