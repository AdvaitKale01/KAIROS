# Migration Guide: v1.0.x → v1.1.0

Version 1.1.0 introduces a major architectural change to make KAIROS lighter and more flexible. This guide helps you update your code.

## 🚨 Breaking Change: External Embedding Models

**Old Behavior (v1.0.x):**
KAIROS automatically downloaded and installed `sentence-transformers` and the `all-MiniLM-L6-v2` model internally.

**New Behavior (v1.1.0):**
KAIROS no longer depends on `sentence-transformers` by default. You must now:
1.  Install an embedding library (like `sentence-transformers`) explicitly.
2.  Load the model yourself.
3.  Pass the model to `KAIROSMemory`.

This allows you to use **any** model (OpenAI, Cohere, HuggingFace, etc.) or hardware custom configurations (CUDA, MPS, IPEX).

## How to Upgrade

### 1. Update Installation

**Old:**
```bash
pip install kairos-memory
```

**New:**
```bash
# To keep using sentence-transformers (recommended):
pip install "kairos-memory[transformers]"

# Or install manually:
pip install kairos-memory sentence-transformers
```

### 2. Update Code

**Old Code:**
```python
from kairos import KAIROSMemory

# Model was loaded automatically here
memory = KAIROSMemory() 
```

**New Code:**
```python
from kairos import KAIROSMemory
from sentence_transformers import SentenceTransformer

# 1. Load model explicitly
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Pass it to KAIROS
memory = KAIROSMemory(embedding_model=model)
```

## Why this change?

1.  **Lighter Core**: KAIROS is now tiny if you don't need local embeddings (e.g., if you plan to use an API).
2.  **Hardware Control**: You can now easily load models on GPU (CUDA), Apple Silicon (MPS), or Intel CPU (IPEX) before passing them to KAIROS.
3.  **Model Flexibility**: Use any model found on HuggingFace, or even your own custom class (just implement an `encode(text)` method).
