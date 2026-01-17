# LangChain/LangGraph Compatibility Roadmap

Currently, KAIROS is **not** natively compatible with LangChain or LangGraph because it does not implement their standard interfaces (`VectorStore`, `BaseMemory`, `Retriever`).

To make it compatible, we need to implement the following wrappers/adapters.

## 1. VectorStore Adapter
LangChain expects a `VectorStore` class for document storage and retrieval.

- [ ] **Create `KairosVectorStore` class**:
    - Inherit from `langchain_core.vectorstores.VectorStore`.
    - Implement `add_texts(texts, metadatas)`: Maps to `memory.consolidate_exchange`.
    - Implement `similarity_search(query, k)`: Maps to `memory.retrieve_relevant`.
    - **Challenge**: KAIROS expects (User, Assistant) pairs, while LangChain often sends single text chunks. We will need a strategy to handle single chunks (e.g., treating them as "User" messages with empty "Assistant" responses or vice-versa).

## 2. Memory Adapter
LangGraph and LangChain Agents use `BaseChatMemory` to manage conversation history.

- [ ] **Create `KairosChatMemory` class**:
    - Inherit from `langchain.memory.chat_memory.BaseChatMemory`.
    - Implement `save_context(inputs, outputs)`:
        - Extract user input and model output.
        - Call `memory.consolidate_exchange(user_msg, assistant_msg)`.
    - Implement `load_memory_variables(inputs)`:
        - Call `memory.retrieve_relevant` using the current input as the query.
        - Format the retrieved memories into a prompt-friendly string (e.g., "Relevant Past Context: ...").
    - Implement `clear()`: Maps to `memory.clear_all()`.

## 3. Retriever Adapter
For RAG chains (`RetrievalQA`), a simple `Retriever` interface is often used.

- [ ] **Create `KairosRetriever` class**:
    - Inherit from `langchain_core.retrievers.BaseRetriever`.
    - Implement `_get_relevant_documents(query)`:
        - Call `memory.retrieve_relevant(query)`.
        - Convert KAIROS results (dicts) into LangChain `Document` objects.

## 4. LangGraph Checkpointer (Advanced)
LangGraph uses "Checkpointers" to save valid graph states.

- [ ] **Create `KairosCheckpointer`**:
    - This would treat KAIROS as a state database for long-running agent threads.
    - *Note*: This is complex and might be overkill. The `Memory Adapter` approach is usually sufficient for LangGraph agents.

## Implementation Plan

1.  **Add `langchain-core` dependency** to `pyproject.toml` (optional group).
2.  **Create `kairos/integrations/langchain.py`** containing the classes above.
3.  **Create `examples/langchain_agent.py`** to demonstrate an agent using KAIROS memory.
