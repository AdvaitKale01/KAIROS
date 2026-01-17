"""
LangChain Integrations for KAIROS Memory
"""
import logging
from typing import Any, Dict, List, Optional, Iterable, Type

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema.messages import BaseMessage, get_buffer_string
except ImportError:
    # Allow import for documentation generation even if langchain is missing
    Embeddings = Any
    VectorStore = object
    Document = Any
    BaseRetriever = object
    BaseChatMemory = object
    BaseMessage = Any

from ..memory import KAIROSMemory

logger = logging.getLogger(__name__)


class KairosVectorStore(VectorStore):
    """
    LangChain VectorStore adapter for KAIROS.
    
    NOTE: KAIROS is designed for conversational exchanges (User+Assistant),
    while VectorStores often handle raw document chunks. 
    Using `add_texts` will store each text as a "User" message with empty "Assistant" response.
    """
    
    def __init__(self, memory: KAIROSMemory):
        self.memory = memory

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            # KAIROS expects exchanges. We treat text as user_msg.
            token_id = self.memory.consolidate_exchange(
                user_msg=text,
                assistant_msg="[Document Chunk]", # Placeholder
                importance=meta.get('importance', 0.5),
                metadata=meta
            )
            if token_id:
                ids.append(token_id)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        results, _ = self.memory.retrieve_relevant(query, top_k=k)
        
        docs = []
        for res in results:
            # Reconstruct content from metadata preview
            content = res['metadata'].get('user_msg_preview', '[Content Missing]')
            # Create a true copy of metadata to avoid mutating the original
            meta = res['metadata'].copy()
            meta['score'] = res['similarity']
            meta['token_id'] = res['token_id']
            
            docs.append(Document(page_content=content, metadata=meta))
            
        return docs
        
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "KairosVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        # KAIROS manages its own embeddings internally via embedding_model
        # We ignore the passed `embedding` object here, as KAIROS is self-contained.
        memory = KAIROSMemory(**kwargs)
        vectorstore = cls(memory)
        vectorstore.add_texts(texts, metadatas)
        return vectorstore


class KairosRetriever(BaseRetriever):
    """LangChain Retriever adapter for KAIROS."""
    
    memory: KAIROSMemory
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        results, _ = self.memory.retrieve_relevant(query, top_k=self.k)
        
        docs = []
        for res in results:
            content = res['metadata'].get('user_msg_preview') or \
                      res['metadata'].get('assistant_msg_preview') or \
                      "[Content Missing]"
            
            meta = res['metadata'].copy()
            meta['kairos_similarity'] = res['similarity']
            
            docs.append(Document(page_content=content, metadata=meta))
            
        return docs


class KairosChatMemory(BaseChatMemory):
    """
    LangChain Memory adapter for KAIROS.
    
    Automatically consolidates user/ai interactions into KAIROS 
    and retrieves relevant context for the prompt.
    """
    
    memory: KAIROSMemory
    memory_key: str = "history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    k_retrieval: int = 3
    
    def clear(self) -> None:
        """Clear memory contents."""
        self.memory.clear_all()

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""
        # Get the input text
        if self.input_key:
            prompt_input = inputs[self.input_key]
        else:
            # Fallback to finding the first string input if key not specified
            prompt_input = next(v for v in inputs.values() if isinstance(v, str))
            
        # 1. Retrieve relevant long-term memories
        relevant_memories, _ = self.memory.retrieve_relevant(prompt_input, top_k=self.k_retrieval)
        
        # 2. Get recent short-term/working memory
        working_mem = self.memory.get_working_memory()
        
        # 3. Format context string
        context_str = "Relevant Past Context:\n"
        for mem in relevant_memories:
            context_str += f"- {mem['content']} (similarity: {mem['similarity']:.2f})\n"
            
        context_str += "\nCurrent Conversation:\n"
        # Helper from langchain to format messages as buffer string
        # We verify if we have messages in working memory (custom KAIROS dicts)
        # and convert them if needed, but here we just construct string manually
        # to avoid complex dependencies on LangChain message classes.
        for item in working_mem:
             context_str += f"{item['role'].capitalize()}: {item['content']}\n"
             
        return {self.memory_key: context_str}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key:
            input_str = inputs[self.input_key]
        else:
             input_str = next(v for v in inputs.values() if isinstance(v, str))
             
        if self.output_key:
            output_str = outputs[self.output_key]
        else:
             output_str = next(v for v in outputs.values() if isinstance(v, str))
             
        # Add to working memory
        self.memory.add_to_working("user", input_str)
        self.memory.add_to_working("assistant", output_str)
        
        # Consolidate into long-term memory
        self.memory.consolidate_exchange(input_str, output_str, importance=0.5)
