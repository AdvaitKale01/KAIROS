import pytest
from unittest.mock import MagicMock
import sys
import os

# Add external/kairos to path so we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kairos.memory import KAIROSMemory

class MockLLM:
    def __init__(self):
        self.generate_text = MagicMock()

def test_llm_importance_filtering():
    """Test that low-importance messages are filtered out by LLM."""
    memory = KAIROSMemory(backend="numpy", storage_path="./test_data/llm_test")
    llm = MockLLM()
    
    # Mock LLM to say NO (not worth remembering)
    llm.generate_text.return_value = "Decision: NO"
    
    # Attempt to consolidate
    result = memory.consolidate_exchange(
        user_msg="hi",
        assistant_msg="hello",
        importance=0.1,
        llm=llm
    )
    
    # Should perform check
    llm.generate_text.assert_called()
    assert "Is it worth remembering" in llm.generate_text.call_args[0][0]
    
    # Should not store (mocked behavior dependent on implementation)
    # Until implemented, this might fail or pass depending on current logic (currently ignores llm)
    # Once implemented, if Result is None, it means it didn't store.
    # OR we can check memory stats.
    
    # Currently consolidate_exchange returns Token ID or None.
    # If filtered, it should likely return None or a special flag.
    # Let's assume for now valid behavior is returning None if filtered.
    assert result is None

def test_llm_hyper_compression():
    """Test that LLM summaries are used for preview."""
    memory = KAIROSMemory(backend="numpy", storage_path="./test_data/llm_test_2")
    llm = MockLLM()
    
    # Mock LLM to return a summary
    summary = "Fact: User likes pizza"
    llm.generate_text.side_effect = [
        "Decision: YES",  # First call: worth remembering?
        summary           # Second call: hyper-compression
    ]
    
    # Consolidate
    token_id = memory.consolidate_exchange(
        user_msg="I really love pizza with extra cheese",
        assistant_msg="That sounds delicious",
        importance=0.8,
        llm=llm
    )
    
    assert token_id is not None
    
    # Verify stored metadata has the summary
    # We need to peek into the store
    tokens = memory.latent_store.get_all_tokens()
    stored_data = tokens[token_id]
    
    assert stored_data['metadata']['user_msg_preview'] == summary.split("Fact: ")[1]
    assert stored_data['metadata']['is_hyper_compressed'] is True
