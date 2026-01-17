"""
KAIROS Memory Unit Tests
Tests for the standalone KAIROS memory package.
"""
import sys
import os
import pytest
import numpy as np
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos import KAIROSMemory, LatentCompressor, NumpyStore, QueryEncoder


class TestLatentCompressor:
    """Test the LatentCompressor component."""
    
    @pytest.fixture
    def compressor(self):
        return LatentCompressor(compression_ratio=16)
    
    def test_compress_produces_valid_output(self, compressor):
        """Test that compression produces valid output structure."""
        exchange = """User: What's the weather like today?
Assistant: I don't have access to real-time weather data, but I can help!"""
        
        compressed = compressor.compress(exchange)
        
        assert 'latent_vector' in compressed
        assert 'original_size' in compressed
        assert 'compressed_size' in compressed
        assert 'compression_ratio' in compressed
        assert compressed['original_size'] > 0
        assert compressed['compression_ratio'] >= 1.0
    
    def test_compress_vector_shape(self, compressor):
        """Test that compressed vector has correct shape."""
        exchange = "User: Hello\nAssistant: Hi there!"
        
        compressed = compressor.compress(exchange)
        
        assert compressed['latent_vector'] is not None
        assert len(compressed['latent_vector'].shape) == 1
    
    def test_multidim_produces_408d(self, compressor):
        """Test multi-dimensional compression produces 408d vectors."""
        exchange = "User: I'm happy today!\nAssistant: That's wonderful!"
        
        compressed = compressor.compress_multidim(exchange)
        
        assert compressed['latent_vector'].shape == (408,)
        assert compressed['is_multidim'] == True


class TestNumpyStore:
    """Test the NumpyStore component."""
    
    @pytest.fixture
    def store(self):
        # Use temp directory for isolation
        temp_dir = tempfile.mkdtemp()
        store = NumpyStore(storage_path=os.path.join(temp_dir, "test_store"))
        yield store
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_vector(self):
        return np.random.randn(384).astype(np.float32)
    
    def test_store_and_retrieve(self, store, sample_vector):
        """Test storing and retrieving tokens."""
        metadata = {'type': 'test', 'importance': 0.5}
        
        success = store.store("test_token_1", sample_vector, metadata)
        
        assert success
        assert "test_token_1" in store.tokens
    
    def test_retrieve_similar(self, store, sample_vector):
        """Test similarity-based retrieval."""
        store.store("token_1", sample_vector, {'importance': 0.5})
        store.store("token_2", sample_vector * 0.9, {'importance': 0.5})
        
        results = store.retrieve_similar(sample_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0][1] >= results[1][1]  # Sorted by similarity
    
    def test_get_stats(self, store, sample_vector):
        """Test stats retrieval."""
        store.store("token_1", sample_vector, {})
        
        stats = store.get_stats()
        
        assert stats['total_tokens'] == 1
        assert stats['total_size_bytes'] > 0
    
    def test_delete_token(self, store, sample_vector):
        """Test token deletion."""
        store.store("token_to_delete", sample_vector, {})
        assert "token_to_delete" in store.tokens
        
        success = store.delete("token_to_delete")
        
        assert success
        assert "token_to_delete" not in store.tokens


class TestQueryEncoder:
    """Test the QueryEncoder component."""
    
    @pytest.fixture
    def encoder(self):
        return QueryEncoder()
    
    def test_encode_query_shape(self, encoder):
        """Test that query encoding produces correct shape."""
        query = "What did we talk about yesterday?"
        
        vec = encoder.encode_query(query)
        
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
    
    def test_encode_with_intent(self, encoder):
        """Test intent-based encoding."""
        query = "Tell me about the weather discussion"
        
        vec = encoder.encode_with_intent(query, "episodic")
        
        assert vec.shape == (384,)
    
    def test_multidim_query_produces_408d(self, encoder):
        """Test multi-dimensional query encoding."""
        query = "What made me happy yesterday?"
        
        vec = encoder.encode_multidim_query(query)
        
        assert vec.shape == (408,)


class TestKAIROSIntegration:
    """Test full KAIROS integration."""
    
    @pytest.fixture
    def kairos(self):
        temp_dir = tempfile.mkdtemp()
        memory = KAIROSMemory(
            storage_path=os.path.join(temp_dir, "test_kairos"),
            use_multidim=True,
            enable_feedback=True
        )
        yield memory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_consolidate_exchange(self, kairos):
        """Test storing an exchange."""
        user_msg = "Can you help me understand how memory works?"
        assistant_msg = "Memory systems have working memory, short-term, and long-term storage."
        
        token_id = kairos.consolidate_exchange(user_msg, assistant_msg)
        
        assert token_id is not None
        stats = kairos.get_stats()
        assert stats['stores'] >= 1
    
    def test_retrieve_relevant(self, kairos):
        """Test retrieval after storing."""
        kairos.consolidate_exchange(
            "I love pizza with pepperoni.",
            "That sounds delicious!"
        )
        
        results, session_id = kairos.retrieve_relevant("What food do I like?", top_k=3)
        
        assert isinstance(results, list)
    
    def test_working_memory(self, kairos):
        """Test working memory operations."""
        kairos.add_to_working("user", "Hello")
        kairos.add_to_working("assistant", "Hi there!")
        
        working = kairos.get_working_memory()
        
        assert len(working) == 2
        assert working[0]['role'] == 'user'
    
    def test_clear_all(self, kairos):
        """Test clearing all memories."""
        kairos.consolidate_exchange("Test", "Response")
        kairos.add_to_working("user", "Message")
        
        kairos.clear_all()
        
        stats = kairos.get_stats()
        assert stats['total_documents'] == 0
        assert len(kairos.get_working_memory()) == 0


def main():
    """Run tests via pytest."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    main()
