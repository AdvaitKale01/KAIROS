"""
KAIROS Memory - NVIDIA CUDA Example

Demonstrates how to use KAIROS with a SentenceTransformer model on NVIDIA GPUs.
Requires: PyTorch with CUDA support and sentence-transformers.
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kairos import KAIROSMemory


def main():
    print("=" * 60)
    print("KAIROS Memory - NVIDIA CUDA Example")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  CUDA not available. Running on CPU.")
        print("   To enable CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        device = "cpu"

    # Load embedding model
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model on {device}...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print("✅ Embedding model loaded")
    except ImportError:
        print("❌ sentence-transformers not installed.")
        print("   Install with: pip install sentence-transformers")
        return

    # Initialize KAIROS with the GPU-accelerated model
    memory = KAIROSMemory(
        storage_path="./cuda_example_memory",
        use_multidim=True,
        enable_feedback=True,
        embedding_model=embedding_model
    )

    # Store some data
    print("\n📝 Storing conversation exchanges...")
    exchanges = [
        ("What is deep learning?", "Deep learning is a subset of machine learning using neural networks.", 0.9),
        ("How does GPU acceleration help?", "GPUs parallelize matrix operations, significantly speeding up training.", 0.85),
        ("What is CUDA?", "CUDA is NVIDIA's parallel computing platform for GPU programming.", 0.8),
    ]

    for user_msg, assistant_msg, importance in exchanges:
        token_id = memory.consolidate_exchange(user_msg, assistant_msg, importance)
        print(f"  ✓ Stored: '{user_msg[:40]}...' (ID: {token_id})")

    # Retrieve data
    print("\n🔍 Retrieving relevant memories...")
    query = "Tell me about GPU computing"
    print(f"  Query: '{query}'")
    results, session_id = memory.retrieve_relevant(query, top_k=3)

    if results:
        for i, mem in enumerate(results, 1):
            print(f"    {i}. (sim={mem['similarity']:.2f}) {mem['content'][:60]}...")
    else:
        print("    No results found")

    # Show stats
    stats = memory.get_stats()
    print(f"\n📊 Stats:")
    print(f"  • Backend: {stats['backend']}")
    print(f"  • Total documents: {stats['total_documents']}")
    print(f"  • Device: {device}")

    print("\n✅ CUDA example complete!")


if __name__ == "__main__":
    main()
