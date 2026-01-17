"""
KAIROS Memory - ChromaDB Backend Example

Demonstrates how to use KAIROS with the ChromaDB backend for persistent vector storage.
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos import KAIROSMemory

def main():
    print("=" * 60)
    print("KAIROS Memory - ChromaDB Usage Example")
    print("=" * 60)
    
    # Check if ChromaDB is installed
    try:
        import chromadb
        print("✅ ChromaDB is installed")
    except ImportError:
        print("❌ ChromaDB not installed. Please run: pip install chromadb")
        return
        
    # Initialize embedding model (optional but recommended for semantic search)
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    except ImportError:
        print("⚠️  sentence-transformers not installed. Using fallback hash encoding.")
        print("   To enable semantic search: pip install sentence-transformers")

    # Initialize memory system with ChromaDB backend
    # storage_path will be where ChromaDB persists its database
    storage_path = "./chroma_data"
    
    print(f"\nInitializing KAIROS with ChromaDB backend at '{storage_path}'...")
    memory = KAIROSMemory(
        storage_path=storage_path,
        backend="chroma",  # Specify ChromaDB backend
        use_multidim=True,
        enable_feedback=True,
        embedding_model=embedding_model
    )
    
    # Store some data
    print("\n📝 Storing conversation exchanges...")
    exchanges = [
        ("What is KAIROS?", "KAIROS is a memory system for GenAI applications.", 0.9),
        ("Does it support ChromaDB?", "Yes, KAIROS supports pluggable backends including ChromaDB.", 0.9),
        ("How does compression work?", "It uses latent semantic compression to reduce token usage.", 0.8),
    ]
    
    for user_msg, assistant_msg, importance in exchanges:
        token_id = memory.consolidate_exchange(user_msg, assistant_msg, importance)
        print(f"  ✓ Stored: '{user_msg[:40]}...' (ID: {token_id})")
        
    # Retrieve data
    print("\n🔍 Retrieving relevant memories...")
    queries = ["Tell me about KAIROS", "What backends are supported?"]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results, session_id = memory.retrieve_relevant(query, top_k=2)
        
        if results:
            for i, mem in enumerate(results, 1):
                print(f"    {i}. (sim={mem['similarity']:.2f}) {mem['content'][:60]}...")
        else:
            print("    No results found")

    # Show stats
    stats = memory.get_stats()
    print("\n📊 ChromaDB Stats:")
    print(f"  • Backend: {stats['backend']}")
    print(f"  • Total documents: {stats['total_documents']}")
    # print(f"  • Collection: {stats['collection_name']}") # Not exposed by wrapper
    
    print("\n✅ ChromaDB example complete!")

if __name__ == "__main__":
    main()
