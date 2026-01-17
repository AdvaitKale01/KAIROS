"""
KAIROS + LangChain Integration Example

Demonstrates how to use KAIROS as the memory backend for a LangChain ConversationChain.
Requires: langchain, langchain-community
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain.chains import ConversationChain
    from langchain_core.prompts import PromptTemplate
    from langchain.llms.fake import FakeListLLM  # For demo purposes
    from kairos import KAIROSMemory
    from kairos.integrations import KairosChatMemory
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install langchain langchain-community sentence-transformers")
    sys.exit(1)

def main():
    print("=" * 60)
    print("KAIROS + LangChain Integration")
    print("=" * 60)
    
    # 1. Initialize KAIROS Core
    print("Initializing KAIROS Memory...")
    # Using a real model for meaningful retrieval
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    kairos_core = KAIROSMemory(
        storage_path="./data/langchain_demo",
        embedding_model=model,
        use_multidim=True
    )
    
    # Pre-populate some memories
    print("Pre-populating memory...")
    kairos_core.consolidate_exchange(
        "My favorite color is blue.", 
        "I'll remember that your favorite color is blue.", 
        importance=0.8
    )
    kairos_core.consolidate_exchange(
        "I live in San Francisco.", 
        "Noted, you live in SF.", 
        importance=0.9
    )
    
    # 2. Create LangChain Adapter
    print("Creating KairosChatMemory adapter...")
    memory_adapter = KairosChatMemory(
        memory=kairos_core,
        memory_key="history",  # Variable to inject into prompt
        k_retrieval=2
    )
    
    # 3. Setup Dummy LLM (FakeListLLM)
    # in a real app, use OpenAI() or ChatAnthropic()
    responses = [
        "Aha! Based on my memory, your favorite color is blue.",
        "San Francisco is a great city with cool weather."
    ]
    llm = FakeListLLM(responses=responses)
    
    # 4. Create Conversation Chain
    # We define a custom prompt that uses the "history" variable where KAIROS injects context
    template = """
    The following is a conversation between a human and an AI. 
    The AI uses its long-term KAIROS memory to answer questions.
    
    {history}
    
    Human: {input}
    AI:"""
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    
    chain = ConversationChain(
        llm=llm, 
        memory=memory_adapter,
        prompt=prompt,
        verbose=True
    )
    
    # 5. Run Interactions
    print("\n--- Interaction 1 ---")
    question1 = "What is my favorite color?"
    print(f"Human: {question1}")
    response1 = chain.predict(input=question1)
    print(f"AI: {response1}")
    
    print("\n--- Interaction 2 ---")
    question2 = "Where do I live?"
    print(f"Human: {question2}")
    response2 = chain.predict(input=question2)
    print(f"AI: {response2}")
    
    print("\n✅ Integration successful!")

if __name__ == "__main__":
    main()
