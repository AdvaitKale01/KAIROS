"""
KAIROS Cognitive Features Example with MLX
------------------------------------------
Demonstrates how to use KAIROS's cognitive features (Importance Filtering 
and Hyper-Compression) powerd by an on-device local LLM using MLX.

This example requires:
- pip install mlx-lm
- A Mac with Apple Silicon (M1/M2/M3/M4)

Usage:
    python cognitive_mlx_example.py
"""

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("This example requires 'mlx-lm'. Install with: pip install mlx-lm")

import sys
import os
import time

# Add parent directory to path to import kairos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kairos import KAIROSMemory

class MLXWrapper:
    """
    Simple wrapper to adapt mlx_lm for KAIROS.
    KAIROS expects an object with a generate_text(prompt, max_tokens) method.
    """
    def __init__(self, model_path="mlx-community/Llama-3.2-1B-Instruct-4bit"):
        print(f"Loading local model: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        print("Model loaded successfully.")
    
    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using the local LLM.
        """
        # Apply strict prompting for instruction following if needed
        # But KAIROS prompts are usually zero-shot instruction style.
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            verbose=False
        )
        return response

def main():
    if not MLX_AVAILABLE:
        return

    print("Initializing KAIROS with Cognitive Features...")
    
    # 1. Initialize Memory
    # using numpy backend for simplicity
    memory = KAIROSMemory(
        storage_path="./data/cognitive_example",
        backend="numpy",
        enable_feedback=True
    )
    
    # 2. Initialize Local LLM (MLX)
    # Using a small, fast 1B model for demonstration
    try:
        llm = MLXWrapper("mlx-community/Llama-3.2-1B-Instruct-4bit")
    except Exception as e:
        print(f"Failed to load MLX model: {e}")
        print("Please ensure you have network access to download the model.")
        return

    print("\n--- TEST 1: Importance Filtering ---")
    print("Storing a trivial message (should be filtered)...")
    
    # Trivial message
    result_1 = memory.consolidate_exchange(
        user_msg="hi testing 123",
        assistant_msg="hello there",
        importance=0.1,
        llm=llm  # <--- Passing the LLM enables filtering!
    )
    
    if result_1 is None:
        print("✅ SUCCESS: Trivial message was filtered out (not stored).")
    else:
        print(f"⚠️ NOTE: Message was stored (Token: {result_1}). LLM might have thought it important.")

    print("\nStoring an important message...")
    
    # Important message
    result_2 = memory.consolidate_exchange(
        user_msg="My favorite color is blue and I am allergic to peanuts.",
        assistant_msg="I have noted that preference and allergy.",
        importance=0.9,
        llm=llm
    )
    
    if result_2:
        print(f"✅ SUCCESS: Important message stored (Token: {result_2}).")
    else:
        print("❌ ERROR: Important message was filtered out!")

    print("\n--- TEST 2: Hyper-Compression ---")
    
    # Check what was actually stored
    if result_2:
        tokens = memory.latent_store.get_all_tokens()
        stored_data = tokens[result_2]
        metadata = stored_data['metadata']
        
        print("Peek at stored metadata:")
        print(f"- Default Preview: '{metadata.get('user_msg_preview')}'")
        print(f"- Is Hyper-Compressed: {metadata.get('is_hyper_compressed')}")
        
        if metadata.get('is_hyper_compressed'):
            print("✅ SUCCESS: Hyper-compressed summary generated!")
        else:
            print("⚠️ NOTE: Hyper-compression did not trigger (LLM error?).")

    # Clean up
    memory.clear_all()
    print("\nExample complete.")

if __name__ == "__main__":
    main()
