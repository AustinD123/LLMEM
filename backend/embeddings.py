import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch
import numpy as np # Adding NumPy import for explicit conversion handling

# Load environment variables
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Initialize the model globally
try:
    EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding Model Loaded: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    EMBEDDER = None


def generate_embedding(text: str) -> list[float]:
    """
    Generates a vector embedding for a given text, explicitly forcing the output 
    to a stable NumPy array before converting it to the required Python list of floats
    for ChromaDB.
    """
    if EMBEDDER is None:
        raise RuntimeError("Embedding model failed to load.")

    # --- 💡 CRITICAL FIX FOR LOW SIMILARITY SCORE ---
    # We explicitly request the output as a NumPy array (convert_to_numpy=True).
    # This bypasses potential issues where the output might be an incompatible PyTorch 
    # tensor stored on a GPU/CPU when ChromaDB expects a simple list of floats.
    embedding = EMBEDDER.encode(
        text, 
        convert_to_numpy=True,  # Force NumPy array output for stability
        convert_to_tensor=False
    )

    # Convert the NumPy array (or any object with a .tolist() method) directly to a Python list
    if hasattr(embedding, 'tolist'):
        return embedding.tolist()
    
    # Fallback to ensure a list is returned
    return list(embedding)
