import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch # We'll need torch to check the object type

# Load environment variables
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Initialize the model globally (NO CHANGE HERE)
try:
    # Set the device explicitly to 'cpu' or 'cuda' for testing.
    # We will let the library decide, but add conversion logic below.
    EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding Model Loaded: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    EMBEDDER = None


def generate_embedding(text: str) -> list[float]:
    """Generates a vector embedding for a given text, ensuring output is a Python list of floats."""
    if EMBEDDER is None:
        raise RuntimeError("Embedding model failed to load.")

    # Generate the embedding. We use the default setting (which often returns a NumPy array).
    # If the output is a tensor (as indicated by your error), the subsequent logic handles it.
    embedding = EMBEDDER.encode(
        text, 
        convert_to_numpy=False, 
        convert_to_tensor=False
    )

    # --- 💡 CRITICAL FIXES FOR CHROMADB COMPATIBILITY ---
    if isinstance(embedding, torch.Tensor):
        # 1. Move the tensor from CUDA device to CPU
        # 2. Convert the tensor to a NumPy array
        # 3. Convert the NumPy array to a Python list of floats (ChromaDB requirement)
        embedding = embedding.cpu().numpy().tolist()
    elif hasattr(embedding, 'tolist'):
        # If the output is a NumPy array, convert it to a list
        embedding = embedding.tolist()
    
    # ChromaDB expects a list of floats. The .tolist() method ensures this.
    return embedding