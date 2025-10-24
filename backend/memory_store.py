import os
import time
import uuid
import chromadb
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "llm_memories"

# --- Pydantic Schemas ---
class MemoryMetadata(BaseModel):
    """Metadata schema for a memory entry."""
    conversation_id: str
    message_type: str = Field(..., description="user or assistant")
    timestamp: float = Field(default_factory=time.time)
    token_count: int = 0
    source: str = "chat"

class StoredMemory(BaseModel):
    """Schema for a memory object before storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: List[float]
    metadata: MemoryMetadata

# --- ChromaDB Client Setup ---
# Initialize ChromaDB client and collection
try:
    # Persistent client to save data to disk
    CLIENT = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    COLLECTION = CLIENT.get_or_create_collection(name=COLLECTION_NAME)
    print(f"ChromaDB initialized. Data saved to: {CHROMA_PERSIST_DIR}")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    CLIENT = None
    COLLECTION = None

# --- Core Operations ---

def add_memory(
    text: str,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> str:
    """Stores a single memory entry (user message or AI response)."""
    if COLLECTION is None:
        raise RuntimeError("ChromaDB is not initialized.")

    memory_id = str(uuid.uuid4())
    
    # ChromaDB requires separate lists for IDs, embeddings, texts, and metadatas
    COLLECTION.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )
    return memory_id

def search_memories(
    query_embedding: List[float],
    top_k: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Searches for the most similar memories to a given query embedding.
    Returns ChromaDB query result dictionary.
    """
    if COLLECTION is None:
        raise RuntimeError("ChromaDB is not initialized.")
        
    try:
        results = COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'] # Ensure distance is included
        )
        return results
    except Exception as e:
        print(f"Error searching memories: {e}")
        return None

def clear_memories() -> bool:
    """Removes all data from the memory collection."""
    
    # FIX: Declare global COLLECTION immediately.
    global COLLECTION 
    
    if COLLECTION is None:
        print("ChromaDB is not initialized, cannot clear.")
        return False
        
    try:
        # Recreate the collection to clear all data
        CLIENT.delete_collection(name=COLLECTION_NAME)
        COLLECTION = CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        print("Memory store cleared successfully.")
        return True
    except Exception as e:
        print(f"Error clearing memories: {e}")
        return False
        
def get_all_memories() -> List[Dict[str, Any]]:
    """Retrieves all documents and metadata from the collection."""
    if COLLECTION is None:
        return []
    
    try:
        # Get all memories (limit can be adjusted for very large dbs)
        count = COLLECTION.count()
        results = COLLECTION.get(
            ids=COLLECTION.get()['ids'], # A way to get all IDs
            include=['documents', 'metadatas']
        )
        
        memories = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            memories.append({"text": doc, "metadata": meta})
            
        return memories
    except Exception as e:
        print(f"Error retrieving all memories: {e}")
        return []

def get_stats() -> Dict[str, Any]:
    """Returns statistics about the memory store."""
    if COLLECTION is None:
        return {"count": 0, "status": "ChromaDB not initialized"}
    
    try:
        return {
            "count": COLLECTION.count(),
            "status": "Operational",
            "persist_dir": CHROMA_PERSIST_DIR
        }
    except Exception as e:
        return {"count": 0, "status": f"Error getting count: {e}"}