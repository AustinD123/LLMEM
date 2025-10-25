import os
import json
import time
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sse_starlette.sse import EventSourceResponse

# Import local modules
from . import memory_store
from .embeddings import generate_embedding
from .groq_client import generate_llm_response_stream, count_tokens

# --- FastAPI Setup ---
app = FastAPI(title="LLM Memory Layer API", version="1.0")
origins = [
    "http://localhost:3000", # Add your frontend's URL/port here
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Allow CORS for simple frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas for API Requests/Responses ---

class ChatRequest(BaseModel):
    """Schema for the incoming chat message."""
    user_query: str
    conversation_id: str
    # Chat history for standard conversation flow
    history: List[Dict[str, str]] 
    # Settings
    top_k: int = 5
    similarity_threshold: float = 0.6 # Cosine distance (1.0 - similarity)

class RetrievedMemory(BaseModel):
    """Schema for a retrieved memory item."""
    text: str
    similarity_score: float
    metadata: memory_store.MemoryMetadata

class ChatResponseChunk(BaseModel):
    """Schema for a streaming response chunk."""
    text: Optional[str] = None
    retrieved_memories: Optional[List[RetrievedMemory]] = None
    is_complete: bool = False
    error: Optional[str] = None

# --- Constants ---
CONVERSATION_ID_HEADER = "X-Conversation-ID"

# --- Helper Functions ---

# Cosine similarity is 1 - Cosine distance
def calculate_similarity(distance: float) -> float:
    """Converts ChromaDB's distance (which is 1-similarity) to similarity score."""
    return 1.0 - distance

def format_retrieved_memories(
    results: Dict[str, Any], 
    threshold: float
) -> List[RetrievedMemory]:
    """Formats raw ChromaDB results into a list of RetrievedMemory objects, filtering by threshold."""
    formatted_memories = []
    
    # ChromaDB returns lists of lists for batch queries, but here we only have one query
    if not results or not results['documents'] or not results['documents'][0]:
        return []

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity_score = calculate_similarity(dist)
        
        # Only include memories that meet the similarity threshold
        # Remember: a higher similarity_score (closer to 1.0) is better
        if similarity_score >= threshold:
            try:
                # Validate and parse metadata using Pydantic
                parsed_meta = memory_store.MemoryMetadata(**meta)
                formatted_memories.append(RetrievedMemory(
                    text=doc,
                    similarity_score=similarity_score,
                    metadata=parsed_meta
                ))
            except Exception as e:
                print(f"Error parsing metadata for memory: {doc}. Error: {e}")
                continue

    return formatted_memories

async def store_message(text: str, conversation_id: str, message_type: str):
    """Asynchronously stores a message and its embedding."""
    try:
        embedding = generate_embedding(text)
        token_count = count_tokens(text)
        
        metadata = memory_store.MemoryMetadata(
            conversation_id=conversation_id,
            message_type=message_type,
            token_count=token_count
        ).dict()
        
        memory_store.add_memory(text, embedding, metadata)
    except Exception as e:
        print(f"Error storing memory ({message_type}): {e}")
        # We allow the chat to continue even if memory storage fails

# --- FastAPI Endpoints ---

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "LLM Memory Layer is operational"}

@app.post("/chat", tags=["Core"])
async def chat_handler(request: ChatRequest):
    """
    Handles user chat query:
    1. Generates embedding for the query.
    2. Searches ChromaDB for relevant memories.
    3. Calls Groq LLM with a RAG-enhanced system prompt.
    4. Streams the response.
    5. Stores user query and LLM response as new memories.
    """
    user_query = request.user_query
    conv_id = request.conversation_id
    top_k = request.top_k
    threshold = request.similarity_threshold

    # --- 1. Store User Query ---
    # Store the user's message *before* retrieval for simple history tracking.
    # Note: We store it here, but it won't be retrieved for THIS query.
    await store_message(user_query, conv_id, "user")
    
    # --- 2. Retrieve Relevant Context ---
    try:
        query_embedding = generate_embedding(user_query)
        raw_results = memory_store.search_memories(
            query_embedding=query_embedding,
            top_k=top_k
        )
        # Filter and format memories
        retrieved_memories = format_retrieved_memories(raw_results, threshold)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        # Continue with a blank context if retrieval fails
        retrieved_memories = []

    # --- 3. Generate LLM Response (Streaming) ---
    async def event_generator():
        # First chunk contains retrieved context (metadata for the frontend)
        first_chunk = ChatResponseChunk(
            retrieved_memories=retrieved_memories
        )
        # FIX: Removed manual 'data: ' prefix. EventSourceResponse handles the prefixing.
        yield f"{first_chunk.json()}\n\n" 
        
        full_ai_response = ""
        
        # Generator from Groq client
        stream_generator = generate_llm_response_stream(
            user_query=user_query,
            retrieved_context=[m.dict() for m in retrieved_memories],
            conversation_history=request.history
        )
        
        try:
            for content_chunk in stream_generator:
                # Check for error message from the Groq client
                if content_chunk.startswith('{"error":'):
                    error_data = json.loads(content_chunk)
                    error_chunk = ChatResponseChunk(error=error_data.get('error', 'Unknown LLM Error'))
                    # FIX: Removed manual 'data: ' prefix.
                    yield f"{error_chunk.json()}\n\n" 
                    break # Stop streaming on error
                
                # Normal text chunk
                full_ai_response += content_chunk
                response_chunk = ChatResponseChunk(text=content_chunk)
                # FIX: Removed manual 'data: ' prefix.
                yield f"{response_chunk.json()}\n\n"
        
        except Exception as e:
            print(f"Streaming Error: {e}")
            error_chunk = ChatResponseChunk(error=f"Streaming failed: {e}")
            # FIX: Removed manual 'data: ' prefix.
            yield f"{error_chunk.json()}\n\n"
        
        finally:
            # --- 4. Store LLM Response ---
            if full_ai_response:
                await store_message(full_ai_response, conv_id, "assistant")
            
            # --- 5. Final Chunk ---
            final_chunk = ChatResponseChunk(is_complete=True)
            # FIX: Removed manual 'data: ' prefix.
            yield f"{final_chunk.json()}\n\n"

    # EventSourceResponse handles the SSE streaming protocol
    return EventSourceResponse(event_generator())

@app.get("/memories", response_model=List[Dict[str, Any]], tags=["Memory Management"])
async def list_memories():
    """Lists all stored memory entries."""
    return memory_store.get_all_memories()

@app.delete("/memories", tags=["Memory Management"])
async def clear_memories():
    """Deletes all stored memory entries."""
    if memory_store.clear_memories():
        return {"status": "success", "message": "All memories cleared."}
    raise HTTPException(status_code=500, detail="Failed to clear memories.")

@app.get("/stats", tags=["Memory Management"])
async def get_stats():
    """Returns memory database statistics."""
    return memory_store.get_stats()
