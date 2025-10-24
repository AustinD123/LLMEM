import os
import json
from groq import Groq
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in the environment.")

# Initialize the Groq client
try:
    GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    GROQ_CLIENT = None

# --- Core LLM Function ---

def generate_llm_response_stream(
    user_query: str,
    retrieved_context: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]], # For standard chat history
    max_tokens: int = 2048
) -> Generator[str, None, None]:
    """
    Generates a streaming response from the Groq API, incorporating
    retrieved memories as context.
    """
    if GROQ_CLIENT is None:
        yield json.dumps({"error": "Groq client is not initialized."}, ensure_ascii=False)
        return

    # 1. Construct the System Prompt with RAG Context
    context_text = "\n".join([f"- {m['text']} (Source: {m['metadata'].get('message_type')}, Sim: {m.get('similarity_score', 'N/A'):.4f})" 
                              for m in retrieved_context])
    
    system_prompt = (
        "You are a helpful and friendly assistant. Your goal is to answer the user's "
        "query based on the provided context and conversation history. "
        "If the context is relevant, use it. If not, answer based on your general knowledge. "
        "The retrieved context below comes from past messages. Use the past conversation "
        "to maintain flow and coherence.\n\n"
        "--- RETRIEVED CONTEXT ---\n"
        f"{context_text if context_text else 'No relevant memory found.'}\n"
        "-------------------------\n"
    )

    # 2. Build the Message List
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add standard conversation history (messages that precede the current one)
    # Note: conversation_history should contain 'role' and 'content' keys
    messages.extend(conversation_history)
    
    # Add the current user query
    messages.append({"role": "user", "content": user_query})

    # 3. Call Groq API
    try:
        stream = GROQ_CLIENT.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        print(f"Groq API Error: {e}")
        # Yield a structured error message to the frontend
        yield json.dumps({"error": f"LLM API Error: {e}"}, ensure_ascii=False)

# --- Utility to get token count (simplified for this project) ---
def count_tokens(text: str) -> int:
    """A very rough estimation of tokens (simple word count)."""
    return len(text.split())