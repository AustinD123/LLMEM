"""
Infinite Memory Chat System with LangChain and Groq
A RAG-based external memory layer that eliminates context window limitations
"""

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from datetime import datetime
from typing import List, Dict, Optional
import json
import os

class InfiniteMemoryChat:
    def __init__(
        self, 
        conversation_id: str,
        groq_api_key: str,
        model_name: str = "mixtral-8x7b-32768",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        recent_buffer_size: int = 5,
        retrieval_top_k: int = 5
    ):
        """
        Initialize the infinite memory chat system with LangChain and Groq.
        
        Args:
            conversation_id: Unique identifier for this conversation
            groq_api_key: Your Groq API key
            model_name: Groq model to use (mixtral-8x7b-32768, llama2-70b-4096, etc.)
            embedding_model: HuggingFace embedding model
            persist_directory: Where to store the vector database
            recent_buffer_size: Number of recent messages to always include
            retrieval_top_k: Number of relevant past messages to retrieve
        """
        self.conversation_id = conversation_id
        self.recent_buffer_size = recent_buffer_size
        self.retrieval_top_k = retrieval_top_k
        self.turn_counter = 0
        
        # Initialize Groq LLM
        print(f"🚀 Initializing Groq with model: {model_name}...")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.7
        )
        
        # Initialize embeddings
        print(f"🔧 Loading embedding model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or load ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=f"conversation_{conversation_id}",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Recent message buffer (in-memory)
        self.recent_buffer: List[Dict] = []
        
        print(f"✅ Infinite Memory Chat initialized for conversation: {conversation_id}")
    
    def store_exchange(
        self, 
        user_message: str, 
        assistant_message: str,
        metadata: Optional[Dict] = None
    ):
        """
        Store a conversation exchange in both vector DB and recent buffer.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            metadata: Optional additional metadata
        """
        self.turn_counter += 1
        
        # Create combined text for better retrieval
        combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
        
        # Prepare metadata
        doc_metadata = {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "turn_number": self.turn_counter,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "type": "conversation_turn"
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        # Create LangChain Document
        document = Document(
            page_content=combined_text,
            metadata=doc_metadata
        )
        
        # Store in ChromaDB using LangChain
        self.vectorstore.add_documents([document])
        
        # Update recent buffer
        self.recent_buffer.append({
            "user": user_message,
            "assistant": assistant_message,
            "turn": self.turn_counter,
            "timestamp": doc_metadata["timestamp"]
        })
        
        # Maintain buffer size
        if len(self.recent_buffer) > self.recent_buffer_size:
            self.recent_buffer.pop(0)
        
        print(f"✓ Stored turn {self.turn_counter}")
    
    def retrieve_relevant_context(self, query: str) -> Dict:
        """
        Retrieve relevant past conversations based on semantic similarity.
        
        Args:
            query: The current user query
            
        Returns:
            Dictionary with relevant_past and recent contexts
        """
        # Use LangChain's similarity search
        relevant_docs = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k,
            filter={"conversation_id": self.conversation_id}
        )
        
        # Parse results
        relevant_past = []
        for doc, score in relevant_docs:
            metadata = doc.metadata
            relevant_past.append({
                "user": metadata.get("user_message", ""),
                "assistant": metadata.get("assistant_message", ""),
                "turn": metadata.get("turn_number", 0),
                "timestamp": metadata.get("timestamp", ""),
                "similarity_score": float(score)
            })
        
        return {
            "relevant_past": relevant_past,
            "recent": self.recent_buffer
        }
    
    def _build_messages(self, context: Dict, user_message: str) -> List:
        """
        Build the message list with retrieved context and recent messages.
        
        Args:
            context: Retrieved context from memory
            user_message: Current user message
            
        Returns:
            List of LangChain message objects
        """
        messages = []
        
        # System message
        system_content = """You are a helpful AI assistant with access to the full conversation history.
You can remember everything discussed in this conversation, no matter how long ago.
Use the provided context to give informed, consistent responses that reference past discussions when relevant."""
        
        messages.append(SystemMessage(content=system_content))
        
        # Add relevant past context if available
        if context["relevant_past"]:
            context_text = "=== Relevant Past Context ===\n"
            for msg in context["relevant_past"]:
                time_str = msg['timestamp'][:10] if msg['timestamp'] else "unknown"
                context_text += f"\n[Turn {msg['turn']} - {time_str}]\n"
                context_text += f"User: {msg['user']}\n"
                context_text += f"Assistant: {msg['assistant']}\n"
            
            messages.append(SystemMessage(content=context_text))
        
        # Add recent conversation history
        if context["recent"]:
            for msg in context["recent"]:
                messages.append(HumanMessage(content=msg["user"]))
                messages.append(AIMessage(content=msg["assistant"]))
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    def chat(
        self, 
        user_message: str,
        temperature: Optional[float] = None,
        store_exchange: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Main chat function - handles the full pipeline.
        
        Args:
            user_message: The user's message
            temperature: Override default temperature
            store_exchange: Whether to store this exchange in memory
            verbose: Print retrieval information
            
        Returns:
            Assistant's response
        """
        if verbose:
            print(f"\n🔍 Retrieving relevant context...")
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(user_message)
        
        if verbose:
            print(f"✓ Found {len(context['relevant_past'])} relevant past exchanges")
            print(f"✓ Using {len(context['recent'])} recent messages")
        
        # Build messages
        messages = self._build_messages(context, user_message)
        
        # Call LLM with optional temperature override
        if verbose:
            print(f"🤖 Generating response with Groq...")
        
        if temperature is not None:
            response = self.llm.invoke(messages, temperature=temperature)
        else:
            response = self.llm.invoke(messages)
        
        assistant_message = response.content
        
        # Store the exchange
        if store_exchange:
            self.store_exchange(user_message, assistant_message)
        
        return assistant_message
    
    def stream_chat(self, user_message: str, store_exchange: bool = True):
        """
        Stream the response token by token (for real-time display).
        
        Args:
            user_message: The user's message
            store_exchange: Whether to store this exchange in memory
            
        Yields:
            Response tokens as they're generated
        """
        print(f"\n🔍 Retrieving relevant context...")
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(user_message)
        
        print(f"✓ Found {len(context['relevant_past'])} relevant past exchanges")
        print(f"✓ Streaming response...\n")
        
        # Build messages
        messages = self._build_messages(context, user_message)
        
        # Stream response
        full_response = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_response += token
            yield token
        
        # Store the exchange
        if store_exchange:
            self.store_exchange(user_message, full_response)
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about the conversation memory"""
        # Count documents in vectorstore
        try:
            all_docs = self.vectorstore.get()
            total_turns = len(all_docs['ids']) if all_docs['ids'] else 0
        except:
            total_turns = 0
        
        return {
            "conversation_id": self.conversation_id,
            "total_turns_stored": total_turns,
            "current_turn": self.turn_counter,
            "recent_buffer_size": len(self.recent_buffer),
            "recent_buffer": self.recent_buffer
        }
    
    def export_conversation(self, filepath: str):
        """Export full conversation history to JSON"""
        try:
            all_docs = self.vectorstore.get(
                where={"conversation_id": self.conversation_id}
            )
            
            conversation = []
            if all_docs['ids']:
                for i in range(len(all_docs['ids'])):
                    metadata = all_docs['metadatas'][i]
                    conversation.append({
                        "turn": metadata.get("turn_number", 0),
                        "timestamp": metadata.get("timestamp", ""),
                        "user": metadata.get("user_message", ""),
                        "assistant": metadata.get("assistant_message", "")
                    })
            
            # Sort by turn number
            conversation.sort(key=lambda x: x['turn'])
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Conversation exported to {filepath}")
        except Exception as e:
            print(f"❌ Error exporting conversation: {e}")
    
    def clear_memory(self):
        """Clear all stored conversation history"""
        try:
            # Get all document IDs for this conversation
            all_docs = self.vectorstore.get(
                where={"conversation_id": self.conversation_id}
            )
            
            if all_docs['ids']:
                self.vectorstore.delete(ids=all_docs['ids'])
            
            self.recent_buffer = []
            self.turn_counter = 0
            print(f"✓ Memory cleared for conversation: {self.conversation_id}")
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize
    chat = InfiniteMemoryChat(
        conversation_id="user_123_session_456",
        groq_api_key="",
        model_name="openai/gpt-oss-20b",
        recent_buffer_size=5,
        retrieval_top_k=3
    )
    
    # Simulate a conversation
    print("\n" + "="*70)
    print("Starting conversation with Groq + LangChain...")
    print("="*70)
    
    # Turn 1
    response = chat.chat("Hi! My name is Alex and I love Python programming.")
    print(f"\n👤 User: Hi! My name is Alex and I love Python programming.")
    print(f"🤖 Assistant: {response}")
    
    # Turn 2
    response = chat.chat("Can you recommend some good Python libraries for data science?")
    print(f"\n👤 User: Can you recommend some good Python libraries for data science?")
    print(f"🤖 Assistant: {response}")
    
    # Simulate many more turns...
    print(f"\n{'='*70}")
    print("Simulating conversation turns 3-14...")
    print(f"{'='*70}")
    
    for i in range(3, 15):
        response = chat.chat(f"This is turn {i}. Tell me something interesting about AI.", verbose=False)
    
    print("\n✓ Simulated 12 more turns")
    
    # Turn 15 - Reference something from turn 1
    print(f"\n{'='*70}")
    print("Testing long-term memory recall...")
    print(f"{'='*70}")
    
    response = chat.chat("What was my name again? And what did I say I love?")
    print(f"\n👤 User: What was my name again? And what did I say I love?")
    print(f"🤖 Assistant: {response}")
    
    # Get stats
    print("\n" + "="*70)
    stats = chat.get_conversation_stats()
    print("📊 Conversation Statistics:")
    print(f"   Total turns stored: {stats['total_turns_stored']}")
    print(f"   Current turn: {stats['current_turn']}")
    print(f"   Recent buffer: {stats['recent_buffer_size']} messages")
    
    # Export conversation
    print("\n" + "="*70)
    chat.export_conversation("conversation_history.json")