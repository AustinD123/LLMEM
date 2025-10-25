import React, { useState, useEffect, useRef } from 'react';

        const API_BASE_URL = 'http://127.0.0.1:8000';

        export default function App() {
          const [conversationId] = useState(() => 'conv-' + Date.now());
          const [conversationHistory, setConversationHistory] = useState([]);
          const [messages, setMessages] = useState([
            { type: 'system', text: "Welcome! I'm ready to chat. I'll store our conversation and use vector search to retrieve relevant context." }
          ]);
          const [input, setInput] = useState('');
          const [loading, setLoading] = useState(false);
          const [memoryCount, setMemoryCount] = useState(0);
          const [topK, setTopK] = useState(5);
          const [threshold, setThreshold] = useState(0.6);
          const [retrievedMemories, setRetrievedMemories] = useState([]);

          // NEW STATE for confirmation dialog and status alerts
          const [showConfirm, setShowConfirm] = useState(false);
          const [statusMessage, setStatusMessage] = useState({ text: '', type: '' }); // type: 'success' or 'error'

          const chatRef = useRef(null);

          // --- Utility Functions ---

          // Custom Status Alert component replaces native alert()
          const StatusAlert = ({ message, type }) => {
            if (!message) return null;

            const color = type === 'error' ? '#d9534f' : '#5cb85c';
            const bgColor = type === 'error' ? '#f2dede' : '#dff0d8';

            return (
              <div style={{
                position: 'fixed',
                top: '20px',
                right: '20px',
                padding: '10px 20px',
                backgroundColor: bgColor,
                color: color,
                border: `1px solid ${color}`,
                zIndex: 1000,
                fontFamily: 'monospace',
                fontSize: '14px',
              }}>
                {message}
              </div>
            );
          };

          // Custom Confirmation Dialog component replaces native confirm()
          const ConfirmationDialog = ({ onConfirm, onCancel, message }) => (
            <div style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              zIndex: 2000,
              fontFamily: 'monospace',
            }}>
              <div style={{
                backgroundColor: '#fff',
                padding: '30px',
                border: '2px solid #000',
                width: '350px',
                boxShadow: '8px 8px 0px #000',
              }}>
                <p style={{ margin: '0 0 20px 0', fontSize: '16px', fontWeight: 'bold' }}>
                  {message}
                </p>
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                  <button
                    onClick={onCancel}
                    style={{
                      padding: '8px 16px',
                      backgroundColor: '#ccc',
                      color: '#000',
                      border: '1px solid #000',
                      cursor: 'pointer',
                      fontFamily: 'monospace'
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={onConfirm}
                    style={{
                      padding: '8px 16px',
                      backgroundColor: '#000',
                      color: '#fff',
                      border: 'none',
                      cursor: 'pointer',
                      fontFamily: 'monospace'
                    }}
                  >
                    Yes, Clear All
                  </button>
                </div>
              </div>
            </div>
          );

          // --- Effects and Handlers ---

          useEffect(() => {
            fetchStats();
          }, []);

          useEffect(() => {
            if (chatRef.current) {
              chatRef.current.scrollTop = chatRef.current.scrollHeight;
            }
          }, [messages]);

          const fetchStats = async () => {
            try {
              const response = await fetch(`${API_BASE_URL}/stats`);
              const data = await response.json();
              setMemoryCount(data.count);
            } catch (error) {
              console.error('Error fetching stats:', error);
            }
          };

          // New handler for showing the confirmation dialog (replaces original clearAllMemories)
          const clearAllMemories = () => {
            setShowConfirm(true);
          };

          // Function executed when user confirms deletion in the dialog
          const handleConfirmClear = async () => {
            setShowConfirm(false); // Hide modal
            try {
              const response = await fetch(`${API_BASE_URL}/memories`, { method: 'DELETE' });
              if (response.ok) {
                setStatusMessage({ text: "All memories cleared successfully!", type: 'success' });
                setMessages([{ type: 'system', text: "Memories cleared. Starting fresh!" }]);
                setConversationHistory([]);
                setRetrievedMemories([]);
                fetchStats();
              } else {
                setStatusMessage({ text: "Failed to clear memories.", type: 'error' });
              }
            } catch (error) {
              console.error('Error clearing memories:', error);
              setStatusMessage({ text: `Network Error: ${error.message}`, type: 'error' });
            }
            // Automatically clear status message after a few seconds
            setTimeout(() => setStatusMessage({ text: '', type: '' }), 4000);
          };

          const handleCancelClear = () => {
            setShowConfirm(false);
          };


          const handleSubmit = async () => {
            if (!input.trim() || loading) return;

            const userQuery = input.trim();
            setInput('');
            setLoading(true);

            // Add the user message and assistant placeholder message in one atomic update
            setMessages(prev => [
              ...prev, 
              { type: 'user', text: userQuery },
              // We don't need the ID, as we will rely on the index (prev.length - 1) for updates
              { type: 'assistant', text: '' } 
            ]);

            const chatData = {
              user_query: userQuery,
              conversation_id: conversationId,
              history: conversationHistory,
              top_k: topK,
              similarity_threshold: threshold
            };

            let fullResponse = '';
            let memories = null;

            try {
              const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(chatData),
              });

              if (!response.body) {
                throw new Error("No response body received, possibly non-streaming response.");
              }

              const reader = response.body.getReader();
              const decoder = new TextDecoder("utf-8");
              let buffer = '';

              while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                let chunks = buffer.split(/\r?\n\r?\n/);
                buffer = chunks.pop();

                for (const chunk of chunks) {
                  const trimmed = chunk.trim();
                  if (!trimmed || trimmed === 'data:') continue;

                  if (trimmed.startsWith('data:')) {
                    try {
                      let lines = trimmed.split(/\r?\n/);
                      let jsonString = null;

                      for (let line of lines) {
                        line = line.trim();
                        if (line.startsWith('data:')) {
                          line = line.substring(5).trim();
                        }
                        if (line && (line.startsWith('{') || line.startsWith('['))) {
                          jsonString = line;
                          break;
                        }
                      }

                      if (!jsonString) continue;
                      const data = JSON.parse(jsonString);

                      if (data.is_complete) break;

                      if (data.text) {
                        fullResponse += data.text;
                        // FIX: Correctly target the last message (the assistant's placeholder)
                        setMessages(prev => {
                          const lastIndex = prev.length - 1;
                          if (lastIndex < 0) return prev;
                          return prev.map((msg, idx) => 
                            idx === lastIndex ? { ...msg, text: fullResponse } : msg
                          );
                        });
                      }

                      if (data.retrieved_memories && !memories) {
                        memories = data.retrieved_memories;
                        setRetrievedMemories(memories);
                      }
                    } catch (e) {
                      console.error("Parse error:", e);
                    }
                  }
                }
              }

              setConversationHistory(prev => [
                ...prev,
                { role: 'user', content: userQuery },
                { role: 'assistant', content: fullResponse }
              ]);
              
              fetchStats();
            } catch (error) {
              console.error("Error:", error);
              // FIX: Correctly target the last message (the assistant's placeholder) in case of an error
              setMessages(prev => {
                const lastIndex = prev.length - 1;
                if (lastIndex < 0) return prev;
                return prev.map((msg, idx) => 
                  idx === lastIndex ? { ...msg, text: `ERROR: ${error.message}` } : msg
                );
              });
            } finally {
              setLoading(false);
            }
          };

          const handleKeyPress = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          };

          return (
            <div style={{ 
              maxWidth: '1400px', 
              margin: '0 auto', 
              padding: '20px',
              fontFamily: 'monospace',
              backgroundColor: '#fff',
              minHeight: '100vh'
            }}>
              {/* Renders the custom status message */}
              <StatusAlert message={statusMessage.text} type={statusMessage.type} />

              <div style={{ 
                borderBottom: '2px solid #000', 
                paddingBottom: '15px', 
                marginBottom: '20px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <h1 style={{ margin: 0, fontSize: '24px' }}>LLM Memory Layer 🧠</h1>
                <button 
                  onClick={clearAllMemories} // Triggers the custom confirmation modal
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#000',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer',
                    fontFamily: 'monospace'
                  }}
                >
                  Clear All
                </button>
              </div>

              <div style={{ display: 'flex', gap: '20px' }}>
                {/* Settings */}
                <div style={{ 
                  width: '250px', 
                  border: '1px solid #000', 
                  padding: '15px',
                  height: 'fit-content'
                }}>
                  <h2 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>Settings</h2>
                  
                  <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                    Top K:
                  </label>
                  <input 
                    type="number" 
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    min="1"
                    style={{
                      width: '100%',
                      padding: '5px',
                      border: '1px solid #000',
                      marginBottom: '15px',
                      fontFamily: 'monospace'
                    }}
                  />

                  <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                    Threshold: {threshold.toFixed(2)}
                  </label>
                  <input 
                    type="range" 
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    min="0"
                    max="1"
                    step="0.05"
                    style={{ width: '100%', marginBottom: '20px' }}
                  />

                  <div style={{ borderTop: '1px solid #000', paddingTop: '15px' }}>
                    <h3 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Stats</h3>
                    <p style={{ margin: '0 0 10px 0', fontSize: '12px' }}>
                      Total Memories: <strong>{memoryCount}</strong>
                    </p>
                    <button 
                      onClick={fetchStats}
                      style={{
                        padding: '5px 10px',
                        backgroundColor: '#000',
                        color: '#fff',
                        border: 'none',
                        cursor: 'pointer',
                        width: '100%',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}
                    >
                      Refresh
                    </button>
                  </div>
                </div>

                {/* Chat Area */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                  <div 
                    ref={chatRef}
                    style={{
                      border: '1px solid #000',
                      padding: '15px',
                      height: '500px',
                      overflowY: 'auto',
                      marginBottom: '15px'
                    }}
                  >
                    {messages.map((msg, idx) => (
                      <div
                        key={idx}
                        style={{
                          padding: '10px',
                          marginBottom: '10px',
                          border: '1px solid #000',
                          backgroundColor: msg.type === 'user' ? '#000' : msg.type === 'system' ? '#f0f0f0' : '#fff',
                          color: msg.type === 'user' ? '#fff' : '#000',
                          maxWidth: msg.type === 'system' ? '100%' : '70%',
                          marginLeft: msg.type === 'user' ? 'auto' : '0',
                          marginRight: msg.type === 'user' ? '0' : 'auto',
                          fontSize: '14px',
                          fontFamily: 'monospace'
                        }}
                      >
                        {msg.text}
                      </div>
                    ))}
                  </div>

                  <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask me anything..."
                      disabled={loading}
                      style={{
                        flex: 1,
                        padding: '10px',
                        border: '1px solid #000',
                        fontFamily: 'monospace'
                      }}
                    />
                    <button
                      onClick={handleSubmit}
                      disabled={loading}
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#000',
                        color: '#fff',
                        border: 'none',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        fontFamily: 'monospace',
                        opacity: loading ? 0.5 : 1
                      }}
                    >
                      {loading ? '...' : 'Send'}
                    </button>
                  </div>
                </div>

                {/* Retrieved Memories */}
                <div style={{ 
                  width: '280px', 
                  border: '1px solid #000', 
                  padding: '15px',
                  height: 'fit-content',
                  maxHeight: '500px',
                  overflowY: 'auto'
                }}>
                  <h2 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>Retrieved Context</h2>
                  {retrievedMemories.length === 0 ? (
                    <p style={{ fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
                      No memories retrieved yet
                    </p>
                  ) : (
                    retrievedMemories.map((mem, idx) => (
                      <div
                        key={idx}
                        style={{
                          border: '1px solid #000',
                          padding: '10px',
                          marginBottom: '10px',
                          fontSize: '12px'
                        }}
                      >
                        <strong style={{ display: 'block', marginBottom: '5px' }}>
                          Sim: {mem.similarity_score.toFixed(4)} | {mem.metadata.message_type.toUpperCase()}
                        </strong>
                        <p style={{ margin: 0 }}>{mem.text}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Renders the custom confirmation dialog when state is true */}
              {showConfirm && (
                <ConfirmationDialog 
                  message="Are you sure you want to clear ALL memories? This cannot be undone."
                  onConfirm={handleConfirmClear}
                  onCancel={handleCancelClear}
                />
              )}
            </div>
          );
        }
