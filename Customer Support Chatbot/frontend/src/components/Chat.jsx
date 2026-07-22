import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, ShieldAlert, Clock, RefreshCw } from 'lucide-react';

const Chat = () => {
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chat_history');
    if (saved) {
      try { return JSON.parse(saved); } catch(e) {}
    }
    return [{ id: 1, sender: 'bot', text: 'Hello! I am your AI assistant. How can I help you today?', time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }];
  });

  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify(messages));
  }, [messages]);

  const clearChat = () => {
    if(window.confirm('Are you sure you want to clear the chat?')) {
      const initial = [{ id: 1, sender: 'bot', text: 'Hello! I am your AI assistant. How can I help you today?', time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }];
      setMessages(initial);
      localStorage.setItem('chat_history', JSON.stringify(initial));
    }
  };
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const newMsg = { id: Date.now(), sender: 'user', text: input, time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) };
    setMessages(prev => [...prev, newMsg]);
    setInput('');
    setIsTyping(true);

    const botMsgId = Date.now() + 1;
    const initialBotMsg = {
        id: botMsgId,
        sender: 'bot',
        text: '',
        confidence: null,
        time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
    };
    
    let isStreamStarted = false;

    try {
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'dev-secret-key'
        },
        body: JSON.stringify({ 
          user_id: 'dashboard-admin', 
          message: input,
          history: messages.map(m => ({ sender: m.sender, text: m.text }))
        })
      });

      if (!res.ok) throw new Error('Network error');
      
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      let done = false;
      let finalConfidence = null;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          if (!isStreamStarted) {
             setIsTyping(false);
             setMessages(prev => [...prev, initialBotMsg]);
             isStreamStarted = true;
          }
          
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.replace('data: ', '').trim();
              if (dataStr === '[DONE]') {
                done = true;
                break;
              }
              if (!dataStr) continue;
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'chunk') {
                  setMessages(prev => prev.map(m => 
                    m.id === botMsgId ? { ...m, text: m.text + data.content } : m
                  ));
                } else if (data.type === 'confidence') {
                  finalConfidence = data.content;
                } else if (data.type === 'error') {
                  setMessages(prev => prev.map(m => 
                    m.id === botMsgId ? { ...m, text: m.text + data.content, error: true } : m
                  ));
                }
              } catch(e) {}
            }
          }
        }
      }
      
      if (finalConfidence !== null) {
          setMessages(prev => prev.map(m => 
             m.id === botMsgId ? { ...m, confidence: finalConfidence } : m
          ));
      }
    } catch (err) {
      setMessages(prev => [...prev, { id: Date.now(), sender: 'bot', text: '⚠️ Error connecting to AI backend. Make sure the server is running.', error: true }]);
      setIsTyping(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '2rem', height: '100%' }}>
      {/* Main Chat Area */}
      <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '45px', height: '45px', borderRadius: '50%', background: 'var(--accent-gradient)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Bot color="white" size={24} />
          </div>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Test Environment</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--success)' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--success)' }}></div>
              RAG Pipeline Active
            </div>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <button onClick={clearChat} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
              <RefreshCw size={14} /> New Chat
            </button>
          </div>
        </div>

        <div ref={chatContainerRef} style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {messages.map(msg => (
            <div key={msg.id} style={{ display: 'flex', gap: '1rem', alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%' }}>
              {msg.sender === 'bot' && (
                <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Bot size={20} color="var(--accent-primary)" />
                </div>
              )}
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                <div style={{ 
                  background: msg.sender === 'user' ? 'var(--accent-gradient)' : msg.error ? 'rgba(239, 68, 68, 0.1)' : 'var(--bg-tertiary)',
                  color: msg.sender === 'user' ? 'white' : (msg.error ? 'var(--error)' : 'var(--text-primary)'),
                  padding: '12px 18px',
                  borderRadius: msg.sender === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                  border: msg.error ? '1px solid var(--error)' : '1px solid var(--border-color)',
                  boxShadow: 'var(--shadow-sm)',
                  lineHeight: '1.5'
                }}>
                  {msg.text}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '8px' }}>
                  <span>{msg.time}</span>
                  {msg.confidence && <span style={{ color: 'var(--success)' }}>Score: {msg.confidence}</span>}
                </div>
              </div>
            </div>
          ))}
          {isTyping && (
            <div style={{ display: 'flex', gap: '1rem', alignSelf: 'flex-start' }}>
               <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <RefreshCw size={18} className="animate-spin" color="var(--accent-primary)" />
                </div>
                <div style={{ padding: '12px 18px', background: 'var(--bg-tertiary)', borderRadius: '20px 20px 20px 4px', border: '1px solid var(--border-color)' }}>
                  <span className="animate-pulse">Typing...</span>
                </div>
            </div>
          )}

        </div>

        <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border-color)', background: 'rgba(22, 25, 37, 0.9)' }}>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <input 
              type="text" 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSend()}
              placeholder="Test the RAG model..." 
              style={{ 
                flex: 1, 
                background: 'var(--bg-secondary)', 
                border: '1px solid var(--border-color)', 
                color: 'var(--text-primary)', 
                padding: '14px 20px', 
                borderRadius: 'var(--radius-lg)', 
                outline: 'none',
                fontSize: '1rem',
                boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)'
              }} 
            />
            <button className="btn-primary" onClick={handleSend} style={{ borderRadius: 'var(--radius-lg)', padding: '0 24px' }}>
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>

      {/* Debug Panel */}
      <div className="glass-panel" style={{ width: '320px', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.02)' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={18} color="var(--warning)" />
            Debug Inspector
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Real-time AI execution trace</p>
        </div>
        <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto' }}>
          <div>
            <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Retrieval Strategy</p>
            <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', borderLeft: '4px solid var(--accent-primary)', fontSize: '0.85rem', fontFamily: 'monospace', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
              similarity_search_with_relevance_scores (k=5)
            </div>
          </div>
          <div>
            <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>System Prompt</p>
            <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', borderLeft: '4px solid var(--info)', fontSize: '0.85rem', color: 'var(--text-secondary)', maxHeight: '150px', overflowY: 'auto', lineHeight: '1.6' }}>
              "You are a professional, friendly AI Customer Support Assistant. RULES: Never reveal your internal thinking..."
            </div>
          </div>
          <div>
             <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Last Retrieved Context</p>
             <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', borderLeft: '4px solid var(--success)', fontSize: '0.85rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              (Context will appear here after a query)
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
