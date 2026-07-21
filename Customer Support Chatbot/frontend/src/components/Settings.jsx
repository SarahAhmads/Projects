import React, { useState, useEffect } from 'react';
import { Save, Settings2, Sliders, Cpu, Sparkles, MessageSquare } from 'lucide-react';

const Settings = () => {
  const [isSaving, setIsSaving] = useState(false);
  const [config, setConfig] = useState({
    systemPrompt: "Loading...",
    temperature: 0.7,
    maxTokens: 500,
    model: "deepseek-chat"
  });

  useEffect(() => {
    fetch('/api/settings', { headers: { 'X-API-Key': 'dev-secret-key' } })
      .then(res => res.json())
      .then(data => setConfig({
        systemPrompt: data.systemPrompt || "",
        temperature: data.temperature || 0.7,
        maxTokens: 500,
        model: data.model || "deepseek-chat"
      }))
      .catch(e => console.error(e));
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'dev-secret-key'
        },
        body: JSON.stringify({
          systemPrompt: config.systemPrompt,
          temperature: config.temperature,
          model: config.model
        })
      });
    } catch (e) {
      console.error(e);
    }
    setIsSaving(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', height: '100%', maxWidth: '900px', margin: '0 auto' }}>
      <div>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Configuration</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Fine-tune your AI assistant's brain and behavior.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
        
        {/* System Prompt Card */}
        <div className="glass-card" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <MessageSquare size={20} color="var(--accent-primary)" />
              System Prompt
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              The core instructions that dictate the assistant's persona, tone, and operational boundaries.
            </p>
          </div>
          
          <textarea 
            value={config.systemPrompt}
            onChange={(e) => setConfig({...config, systemPrompt: e.target.value})}
            style={{ 
              width: '100%', 
              height: '160px', 
              background: 'var(--bg-tertiary)', 
              border: '1px solid var(--border-color)', 
              color: 'var(--text-primary)', 
              padding: '1.5rem', 
              borderRadius: 'var(--radius-md)', 
              outline: 'none',
              resize: 'vertical',
              fontFamily: 'monospace',
              fontSize: '0.9rem',
              lineHeight: '1.6',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.02)',
              transition: 'border-color 0.2s ease, box-shadow 0.2s ease'
            }}
            onFocus={e => {
              e.target.style.borderColor = 'var(--accent-primary)';
              e.target.style.boxShadow = '0 0 0 3px rgba(99, 102, 241, 0.1)';
            }}
            onBlur={e => {
              e.target.style.borderColor = 'var(--border-color)';
              e.target.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.02)';
            }}
          />
        </div>

        {/* Model Parameters Card */}
        <div className="glass-card" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
             <h3 style={{ fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Sliders size={20} color="var(--accent-primary)" />
              Generation Parameters
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Control how the AI generates text, which model it uses, and its creativity level.
            </p>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
            
            {/* Model Dropdown */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <label style={{ fontSize: '0.95rem', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={16} color="var(--text-muted)" />
                AI Model
              </label>
              <div style={{ position: 'relative' }}>
                <select 
                  value={config.model}
                  onChange={(e) => setConfig({...config, model: e.target.value})}
                  style={{ 
                    width: '100%',
                    background: 'var(--bg-tertiary)', 
                    border: '1px solid var(--border-color)', 
                    color: 'var(--text-primary)', 
                    padding: '12px 16px', 
                    borderRadius: 'var(--radius-md)', 
                    outline: 'none',
                    appearance: 'none',
                    cursor: 'pointer',
                    fontSize: '0.95rem',
                    transition: 'border-color 0.2s ease'
                  }}
                  onFocus={e => e.target.style.borderColor = 'var(--accent-primary)'}
                  onBlur={e => e.target.style.borderColor = 'var(--border-color)'}
                >
                  <option value="deepseek-chat">DeepSeek Chat (V3)</option>
                  <option value="deepseek-reasoner">DeepSeek Reasoner (R1)</option>
                  <option value="deepseek-v4-flash">DeepSeek Flash (V4)</option>
                  <option value="openrouter/free">OpenRouter Free Auto-Router (Free)</option>
                  <option value="google/gemma-4-31b-it:free">Gemma 4 31B (Free)</option>
                  <option value="openai/gpt-oss-20b:free">GPT-OSS 20B (Free)</option>
                </select>
                <div style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-muted)' }}>
                  ▼
                </div>
              </div>
            </div>

            {/* Temperature Slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.95rem', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Sparkles size={16} color="var(--text-muted)" />
                  Temperature
                </label>
                <span style={{ 
                  background: 'rgba(99, 102, 241, 0.1)', 
                  color: 'var(--accent-primary)', 
                  padding: '4px 10px', 
                  borderRadius: '12px', 
                  fontSize: '0.85rem', 
                  fontWeight: 600 
                }}>
                  {config.temperature}
                </span>
              </div>
              <input 
                type="range" 
                min="0" max="1" step="0.1" 
                value={config.temperature}
                onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
                style={{ 
                  width: '100%', 
                  accentColor: 'var(--accent-primary)',
                  height: '6px',
                  borderRadius: '4px',
                  outline: 'none',
                  marginTop: '8px',
                  cursor: 'pointer'
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span>Precise</span>
                <span>Balanced</span>
                <span>Creative</span>
              </div>
            </div>

          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem', paddingBottom: '2rem' }}>
          <button className="btn-primary" onClick={handleSave} disabled={isSaving} style={{ padding: '14px 28px', opacity: isSaving ? 0.7 : 1, fontSize: '1rem' }}>
            <Save size={20} />
            {isSaving ? 'Saving Changes...' : 'Save Configuration'}
          </button>
        </div>

      </div>
    </div>
  );
};
export default Settings;
