import React, { useState, useEffect } from 'react';
import { Upload, FileText, Trash2, Search, RefreshCw } from 'lucide-react';

const KnowledgeBase = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/knowledge', { headers: { 'X-API-Key': 'dev-secret-key' } });
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await fetch('/api/knowledge/upload', {
        method: 'POST',
        headers: { 'X-API-Key': 'dev-secret-key' },
        body: formData
      });
      fetchDocuments();
    } catch (e) {
      console.error(e);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete ${filename}?`)) return;
    
    try {
      await fetch(`/api/knowledge/${filename}`, {
        method: 'DELETE',
        headers: { 'X-API-Key': 'dev-secret-key' }
      });
      fetchDocuments();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Knowledge Base</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Manage the documents and FAQs the AI uses to generate answers.</p>
        </div>
        <label className="btn-primary" style={{ cursor: isUploading ? 'not-allowed' : 'pointer', opacity: isUploading ? 0.7 : 1 }}>
          <input type="file" style={{ display: 'none' }} onChange={handleUpload} disabled={isUploading} />
          {isUploading ? <RefreshCw className="animate-spin" size={18} /> : <Upload size={18} />}
          {isUploading ? 'Uploading...' : 'Upload Document'}
        </label>
      </div>

      <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Indexed Documents</h3>
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input 
              type="text" 
              placeholder="Search documents..." 
              style={{ 
                background: 'var(--bg-secondary)', 
                border: '1px solid var(--border-color)', 
                color: 'var(--text-primary)', 
                padding: '8px 12px 8px 36px', 
                borderRadius: 'var(--radius-md)', 
                outline: 'none',
                width: '250px'
              }} 
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', fontSize: '0.85rem', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '1rem', fontWeight: 500 }}>Document Name</th>
                <th style={{ padding: '1rem', fontWeight: 500 }}>Size</th>
                <th style={{ padding: '1rem', fontWeight: 500 }}>Upload Date</th>
                <th style={{ padding: '1rem', fontWeight: 500 }}>Status</th>
                <th style={{ padding: '1rem', fontWeight: 500, textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map(doc => (
                <tr key={doc.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)', transition: 'background 0.2s' }} onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.02)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ padding: '8px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', color: 'var(--accent-primary)' }}>
                        <FileText size={18} />
                      </div>
                      <span style={{ fontWeight: 500 }}>{doc.name}</span>
                    </div>
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--text-secondary)' }}>{doc.size}</td>
                  <td style={{ padding: '1rem', color: 'var(--text-secondary)' }}>{doc.date}</td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{ 
                      padding: '4px 10px', 
                      borderRadius: 'var(--radius-full)', 
                      fontSize: '0.8rem', 
                      fontWeight: 600,
                      background: doc.status === 'Indexed' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                      color: doc.status === 'Indexed' ? 'var(--success)' : 'var(--warning)'
                    }}>
                      {doc.status}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', textAlign: 'right' }}>
                    <button onClick={() => handleDelete(doc.id)} style={{ color: 'var(--text-muted)', padding: '6px' }} onMouseEnter={e => e.currentTarget.style.color = 'var(--error)'} onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}>
                      <Trash2 size={18} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeBase;
