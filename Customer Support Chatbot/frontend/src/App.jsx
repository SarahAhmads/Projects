import React, { useState } from 'react';

import './index.css';

import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import Analytics from './components/Analytics';
import KnowledgeBase from './components/KnowledgeBase';
import Settings from './components/Settings';

function App() {
  const [activeTab, setActiveTab] = useState('analytics');

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return <Chat />;
      case 'analytics':
        return <Analytics />;
      case 'knowledge':
        return <KnowledgeBase />;
      case 'settings':
        return <Settings />;
      default:
        return <Analytics />;
    }
  };

  return (
    <div className="app-container">
      <div className="bg-glow-1"></div>
      <div className="bg-glow-2"></div>
      
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="main-content animate-fade-in" style={{ padding: '2rem', overflowY: 'auto' }}>
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
