import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, MessageSquare, TrendingUp, Clock } from 'lucide-react';

const StatCard = ({ title, value, icon: Icon, trend }) => (
  <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500, marginBottom: '0.5rem' }}>{title}</p>
        <h3 style={{ fontSize: '2rem', margin: 0, fontWeight: 700 }}>{value}</h3>
      </div>
      <div style={{ padding: '10px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '12px', color: 'var(--accent-primary)' }}>
        <Icon size={24} />
      </div>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem' }}>
      <TrendingUp size={16} color="var(--success)" />
      <span style={{ color: 'var(--success)', fontWeight: 600 }}>{trend}</span>
      <span style={{ color: 'var(--text-muted)' }}>vs last week</span>
    </div>
  </div>
);

const Analytics = () => {
  const [stats, setStats] = useState({
    total_messages: 0,
    active_users: 0,
    avg_latency: "0s",
    avg_confidence: 0
  });

  const [chartData, setChartData] = useState([
    { name: 'Mon', messages: 0 },
    { name: 'Tue', messages: 0 },
    { name: 'Wed', messages: 0 },
    { name: 'Thu', messages: 0 },
    { name: 'Fri', messages: 0 },
    { name: 'Sat', messages: 0 },
    { name: 'Sun', messages: 0 },
  ]);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/stats', { headers: { 'X-API-Key': 'dev-secret-key' } });        const data = await res.json();
        if (data && data.total_messages !== undefined) {
          setStats(data);
          
          const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
          const today = days[new Date().getDay()];
          setChartData(prev => prev.map(d => 
            d.name === today ? { ...d, messages: data.total_messages } : d
          ));
        } else {
          console.error("Invalid stats data:", data);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchStats();
  }, []);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', height: '100%' }}>
      <div>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Dashboard Overview</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Welcome back. Here is what's happening with your AI Chatbot today.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem' }}>
        <StatCard title="Total Messages" value={stats.total_messages.toLocaleString()} icon={MessageSquare} trend="0%" />
        <StatCard title="Active Users" value={stats.active_users.toLocaleString()} icon={Users} trend="0%" />
        <StatCard title="Avg Response Time" value={stats.avg_latency} icon={Clock} trend="0s" />
        <StatCard title="Avg Confidence Score" value={`${(stats.avg_confidence * 100).toFixed(1)}%`} icon={TrendingUp} trend="0%" />
      </div>

      <div className="glass-panel" style={{ padding: '2rem', flex: 1, minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Interaction Volume</h3>
        </div>
        <div style={{ flex: 1, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorMessages" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--text-muted)" tick={{fill: 'var(--text-muted)'}} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-muted)" tick={{fill: 'var(--text-muted)'}} axisLine={false} tickLine={false} />
              <Tooltip 
                contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
              <Area type="monotone" dataKey="messages" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorMessages)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
