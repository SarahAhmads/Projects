# AI Customer Support Chatbot 🤖

A production-ready, intelligent customer support system powered by **Retrieval-Augmented Generation (RAG)**. This project features a high-performance Python FastAPI backend integrated with LangChain and DeepSeek models, paired with a stunning, modern React dashboard (Glassmorphism UI) for complete control and analytics.

---

## ✨ Key Features

### 1. 🚀 Ultra-Fast AI Chat (with Streaming)
*   **Real-time Streaming (SSE):** AI responses are streamed word-by-word instantly, just like ChatGPT, eliminating perceived latency.
*   **Conversational Memory:** Remembers the context of the conversation allowing seamless back-and-forth interactions.
*   **Powered by DeepSeek:** Uses the cutting-edge `deepseek-v4-flash` (or `deepseek-chat`/`reasoner`) for lightning-fast and highly accurate responses.

### 2. 📚 Dynamic Knowledge Base (RAG)
*   **Instant Indexing:** Upload PDF, TXT, or Document files directly from the dashboard. The system instantly embeds them into the ChromaDB vector store.
*   **Smart Retrieval:** Uses `sentence-transformers/all-MiniLM-L6-v2` for precise semantic search.
*   **Live Management:** Delete or manage knowledge files directly from the user interface.

### 3. ⚙️ Live Configuration & Settings
*   **Dynamic Settings:** Change the AI Model, Temperature, and System Prompt on the fly. No backend restart required!
*   **Premium UI:** A beautifully crafted glass-card interface for seamless configuration.

### 4. 📊 Real-Time Analytics Dashboard
*   **Interactive Charts:** Dynamic area charts showing daily interaction volume based on real server data.
*   **Live Statistics:** Tracks Total Messages, Average Response Time (Latency), and AI Confidence Scores.

---

## 🛠️ Technology Stack

**Backend:**
*   **Framework:** FastAPI (Python)
*   **AI/LLM:** LangChain, DeepSeek API
*   **Vector Database:** ChromaDB
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

**Frontend:**
*   **Library:** React.js
*   **Styling:** Vanilla CSS (Modern Glassmorphism & Animations)
*   **Icons:** Lucide React
*   **Charts:** Recharts

---

## 📁 Project Structure

```text
.
├── backend/
│   ├── ai/
│   │   └── rag_pipeline.py    # Core RAG logic, LangChain chain, and streaming setup
│   └── api/
│       └── main.py            # FastAPI server, endpoints, and statistics tracker
├── data/
│   ├── chroma_db/             # Vector database storage
│   ├── raw/                   # Uploaded knowledge base files
│   ├── settings.json          # Persistent AI configurations
│   └── stats.json             # Persistent analytics data
├── frontend/                  
│   ├── src/
│   │   ├── components/        # React Components (Chat, Analytics, Settings, KnowledgeBase)
│   │   └── App.jsx            # Main React routing and layout
│   └── index.css              # Global styles and CSS variables
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Start the Backend
Navigate to the root directory and start the FastAPI server:
```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8001 --reload
```
*(The backend runs on port 8001 to avoid conflicts).*

### 2. Start the Frontend
Navigate to the `frontend` directory and start the React development server:
```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables
Create a `.env` file in the root directory using `.env.example` as a template:
```bash
cp .env.example .env
```
Then fill in your keys:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
API_KEY=dev-secret-key
```
> **Get a free API key** at [openrouter.ai](https://openrouter.ai) — no credit card required for free-tier models.

---

## 🌟 Future Enhancements
*   Add multi-tenant support for different organizations.
*   Implement user authentication for the dashboard.
*   Expand analytics with sentiment analysis on customer interactions.
