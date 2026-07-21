# AI Backend Setup

## 1) Run the ingest file

Open a terminal in the project root and run:

```powershell
python backend/ai/ingest.py
```

This will generate the ChromaDB files in `data/chroma_db/`.

If you want to recreate the database from scratch, run:

```powershell
python backend/ai/ingest.py --reset
```

## 2) Create the `.env` file

Create a `.env` file in the project root with this content:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

This key is required for the RAG pipeline to work.