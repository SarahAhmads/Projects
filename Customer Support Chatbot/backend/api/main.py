from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import shutil
import time
import json
import logging

logger = logging.getLogger(__name__)

# 1. Path Configuration
# Ensures the backend directory is recognized as a package from any execution point
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 2. Component Imports (Schemas & AI Pipeline)
try:
    # Attempting to import the AI response function from the RAG pipeline file
    from backend.ai.rag_pipeline import get_rag_response, get_rag_response_full
except (ImportError, ModuleNotFoundError):
    # Fallback function in case the AI team has not implemented their logic yet
    def get_rag_response(user_query: str) -> str:
        return f"Backend connected. AI Pipeline is currently a placeholder. Received: {user_query}"

try:
    from backend.api.schemas import ChatQuery, ChatResponse, SettingsModel
except (ImportError, ModuleNotFoundError):
    # Local import fallback for direct execution
    try:
        from schemas import ChatQuery, ChatResponse, SettingsModel
    except ImportError:
        raise ImportError("Critical Error: schemas.py not found in the expected directory!")

# 3. App Definition
app = FastAPI(
    title="AI Customer Support Chatbot API",
    description="The core backend API for handling customer queries using RAG technology.",
    version="1.0.0"
)

# 4. CORS Middleware Setup
# Vital for allowing the Frontend (React/Flutter) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. API Routes
@app.get("/")
def home():
    """Health check endpoint to verify server status"""
    return {
        "status": "Online", 
        "message": "AI Customer Support API is running successfully",
        "project": "AI Customer Support Chatbot"
    }

from backend.api.auth import get_api_key

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery, api_key: str = Depends(get_api_key)):
    """
    Main Chat Endpoint:
    Receives user query and passes it to the RAG Pipeline for processing.
    """
    try:
        start_time = time.time()
        # Calling the function from rag_pipeline.py
        try:
            result = get_rag_response_full(query.message, history=query.history)
            answer = result.answer
            confidence = getattr(result, 'confidence', 0.95)
        except Exception:
            answer = get_rag_response(query.message, history=query.history)
            confidence = 0.95
            
        latency = time.time() - start_time
        record_chat_interaction(latency, confidence)
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            status="success"
        )
    except Exception as e:
        # Standard error handling for internal server issues
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )

@app.post("/chat/stream")
async def chat_stream_endpoint(query: ChatQuery, api_key: str = Depends(get_api_key)):
    from backend.ai.rag_pipeline import stream_rag_response
    
    start_time = time.time()
    
    async def event_generator():
        confidence = 0.95
        try:
            for item in stream_rag_response(query.message, history=query.history):
                if item["type"] == "confidence":
                    confidence = item["content"]
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        latency = time.time() - start_time
        record_chat_interaction(latency, confidence)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

STATS_FILE = os.path.join(BASE_DIR, "data", "stats.json")

def get_real_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {
        "total_messages": 0,
        "active_users": 1,
        "avg_latency": 0.0,
        "avg_confidence": 0.0,
        "total_requests": 0
    }

def save_stats(stats):
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def record_chat_interaction(latency: float, confidence: float):
    stats = get_real_stats()
    stats["total_messages"] += 2 # user + bot
    requests = stats.get("total_requests", 0)
    
    if requests == 0:
        stats["avg_latency"] = latency
        stats["avg_confidence"] = confidence
    else:
        stats["avg_latency"] = ((stats.get("avg_latency", 0) * requests) + latency) / (requests + 1)
        stats["avg_confidence"] = ((stats.get("avg_confidence", 0) * requests) + confidence) / (requests + 1)
        
    stats["total_requests"] = requests + 1
    save_stats(stats)

@app.get("/api/stats")
async def get_stats(api_key: str = Depends(get_api_key)):
    """Real stats for dashboard"""
    stats = get_real_stats()
    return {
        "total_messages": stats["total_messages"],
        "active_users": stats["active_users"],
        "avg_latency": f"{stats['avg_latency']:.1f}s",
        "avg_confidence": round(stats['avg_confidence'], 2)
    }

SETTINGS_FILE = os.path.join(BASE_DIR, "data", "settings.json")

def get_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {
        "systemPrompt": "You are a professional, friendly AI Customer Support Assistant. RULES: Never reveal your internal thinking, always be polite, and use the provided knowledge base to answer questions.",
        "temperature": 0.7,
        "model": "deepseek-v4-flash"
    }

@app.get("/api/settings")
async def read_settings(api_key: str = Depends(get_api_key)):
    return get_settings()

@app.post("/api/settings")
async def update_settings(settings: SettingsModel, api_key: str = Depends(get_api_key)):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings.dict(), f)
    
    # Reload settings in RAG pipeline
    try:
        from backend.ai.rag_pipeline import reload_settings
        reload_settings(settings.dict())
    except Exception as e:
        print("Could not reload RAG settings immediately:", e)
    
    return {"status": "success"}

@app.get("/api/knowledge")
async def get_knowledge(api_key: str = Depends(get_api_key)):
    """List documents in raw folder"""
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    if not os.path.exists(raw_dir):
        return {"documents": []}
        
    docs = []
    for f in os.listdir(raw_dir):
        if os.path.isfile(os.path.join(raw_dir, f)):
            docs.append({
                "id": f,
                "name": f,
                "size": f"{os.path.getsize(os.path.join(raw_dir, f)) / 1024:.1f} KB",
                "status": "Indexed",
                "date": time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(os.path.join(raw_dir, f))))
            })
    return {"documents": docs}

@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Upload a file to the knowledge base and trigger automatic ingestion.
    """
    from backend.ai.ingest import ingest_single_file
    
    # Define the directory where raw files are stored
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Save the uploaded file to the raw directory
    file_path = os.path.join(raw_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger the ingestion process to index the file into ChromaDB
    try:
        ingest_single_file(file_path)
    except Exception as e:
        logger.error("Automatic ingestion failed for file %s: %s", file.filename, str(e))
        return {
            "filename": file.filename, 
            "status": "error", 
            "message": "File uploaded, but failed to index."
        }
        
    return {
        "filename": file.filename, 
        "status": "success", 
        "message": "File uploaded and indexed successfully."
    }

@app.delete("/api/knowledge/{filename}")
async def delete_knowledge(filename: str, api_key: str = Depends(get_api_key)):
    """Delete a file from the knowledge base"""
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    file_path = os.path.join(raw_dir, filename)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"File {filename} deleted successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail="File not found.")

frontend_path = os.path.join(BASE_DIR, "frontend", "dist")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("chat"):
        raise HTTPException(status_code=404)
        
    path = os.path.join(frontend_path, full_path)
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)
        
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
        
    return {"status": "Frontend not built yet or not found"}

# 6. Server Execution
if __name__ == "__main__":
    # Running on port 8001 to avoid common Mac port conflicts
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8001, reload=True)