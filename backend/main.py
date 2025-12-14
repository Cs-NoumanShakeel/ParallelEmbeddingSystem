import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Import existing modules
# We need to ensure the root directory is in python path or imports work relative to where we run the script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_docs
from src.embedding import EmbeddingPipeline
from src.vectorstore import CHROMAVectorStore
from src.search import RAGRetriever
from chat_db import ChatDatabase

app = FastAPI()

# Mount static files (we will put index.html here)
# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

# Initialize components
print("Initializing dependencies...")
try:
    vector_store = CHROMAVectorStore()
    embedding_pipeline = EmbeddingPipeline(load_model=True) 
    chat_db = ChatDatabase()
except Exception as e:
    print(f"CRITICAL ERROR during initialization: {e}")
    raise e

# Ensure data directory exists
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)


# =====================
# Pydantic Models
# =====================

class QueryRequest(BaseModel):
    query: str
    chat_id: str  # Required now
    filename: Optional[str] = None

class CreateChatRequest(BaseModel):
    title: Optional[str] = "New Chat"
    filename: Optional[str] = None

class UpdateChatRequest(BaseModel):
    title: Optional[str] = None
    filename: Optional[str] = None


# =====================
# Static Pages
# =====================

@app.get("/")
async def get_index():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse(content=f"<h1>Error: index.html not found at {INDEX_PATH}</h1>", status_code=404)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =====================
# Chat History Endpoints
# =====================

@app.post("/chats")
async def create_chat(request: CreateChatRequest = None):
    """Create a new chat session"""
    title = request.title if request else "New Chat"
    filename = request.filename if request else None
    chat = chat_db.create_chat(title=title, filename=filename)
    return chat


@app.get("/chats")
async def get_chats():
    """Get all chat sessions"""
    chats = chat_db.get_all_chats()
    return {"chats": chats}


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat with its messages"""
    chat = chat_db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    messages = chat_db.get_messages(chat_id)
    return {"chat": chat, "messages": messages}


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, request: UpdateChatRequest):
    """Update chat title or filename"""
    success = chat_db.update_chat(chat_id, title=request.title, filename=request.filename)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "updated"}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat and all its messages"""
    success = chat_db.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "deleted"}


# =====================
# File Upload
# =====================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Load Document
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source_pdf"] = file.filename
            doc.metadata["source_path"] = file_path
        
        # 2. Chunk
        chunks = embedding_pipeline.chunk_documents(documents)
        
        if not chunks:
            print("Warning: No text chunks found in document.")
            return {"filename": file.filename, "status": "Warning: No text found in PDF (maybe scanned image?)", "chunks": 0}

        # 3. Preprocess & Embed (Synchronous)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Preprocess documents (but NOT queries)
        preprocessed_texts = embedding_pipeline.preprocess_texts(texts)
        
        # Double check if preprocessing stripped everything
        preprocessed_texts = [t for t in preprocessed_texts if t.strip()]
        if not preprocessed_texts:
             return {"filename": file.filename, "status": "Warning: Text was empty after preprocessing", "chunks": 0}

        # Re-align metadatas if we filtered texts
        filtered_data = [(t, m) for t, m in zip(embedding_pipeline.preprocess_texts(texts), metadatas) if t.strip()]
        if not filtered_data:
             return {"filename": file.filename, "status": "Warning: Text was empty after preprocessing", "chunks": 0}
        
        preprocessed_texts, metadatas = zip(*filtered_data)
        preprocessed_texts = list(preprocessed_texts)
        metadatas = list(metadatas)

        # Embed
        embeddings = embedding_pipeline.embed_texts(preprocessed_texts)
        
        # 4. Store
        vector_store.add_embeddings(
            texts=preprocessed_texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return {"filename": file.filename, "status": "Files successfully uploaded and indexed", "chunks": len(chunks)}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Chat / RAG Query
# =====================

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        # Get conversation history for context
        recent_messages = chat_db.get_recent_messages(request.chat_id, count=6)
        
        # Build conversation context
        conversation_context = ""
        if recent_messages:
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Initialize Retriever
        retriever = RAGRetriever(vector_store=vector_store, llm_repo='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
        
        # Construct filter if filename is provided
        search_filter = {"source_pdf": request.filename} if request.filename else None
        
        # Search with conversation context
        response = retriever.search(
            query=request.query, 
            filter=search_filter,
            conversation_context=conversation_context
        )
        
        # Save messages to database
        chat_db.add_message(request.chat_id, "user", request.query)
        chat_db.add_message(request.chat_id, "assistant", response)
        
        # Auto-generate title from first message if it's "New Chat"
        chat = chat_db.get_chat(request.chat_id)
        if chat and chat["title"] == "New Chat":
            # Use first 40 chars of first query as title
            new_title = request.query[:40] + "..." if len(request.query) > 40 else request.query
            chat_db.update_chat(request.chat_id, title=new_title)
        
        return {"response": response}
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"response": f"Error processing query: {str(e)}. (Did you set the HF_API_TOKEN?)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

