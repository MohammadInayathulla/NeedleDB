import numpy as np
from needledb.ollama_client import OllamaClient
from needledb.document_db import DocumentDB
from needledb.rag import RAGPipeline
import uvicorn
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from needledb import VectorDB


# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NeedleDB",
    description = "A vector database built from scratch — HNSW + KD-Tree + Brute Force.",
    version     = "0.5.0",
)

# Allow the frontend (index.html) to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Single shared database instance ───────────────────────────────────────

db = VectorDB(M=16, ef_construction=200, ef_search=50)
ollama = OllamaClient()
doc_db = DocumentDB(db=db, ollama=ollama)
rag = RAGPipeline(doc_db=doc_db, ollama=ollama)

# ── Request / Response models ──────────────────────────────────────────────
# Pydantic validates every incoming request automatically.
# If a field is wrong type or missing, FastAPI returns a 422 with details.

class InsertRequest(BaseModel):
    id:       str
    vector:   List[float]
    metadata: Optional[Dict[str, Any]] = {}

class SearchRequest(BaseModel):
    vector: List[float]
    k:      int = Field(default=5, ge=1, le=100,
                        description="Number of nearest neighbors to return")
    metric: str = Field(default="cosine",
                        description="cosine | euclidean | manhattan")
    algo:   str = Field(default="hnsw",
                        description="hnsw | kdtree | brute_force")

class BenchmarkRequest(BaseModel):
    vector: List[float]
    k:      int = Field(default=5, ge=1, le=100)
    metric: str = Field(default="cosine")


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """API info — useful for a quick sanity check."""
    return {
        "name":        "NeedleDB",
        "tagline":     "Finding the needle in the haystack — built from scratch",
        "version":     "0.5.0",
        "total":       len(db),
        "interactive_docs": "http://localhost:8000/docs",
    }


@app.get("/health", tags=["General"])
def health():
    """Health check endpoint."""
    return {
        "status":  "ok",
        "vectors": len(db),
    }


# ── Vector CRUD ────────────────────────────────────────────────────────────

@app.post("/vectors/insert", tags=["Vectors"])
def insert(req: InsertRequest):
    """
    Insert a vector into all three indexes simultaneously.

    Example body:
        {
            "id":       "my_vec_001",
            "vector":   [0.9, 0.8, 0.1, 0.1, ...],
            "metadata": { "label": "binary search", "category": "cs" }
        }
    """
    if not req.vector:
        raise HTTPException(status_code=400, detail="Vector cannot be empty.")

    db.insert(id=req.id, vector=req.vector, metadata=req.metadata)

    return {
        "inserted": req.id,
        "dims":     len(req.vector),
        "total":    len(db),
    }


@app.delete("/vectors/{id}", tags=["Vectors"])
def delete(id: str):
    """Delete a vector by ID from all three indexes."""
    deleted = db.delete(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Vector '{id}' not found.")
    return {
        "deleted": id,
        "total":   len(db),
    }


@app.get("/vectors", tags=["Vectors"])
def list_vectors():
    """List all stored vectors (IDs + metadata, no raw vectors)."""
    return {
        "items": db.list_items(),
        "total": len(db),
    }


@app.get("/vectors/{id}", tags=["Vectors"])
def get_vector(id: str):
    """Retrieve a single vector including its raw values."""
    item = db.get_item(id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Vector '{id}' not found.")
    return {"id": id, **item}


# ── Search ─────────────────────────────────────────────────────────────────

@app.post("/search", tags=["Search"])
def search(req: SearchRequest):
    """
    Search using one algorithm.

    algo options : hnsw | kdtree | brute_force
    metric options: cosine | euclidean | manhattan

    Example body:
        {
            "vector": [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "k": 3,
            "metric": "cosine",
            "algo": "hnsw"
        }
    """
    if len(db) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = "Database is empty. POST /demo/load first.",
        )

    try:
        result = db.search(
            query  = np.array(req.vector, dtype=np.float32),
            k      = req.k,
            metric = req.metric,
            algo   = req.algo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.post("/benchmark", tags=["Search"])
def benchmark(req: BenchmarkRequest):
    """
    Run all three algorithms on the same query and compare.

    Returns timing for each algorithm and HNSW recall vs brute-force
    ground truth. This is the core demo endpoint.
    """
    if len(db) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = "Database is empty. POST /demo/load first.",
        )

    result = db.benchmark(
        query  = np.array(req.vector, dtype=np.float32),
        k      = req.k,
        metric = req.metric,
    )
    return result


# ── Stats & Introspection ──────────────────────────────────────────────────

@app.get("/stats", tags=["Introspection"])
def stats():
    """Return database-level statistics."""
    return db.get_stats()


@app.get("/hnsw-info", tags=["Introspection"])
def hnsw_info():
    """
    Return HNSW graph structure details — layer counts,
    average connections per layer, entry point, M, ef values.
    Great for understanding how the graph grows as you insert data.
    """
    return db.hnsw.get_graph_info()


# ── Demo data ──────────────────────────────────────────────────────────────

@app.post("/demo/load", tags=["Demo"])
def load_demo():
    """
    Load the 20 pre-built semantic demo vectors (CS, Math, Food, Sports).
    Safe to call multiple times — skips if already loaded.
    """
    if len(db) > 0:
        return {
            "message": "Demo data already loaded.",
            "total":   len(db),
        }

    count = db.load_demo_data()
    return {
        "message": f"Loaded {count} demo vectors across 4 categories.",
        "total":   len(db),
    }

# ── Ollama ─────────────────────────────────────────────────────────────

@app.get("/status", tags=["Ollama"])
def ollama_status():
    """
    Check if Ollama is running and both models are pulled.
    Hit this first whenever you restart NeedleDB.
    """
    return ollama.status()


@app.post("/embed", tags=["Ollama"])
def embed_text(body: dict):
    """
    Embed any text into a 768D vector using nomic-embed-text.

    This is the bridge between raw text and the vector index.
    The returned vector can be passed directly to /search or /benchmark.

    Example body:
        { "text": "What is a binary search tree?" }

    Returns:
        {
            "text"      : original text,
            "dims"      : 768,
            "vector"    : [0.023, -0.041, ...]   ← 768 floats
        }
    """
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required.")

    try:
        vector = ollama.embed(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "text":   text,
        "dims":   len(vector),
        "vector": vector,
    }

# ── Documents ──────────────────────────────────────────────────────────

@app.post("/documents/add", tags=["Documents"])
def add_document(body: dict):
    """
    Add a document — chunks it, embeds each chunk, inserts into VectorDB.

    Example body:
        {
            "doc_id":     "python_intro",
            "text":       "Python is a high-level programming language...",
            "metadata":   { "title": "Python Intro", "source": "wikipedia" },
            "chunk_size": 200,
            "overlap":    40
        }
    """
    doc_id     = body.get("doc_id",     "").strip()
    text       = body.get("text",       "").strip()
    metadata   = body.get("metadata",   {})
    chunk_size = body.get("chunk_size", 200)
    overlap    = body.get("overlap",    40)

    if not doc_id:
        raise HTTPException(status_code=400, detail="Field 'doc_id' is required.")
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required.")

    try:
        result = doc_db.add_document(
            doc_id     = doc_id,
            text       = text,
            metadata   = metadata,
            chunk_size = chunk_size,
            overlap    = overlap,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Delete a document and all its chunk vectors."""
    deleted = doc_db.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code = 404,
            detail      = f"Document '{doc_id}' not found."
        )
    return {"deleted": doc_id}


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all documents — id, metadata, chunk count."""
    return {
        "documents": doc_db.list_documents(),
        "total":     len(doc_db),
    }


@app.get("/documents/{doc_id}", tags=["Documents"])
def get_document(doc_id: str):
    """Get a single document including full text and chunk IDs."""
    doc = doc_db.get_document(doc_id)
    if doc is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"Document '{doc_id}' not found."
        )
    return doc


@app.post("/documents/search", tags=["Documents"])
def search_documents(body: dict):
    """
    Semantic search over all document chunks using plain text query.

    Embeds the query, finds nearest chunks, returns text + source info.
    The results from this endpoint feed directly into the RAG pipeline
    on Day 8 as the context for LLM answer generation.

    Example body:
        {
            "query": "what is a binary search tree?",
            "k": 5,
            "metric": "cosine",
            "algo": "hnsw"
        }
    """
    query  = body.get("query",  "").strip()
    k      = body.get("k",      5)
    metric = body.get("metric", "cosine")
    algo   = body.get("algo",   "hnsw")

    if not query:
        raise HTTPException(status_code=400, detail="Field 'query' is required.")
    if len(doc_db) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = "No documents added yet. POST /documents/add first."
        )

    try:
        return doc_db.search(query=query, k=k, metric=metric, algo=algo)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
# ── RAG ────────────────────────────────────────────────────────────────

@app.post("/rag/ask", tags=["RAG"])
def rag_ask(body: dict):
    """
    Ask a question against your documents — full RAG pipeline.

    Steps internally:
        1. Embeds your question (Ollama → 768D)
        2. Finds top-k relevant chunks (HNSW search)
        3. Builds a grounded prompt (context + question)
        4. Generates an answer (llama3.2)
        5. Returns answer + source chunks for transparency

    Example body:
        {
            "question": "What is a binary search tree?",
            "k": 5,
            "metric": "cosine",
            "algo": "hnsw"
        }

    Note: Requires Ollama running + both models pulled.
          Add documents first via POST /documents/add.
          Generation takes 10-30s on CPU — this is normal.
    """
    question = body.get("question", "").strip()
    k        = body.get("k",        5)
    metric   = body.get("metric",   "cosine")
    algo     = body.get("algo",     "hnsw")

    if not question:
        raise HTTPException(
            status_code = 400,
            detail      = "Field 'question' is required."
        )

    if len(doc_db) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = "No documents in store. POST /documents/add first."
        )

    try:
        result = rag.ask(
            question = question,
            k        = k,
            metric   = metric,
            algo     = algo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
