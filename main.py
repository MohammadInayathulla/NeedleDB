import numpy as np
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


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
