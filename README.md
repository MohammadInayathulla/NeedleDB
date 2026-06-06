# 🪡 NeedleDB

A Vector Database built from scratch in Python.

Implements HNSW, KD-Tree, and Brute Force search side-by-side
with a RAG pipeline powered by a local LLM via Ollama.

> Named after the core problem: finding a needle (the right answer)
> in a haystack (millions of vectors).

## Why this exists
Most people use Pinecone or Chroma as a black box.
This project builds the engine itself — every algorithm, every index,
every API endpoint written from scratch — to understand how
production vector databases actually work under the hood.

## Stack
- Python + NumPy (vector engine)
- FastAPI (REST API)
- Ollama (local embeddings + LLM)
- Vanilla HTML/JS (frontend)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Server starts at http://localhost:8000
Interactive docs at http://localhost:8000/docs

## API Reference

| Method   | Endpoint           | Description                          |
|----------|--------------------|--------------------------------------|
| GET      | /                  | API info                             |
| GET      | /health            | Health check                         |
| POST     | /vectors/insert    | Insert a vector                      |
| DELETE   | /vectors/{id}      | Delete a vector                      |
| GET      | /vectors           | List all vectors                     |
| GET      | /vectors/{id}      | Get a single vector                  |
| POST     | /search            | Search (hnsw / kdtree / brute_force) |
| POST     | /benchmark         | Compare all 3 algorithms             |
| GET      | /stats             | Database statistics                  |
| GET      | /hnsw-info         | HNSW graph structure                 |
| POST     | /demo/load         | Load 20 demo vectors                 |

## Roadmap
- [x] Day 1 — Distance metrics + Brute Force search
- [x] Day 2 — KD-Tree
- [x] Day 3 — HNSW (Hierarchical Navigable Small World)
- [x] Day 4 — Unified VectorDB interface + benchmark + demo data
- [x] Day 5 — FastAPI REST server
- [x] Day 6 — Ollama client (real 768D text embeddings + LLM generation)
- [x] Day 7 — Document store + text chunking pipeline
- [x] Day 8 — RAG pipeline (retrieval + grounded generation)
- [x] Day 9 — PCA visualization + Web UI
