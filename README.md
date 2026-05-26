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

## Roadmap
- [x] Day 1 — Distance metrics + Brute Force search
- [ ] Day 2 — KD-Tree
- [ ] Day 3 — HNSW
- [ ] Day 4 — Unified VectorDB interface
- [ ] Day 5 — FastAPI server
- [ ] Day 6 — Ollama embeddings
- [ ] Day 7 — Document store + chunking
- [ ] Day 8 — RAG pipeline
- [ ] Day 9 — PCA visualization + Web UI
