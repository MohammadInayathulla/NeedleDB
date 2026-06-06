import re
import numpy as np
from typing import Any, Dict, List, Optional

from .ollama_client import OllamaClient
from .vectordb import VectorDB


# ── Chunking ──────────────────────────────────────────────────────────────

def chunk_text(
    text:       str,
    chunk_size: int = 200,   # words per chunk
    overlap:    int = 40,    # words shared between consecutive chunks
) -> List[str]:
    """
    Split text into overlapping fixed-size word chunks.

    ── WHY CHUNK AT ALL? ────────────────────────────────────────────────
    Embedding models have a token limit (usually 512 tokens for
    nomic-embed-text). A long document can't be embedded whole — you'd
    either truncate it (losing content) or error out.
    Chunking solves this by breaking the document into embeddable pieces.

    ── WHY OVERLAP? ─────────────────────────────────────────────────────
    Without overlap a sentence on a chunk boundary gets split:
        - The previous chunk loses its conclusion
        - The next chunk loses its context
    Overlap ensures every sentence is fully inside at least one chunk.

    Example  (chunk_size=5, overlap=2):
        text   → "the cat sat on the mat near the door"
        chunk0 → "the cat sat on the"
        chunk1 → "on the mat near the"   ← "on the" repeats from chunk0
        chunk2 → "near the door"

    Args:
        text       : raw document text
        chunk_size : number of words per chunk
        overlap    : number of words repeated at the start of next chunk

    Returns:
        list of text strings, each ≤ chunk_size words
    """
    text  = re.sub(r'\s+', ' ', text.strip())   # normalize whitespace
    words = text.split()

    if not words:
        return []

    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap   # slide forward by stride

    return chunks


# ── DocumentDB ────────────────────────────────────────────────────────────

class DocumentDB:
    """
    Document store built on top of VectorDB.

    Turns raw text into searchable semantic chunks:

        add_document("doc1", "Python is a language...")
             │
             ▼
        chunk_text()  →  ["Python is a language...", "a language used for..."]
             │
             ▼
        ollama.embed(chunk)  →  [0.023, -0.041, ...]   (768 floats each)
             │
             ▼
        db.insert(chunk_id, vector, metadata={text, doc_id, chunk_index})
             │
             ▼
        search("what is Python?")  →  embed query  →  HNSW search  →  top chunks

    ── THIS IS THE RAG RETRIEVAL LAYER ──────────────────────────────────
    This is exactly what ChatPDF, NotebookLM, and every enterprise
    document Q&A system does — chunk, embed, index, retrieve.
    Day 8 adds the final piece: passing retrieved chunks to the LLM
    to generate a grounded answer.
    """

    def __init__(self, db: VectorDB, ollama: OllamaClient):
        self.db     = db
        self.ollama = ollama
        # doc_id → { doc_id, metadata, chunk_ids, text }
        self._docs: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def add_document(
        self,
        doc_id:     str,
        text:       str,
        metadata:   Optional[Dict[str, Any]] = None,
        chunk_size: int = 200,
        overlap:    int = 40,
        metric:     str = "cosine",
    ) -> Dict:
        """
        Chunk a document, embed each chunk, and insert into VectorDB.

        Each chunk gets a unique ID: "{doc_id}__chunk_{index}"
        Each chunk's metadata carries: doc_id, chunk_index, total_chunks,
        text (the raw chunk string), and any custom metadata passed in.

        Args:
            doc_id     : unique identifier for this document
            text       : the full document text
            metadata   : optional dict (title, source, author, etc.)
            chunk_size : words per chunk
            overlap    : overlapping words between chunks

        Returns:
            { doc_id, chunks_created, dims }

        Raises:
            ValueError  : duplicate doc_id or empty text
            RuntimeError: Ollama offline or model not pulled
        """
        if doc_id in self._docs:
            raise ValueError(
                f"Document '{doc_id}' already exists. "
                "Call delete_document() first."
            )
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty.")

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("Document produced no chunks after splitting.")

        meta      = metadata or {}
        chunk_ids = []
        dims      = 0

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}__chunk_{idx}"

            # ── Embed the chunk ───────────────────────────────────────
            vector = self.ollama.embed(chunk)
            dims   = len(vector)

            chunk_meta = {
                "doc_id":        doc_id,
                "chunk_index":   idx,
                "total_chunks":  len(chunks),
                "text":          chunk,
                **meta,          # inherit parent document metadata
            }

            self.db.insert(
                id       = chunk_id,
                vector   = vector,
                metadata = chunk_meta,
                metric   = metric,
            )
            chunk_ids.append(chunk_id)

        # Register document in the local store
        self._docs[doc_id] = {
            "doc_id":    doc_id,
            "metadata":  meta,
            "chunk_ids": chunk_ids,
            "text":      text,
        }

        return {
            "doc_id":         doc_id,
            "chunks_created": len(chunks),
            "dims":           dims,
        }

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and ALL its chunk vectors from VectorDB.
        Returns True if found and deleted, False if not found.
        """
        if doc_id not in self._docs:
            return False

        for chunk_id in self._docs[doc_id]["chunk_ids"]:
            self.db.delete(chunk_id)

        del self._docs[doc_id]
        return True

    # ------------------------------------------------------------------ #
    #  Read operations                                                     #
    # ------------------------------------------------------------------ #

    def list_documents(self) -> List[Dict]:
        """Return all documents — id, metadata, chunk count. No raw text."""
        return [
            {
                "doc_id":      doc_id,
                "metadata":    data["metadata"],
                "chunk_count": len(data["chunk_ids"]),
            }
            for doc_id, data in self._docs.items()
        ]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Return a single document including its full text and chunk IDs."""
        return self._docs.get(doc_id)

    # ------------------------------------------------------------------ #
    #  Semantic search                                                     #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query:  str,
        k:      int = 5,
        metric: str = "cosine",
        algo:   str = "hnsw",
    ) -> Dict:
        """
        Semantic search over all document chunks.

        Steps:
            1. Embed the query text (Ollama → 768D vector)
            2. Search VectorDB for k nearest chunk vectors (HNSW by default)
            3. Return matching text chunks with source document info

        The returned `text` fields are what gets passed to the LLM
        as context in the RAG pipeline on Day 8.

        Returns:
            {
                query, algo, k, elapsed_ms,
                results: [
                    {
                        chunk_id, distance,
                        text        : the raw chunk text (this is the context),
                        doc_id      : which document it came from,
                        chunk_index : position within the document,
                        total_chunks: total chunks in that document
                    },
                    ...
                ]
            }
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector  = self.ollama.embed(query)
        search_result = self.db.search(
            query  = np.array(query_vector, dtype=np.float32),
            k      = k,
            metric = metric,
            algo   = algo,
        )

        results = [
            {
                "chunk_id":    item["id"],
                "distance":    item["distance"],
                "text":        item["metadata"].get("text", ""),
                "doc_id":      item["metadata"].get("doc_id", ""),
                "chunk_index": item["metadata"].get("chunk_index", 0),
                "total_chunks":item["metadata"].get("total_chunks", 1),
            }
            for item in search_result["results"]
        ]

        return {
            "query":      query,
            "algo":       algo,
            "k":          k,
            "elapsed_ms": search_result["elapsed_ms"],
            "results":    results,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self._docs)

    def __repr__(self):
        total = sum(len(d["chunk_ids"]) for d in self._docs.values())
        return f"DocumentDB(documents={len(self._docs)}, total_chunks={total})"