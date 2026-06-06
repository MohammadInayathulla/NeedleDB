from typing import Dict, List, Optional

from .document_db import DocumentDB
from .ollama_client import OllamaClient


class RAGPipeline:
    """
    Retrieval-Augmented Generation — the complete pipeline.

    Connects DocumentDB (retrieval) with OllamaClient (generation).

    ── WHAT IS RAG? ─────────────────────────────────────────────────────
    A plain LLM answers from training data — it can hallucinate facts,
    has a knowledge cutoff, and knows nothing about YOUR documents.

    RAG fixes this by grounding every answer in retrieved evidence:
        1. Embed the question → search your vector index
        2. Pull the most relevant text chunks
        3. Feed those chunks as context to the LLM
        4. The LLM can ONLY answer from what's in the context

    Every answer becomes traceable back to a source chunk.
    This is the architecture behind ChatPDF, NotebookLM, and every
    enterprise document Q&A system in production today.

    ── FLOW ─────────────────────────────────────────────────────────────

        User question
             │
             ▼
        ollama.embed(question)          → 768D query vector
             │
             ▼
        doc_db.search(query_vector)     → top-k relevant text chunks
             │
             ▼
        _build_prompt(question, chunks) → grounded prompt
             │
             ▼
        ollama.generate(prompt)         → answer grounded in your docs
             │
             ▼
        { answer, sources, timing }

    ── PARAMETERS ───────────────────────────────────────────────────────
    k      : how many chunks to retrieve (more = more context, slower)
    metric : distance metric for retrieval (cosine works best for text)
    algo   : search algorithm (hnsw is fastest)
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions strictly based "
        "on the provided context.\n\n"
        "Rules:\n"
        "- Only use information explicitly present in the context below.\n"
        "- If the context does not contain enough information to answer, "
        "say exactly: 'I don't have enough information in the provided "
        "documents to answer this.'\n"
        "- Be concise, accurate, and direct.\n"
        "- Never make up facts not present in the context."
    )

    def __init__(
        self,
        doc_db: DocumentDB,
        ollama: OllamaClient,
        k:      int = 5,
        metric: str = "cosine",
        algo:   str = "hnsw",
    ):
        self.doc_db = doc_db
        self.ollama = ollama
        self.k      = k
        self.metric = metric
        self.algo   = algo

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Assemble the final prompt from retrieved chunks + the question.

        Format:
            CONTEXT:
            [Source: doc_id | Chunk 1 of 5]
            <chunk text>

            [Source: doc_id | Chunk 2 of 5]
            <chunk text>

            QUESTION:
            <user question>

            ANSWER:

        The LLM sees the context BEFORE the question — this is standard
        RAG prompt structure. The "ANSWER:" at the end signals the model
        to begin its response.
        """
        context_blocks = []

        for chunk in chunks:
            header = (
                f"[Source: {chunk['doc_id']} | "
                f"Chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']}]"
            )
            context_blocks.append(f"{header}\n{chunk['text']}")

        context = "\n\n".join(context_blocks)

        return (
            f"CONTEXT:\n"
            f"{context}\n\n"
            f"QUESTION:\n"
            f"{question}\n\n"
            f"ANSWER:"
        )

    # ------------------------------------------------------------------ #
    #  Core RAG call                                                       #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        k:        Optional[int] = None,
        metric:   Optional[str] = None,
        algo:     Optional[str] = None,
    ) -> Dict:
        """
        Ask a question — retrieve relevant chunks, generate a grounded answer.

        Args:
            question : plain-text user question
            k        : override default chunk retrieval count
            metric   : override default distance metric
            algo     : override default search algorithm

        Returns:
            {
                question     : original question
                answer       : LLM-generated answer (grounded in docs)
                sources      : list of chunks used as context
                               each has: text, doc_id, chunk_index,
                                         total_chunks, distance
                retrieval_ms : time taken for vector search (ms)
                k            : number of chunks retrieved
            }

        Raises:
            ValueError  : empty question
            RuntimeError: Ollama offline or model not pulled
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k      = k      or self.k
        metric = metric or self.metric
        algo   = algo   or self.algo

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        retrieval = self.doc_db.search(
            query  = question,
            k      = k,
            metric = metric,
            algo   = algo,
        )
        chunks = retrieval["results"]

        if not chunks:
            return {
                "question":     question,
                "answer":       (
                    "No relevant documents found. "
                    "Please add documents first via POST /documents/add."
                ),
                "sources":      [],
                "retrieval_ms": retrieval["elapsed_ms"],
                "k":            k,
            }

        # ── Step 2: Build grounded prompt ─────────────────────────────────
        prompt = self._build_prompt(question, chunks)

        # ── Step 3: Generate answer ───────────────────────────────────────
        # This is the only LLM call — and the model can only use
        # what's in the context we just built.
        answer = self.ollama.generate(
            prompt = prompt,
            system = self.SYSTEM_PROMPT,
        )

        return {
            "question":     question,
            "answer":       answer,
            "sources":      chunks,
            "retrieval_ms": retrieval["elapsed_ms"],
            "k":            k,
        }