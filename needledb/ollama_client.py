import requests
from typing import Dict, List, Optional


OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"
GEN_MODEL       = "llama3.2"


class OllamaClient:
    """
    HTTP client for Ollama — a local LLM server.

    Ollama runs AI models entirely on your machine.
    No API key. No internet. No cost per token.

    Two models power NeedleDB:
        nomic-embed-text : text → 768D vector  (semantic embedding)
        llama3.2         : text → text         (answer generation, Day 8)

    ── WHY HTTP DIRECTLY? ───────────────────────────────────────────────
    We call Ollama's REST API manually instead of using the ollama
    Python library. This means you can see exactly what goes over the
    wire — the same JSON you'd send from curl or any other language.
    Transparency beats convenience for learning.

    ── WHAT IS AN EMBEDDING? ────────────────────────────────────────────
    An embedding is a list of 768 floats that encodes the *meaning*
    of a piece of text. The magic property:

        "binary search" and "algorithm efficiency" → nearby in 768D space
        "binary search" and "sushi"               → far apart in 768D space

    This is what makes semantic search work. Instead of matching keywords,
    NeedleDB matches *meaning*.
    """

    def __init__(
        self,
        base_url:    str = OLLAMA_BASE_URL,
        embed_model: str = EMBED_MODEL,
        gen_model:   str = GEN_MODEL,
    ):
        self.base_url    = base_url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model   = gen_model

    # ------------------------------------------------------------------ #
    #  Health checks                                                       #
    # ------------------------------------------------------------------ #

    def is_online(self) -> bool:
        """Ping Ollama. Returns True if the server is reachable."""
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> List[str]:
        """Return names of all locally available models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def status(self) -> Dict:
        """
        Full status report — is Ollama up? Are both models pulled?

        Returns:
            {
                ollama      : "ONLINE" | "OFFLINE"
                embed_model : model name
                gen_model   : model name
                embed_ready : bool
                gen_ready   : bool
                models      : [list of all pulled models]
            }
        """
        if not self.is_online():
            return {
                "ollama":      "OFFLINE",
                "message":     "Run 'ollama serve' in a terminal.",
                "embed_model": self.embed_model,
                "gen_model":   self.gen_model,
                "embed_ready": False,
                "gen_ready":   False,
            }

        models      = self.list_models()
        embed_ready = any(self.embed_model in m for m in models)
        gen_ready   = any(self.gen_model   in m for m in models)

        return {
            "ollama":      "ONLINE",
            "models":      models,
            "embed_model": self.embed_model,
            "gen_model":   self.gen_model,
            "embed_ready": embed_ready,
            "gen_ready":   gen_ready,
        }

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        """
        Convert text into a 768-dimensional embedding vector.

        Sends a POST to Ollama's /api/embeddings endpoint.
        The first call may take a few seconds (model loading).
        Subsequent calls are fast.

        Args:
            text : any string — a word, sentence, paragraph, or chunk

        Returns:
            list of 768 floats representing the semantic meaning of text

        Raises:
            ValueError  : if text is empty
            RuntimeError: if Ollama is offline or model is not pulled
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        payload = {
            "model":  self.embed_model,
            "prompt": text.strip(),
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json    = payload,
                timeout = 60,     # generous — first call loads model into RAM
            )
            r.raise_for_status()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is offline. Run 'ollama serve' in a terminal."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama embed failed: {e}\n"
                f"Is '{self.embed_model}' pulled? Run: ollama pull {self.embed_model}"
            )

        embedding = r.json().get("embedding")

        if not embedding:
            raise RuntimeError("Ollama returned an empty embedding.")

        return embedding    # 768 floats

    # ------------------------------------------------------------------ #
    #  Text generation  (used in RAG pipeline on Day 8)                   #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate a text response from llama3.2.

        Non-streaming for now — waits for the full response.
        Takes 10–30 seconds on a laptop CPU, which is normal.

        Args:
            prompt : the user's question or instruction
            system : optional system prompt to constrain behaviour

        Returns:
            generated text as a string
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload: Dict = {
            "model":  self.gen_model,
            "prompt": prompt.strip(),
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json    = payload,
                timeout = 180,    # generation on CPU can be slow
            )
            r.raise_for_status()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is offline. Run 'ollama serve' in a terminal."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama generate failed: {e}\n"
                f"Is '{self.gen_model}' pulled? Run: ollama pull {self.gen_model}"
            )

        response = r.json().get("response", "").strip()

        if not response:
            raise RuntimeError("Ollama returned an empty response.")

        return response