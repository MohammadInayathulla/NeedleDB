import numpy as np
from typing import Any, Dict, List, Tuple

from .brute_force import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW


# ── Demo Data ─────────────────────────────────────────────────────────────
# 20 hand-crafted 16D vectors across 4 semantic categories.
# Each category occupies a distinct "region" of 16D space so that
# semantic clustering is visible when projected to 2D later (Day 9).

DEMO_VECTORS = [
    # ── Computer Science (high values in dims 0-3) ──────────────────────
    {"id": "cs_001", "label": "binary search",      "category": "cs",
     "vector": [0.90,0.80,0.70,0.60, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "cs_002", "label": "linked list",         "category": "cs",
     "vector": [0.80,0.90,0.60,0.70, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "cs_003", "label": "dynamic programming", "category": "cs",
     "vector": [0.70,0.70,0.90,0.80, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "cs_004", "label": "hash table",          "category": "cs",
     "vector": [0.60,0.80,0.70,0.90, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "cs_005", "label": "neural network",      "category": "cs",
     "vector": [0.80,0.60,0.80,0.70, 0.20,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},

    # ── Mathematics (high values in dims 4-7) ───────────────────────────
    {"id": "ma_001", "label": "linear algebra",      "category": "math",
     "vector": [0.10,0.10,0.10,0.10, 0.90,0.80,0.70,0.60, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "ma_002", "label": "calculus",            "category": "math",
     "vector": [0.10,0.10,0.10,0.10, 0.80,0.90,0.60,0.70, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "ma_003", "label": "probability theory",  "category": "math",
     "vector": [0.10,0.10,0.10,0.10, 0.70,0.70,0.90,0.80, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "ma_004", "label": "number theory",       "category": "math",
     "vector": [0.10,0.10,0.10,0.10, 0.60,0.80,0.70,0.90, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},
    {"id": "ma_005", "label": "statistics",          "category": "math",
     "vector": [0.10,0.10,0.10,0.10, 0.80,0.60,0.80,0.70, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10]},

    # ── Food (high values in dims 8-11) ─────────────────────────────────
    {"id": "fd_001", "label": "sushi",               "category": "food",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.90,0.80,0.70,0.60, 0.10,0.10,0.10,0.10]},
    {"id": "fd_002", "label": "pizza",               "category": "food",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.80,0.90,0.60,0.70, 0.10,0.10,0.10,0.10]},
    {"id": "fd_003", "label": "tacos",               "category": "food",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.70,0.70,0.90,0.80, 0.10,0.10,0.10,0.10]},
    {"id": "fd_004", "label": "ramen",               "category": "food",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.60,0.80,0.70,0.90, 0.10,0.10,0.10,0.10]},
    {"id": "fd_005", "label": "pasta",               "category": "food",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.80,0.60,0.80,0.70, 0.10,0.10,0.10,0.10]},

    # ── Sports (high values in dims 12-15) ──────────────────────────────
    {"id": "sp_001", "label": "basketball",          "category": "sports",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.90,0.80,0.70,0.60]},
    {"id": "sp_002", "label": "football",            "category": "sports",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.80,0.90,0.60,0.70]},
    {"id": "sp_003", "label": "tennis",              "category": "sports",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.70,0.70,0.90,0.80]},
    {"id": "sp_004", "label": "swimming",            "category": "sports",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.60,0.80,0.70,0.90]},
    {"id": "sp_005", "label": "cricket",             "category": "sports",
     "vector": [0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.10,0.10,0.10,0.10, 0.80,0.60,0.80,0.70]},
]


# ── VectorDB ──────────────────────────────────────────────────────────────

class VectorDB:
    """
    Unified interface over BruteForce, KDTree, and HNSW.

    ── WHY THIS EXISTS ───────────────────────────────────────────────────
    Each algorithm has different trade-offs:
        BruteForce  → always exact, always slow at scale          O(N·d)
        KDTree      → exact, fast for low dims, fails at high d   O(log N) avg
        HNSW        → approximate, fast at any dimension          O(log N)

    VectorDB keeps all three in sync. Every insert and delete is applied
    to all three simultaneously, so you can run the same query through
    all three and compare speed vs accuracy side-by-side.

    ── BENCHMARK FEATURE ─────────────────────────────────────────────────
    The benchmark() method is the star — it runs all three on the same
    query and computes HNSW recall (how many of HNSW's results match the
    exact brute-force ground truth). This is the killer demo for the UI.
    """

    def __init__(
        self,
        M:               int = 16,
        ef_construction: int = 200,
        ef_search:       int = 50,
    ):
        self.brute_force = BruteForce()
        self.kdtree      = KDTree()
        self.hnsw        = HNSW(M=M, ef_construction=ef_construction, ef_search=ef_search)

        # Master record — id → { vector, metadata }
        self._items: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def insert(
        self,
        id:       str,
        vector:   np.ndarray,
        metadata: Any = None,
        metric:   str = "cosine",
    ) -> None:
        """Insert a vector into all three indexes simultaneously."""
        vec  = np.array(vector, dtype=np.float32)
        meta = metadata or {}

        self.brute_force.insert(id, vec, meta)
        self.kdtree.insert(id, vec, meta)
        self.hnsw.insert(id, vec, meta, metric=metric)

        self._items[id] = {"vector": vec.tolist(), "metadata": meta}

    def delete(self, id: str) -> bool:
        """Delete a vector from all three indexes simultaneously."""
        if id not in self._items:
            return False

        self.brute_force.delete(id)
        self.kdtree.delete(id)
        self.hnsw.delete(id)
        del self._items[id]
        return True

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query:  np.ndarray,
        k:      int = 5,
        metric: str = "cosine",
        algo:   str = "hnsw",
    ) -> Dict:
        """
        Search using one specific algorithm.

        Args:
            algo : "hnsw" | "kdtree" | "brute_force"

        Returns:
            {
                algo, metric, k,
                results    : [{ id, distance, metadata }, ...],
                elapsed_ms : float
            }
        """
        algo  = algo.lower().replace("-", "_").replace(" ", "_")
        query = np.array(query, dtype=np.float32)

        engines = {
            "hnsw":        self.hnsw,
            "kdtree":      self.kdtree,
            "brute_force": self.brute_force,
        }

        if algo not in engines:
            raise ValueError(
                f"Unknown algo '{algo}'. Choose from: {list(engines.keys())}"
            )

        results, elapsed_ms = engines[algo].search(query, k=k, metric=metric)

        return {
            "algo":       algo,
            "metric":     metric,
            "k":          k,
            "results":    results,
            "elapsed_ms": elapsed_ms,
        }

    def benchmark(
        self,
        query:  np.ndarray,
        k:      int = 5,
        metric: str = "cosine",
    ) -> Dict:
        """
        Run all three algorithms on the same query and compare.

        Also computes HNSW recall against brute-force ground truth:
            recall = |hnsw_results ∩ brute_force_results| / k

        A recall of 1.0 means HNSW found exactly the same neighbors as
        brute force — perfect accuracy despite being approximate.

        Returns:
            {
                metric, k,
                benchmark: {
                    brute_force : { results, elapsed_ms },
                    kdtree      : { results, elapsed_ms },
                    hnsw        : { results, elapsed_ms },
                },
                hnsw_recall : float   (0.0 – 1.0)
            }
        """
        query  = np.array(query, dtype=np.float32)
        report = {}

        for algo in ("brute_force", "kdtree", "hnsw"):
            results, elapsed_ms = getattr(self, algo).search(
                query, k=k, metric=metric
            )
            report[algo] = {
                "results":    results,
                "elapsed_ms": elapsed_ms,
            }

        # Recall: fraction of HNSW results that match ground truth
        bf_ids   = {r["id"] for r in report["brute_force"]["results"]}
        hnsw_ids = {r["id"] for r in report["hnsw"]["results"]}
        recall   = len(bf_ids & hnsw_ids) / len(bf_ids) if bf_ids else 1.0

        return {
            "metric":      metric,
            "k":           k,
            "benchmark":   report,
            "hnsw_recall": round(recall, 4),
        }

    # ------------------------------------------------------------------ #
    #  Demo data                                                           #
    # ------------------------------------------------------------------ #

    def load_demo_data(self, metric: str = "cosine") -> int:
        """
        Load the 20 pre-built semantic demo vectors.
        Returns the number of vectors inserted.
        """
        for item in DEMO_VECTORS:
            self.insert(
                id       = item["id"],
                vector   = item["vector"],
                metadata = {
                    "label":    item["label"],
                    "category": item["category"],
                },
                metric=metric,
            )
        return len(DEMO_VECTORS)

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def list_items(self) -> List[Dict]:
        """Return all stored items — id + metadata, no raw vectors."""
        return [
            {"id": id, "metadata": data["metadata"]}
            for id, data in self._items.items()
        ]

    def get_item(self, id: str) -> Dict | None:
        """Return a single item including its raw vector."""
        return self._items.get(id)

    def get_stats(self) -> Dict:
        """Return database-level statistics."""
        return {
            "total_vectors": len(self._items),
            "hnsw_info":     self.hnsw.get_graph_info(),
        }

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"VectorDB(vectors={len(self._items)})"
