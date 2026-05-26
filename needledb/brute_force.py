import time
import numpy as np
from typing import Any, Dict, List, Tuple

from .metrics import get_metric


class BruteForce:
    """
    Brute Force K-Nearest Neighbor Search.

    How it works:
        For every search query, compute the distance between the query
        and EVERY stored vector, then return the k smallest distances.

    Complexity:
        Insert : O(1)
        Search : O(N * d)   — N vectors, d dimensions
        Delete : O(1)

    This is the baseline. It is always exact (never misses a neighbor)
    but gets slow as N grows. Every other algorithm (KD-Tree, HNSW) is
    trying to beat this while staying as accurate as possible.
    """

    def __init__(self):
        self.vectors:  Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any]        = {}

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def insert(self, id: str, vector: np.ndarray, metadata: Any = None) -> None:
        """Store a vector under the given id."""
        self.vectors[id]  = np.array(vector, dtype=np.float32)
        self.metadata[id] = metadata or {}

    def delete(self, id: str) -> bool:
        """Remove a vector by id. Returns True if it existed."""
        if id in self.vectors:
            del self.vectors[id]
            del self.metadata[id]
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query:  np.ndarray,
        k:      int = 5,
        metric: str = "cosine",
    ) -> Tuple[List[Dict], float]:
        """
        Find the k nearest neighbors to `query`.

        Returns:
            results    : list of { id, distance, metadata }  sorted by distance
            elapsed_ms : wall-clock time the search took (milliseconds)
        """
        if not self.vectors:
            return [], 0.0

        dist_fn = get_metric(metric)
        query   = np.array(query, dtype=np.float32)

        start = time.perf_counter()

        # Core loop — this is the O(N*d) part
        scores = [
            (id, dist_fn(query, vec))
            for id, vec in self.vectors.items()
        ]
        scores.sort(key=lambda x: x[1])   # ascending: closest first

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            {
                "id":       id,
                "distance": round(dist, 6),
                "metadata": self.metadata[id],
            }
            for id, dist in scores[:k]
        ]

        return results, round(elapsed_ms, 4)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.vectors)

    def __repr__(self):
        return f"BruteForce(vectors={len(self.vectors)})"
