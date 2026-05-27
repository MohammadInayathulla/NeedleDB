import heapq
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .metrics import get_metric


class _Node:
    """A single node in the KD-Tree."""
    __slots__ = ("id", "vector", "metadata", "axis", "left", "right")

    def __init__(self, id: str, vector: np.ndarray, metadata: Any, axis: int):
        self.id       = id
        self.vector   = vector
        self.metadata = metadata
        self.axis     = axis        # which dimension this node splits on
        self.left:  Optional[_Node] = None
        self.right: Optional[_Node] = None


class KDTree:
    """
    KD-Tree (K-Dimensional Tree) — exact nearest neighbor search.

    ── BUILD ───────────────────────────────────────────────────────────────
    At each level, pick a splitting axis by cycling through dimensions:
    depth 0 → axis 0, depth 1 → axis 1, depth 2 → axis 2, ...
    Sort all points by that axis, pick the MEDIAN as the current node,
    recurse left (smaller values) and right (larger values).
    Result: a balanced binary tree that partitions space.

    ── SEARCH ──────────────────────────────────────────────────────────────
    Traverse like a BST using the query's value on the split axis.
    When you hit a leaf, unwind back up the tree. At each node ask:
        "Could ANY point in the subtree I skipped be closer than
         my current best neighbor?"
    The answer is NO if the perpendicular distance to the split plane
    is already larger than the current best distance → prune that subtree.
    This pruning is what makes KD-Tree faster than brute force.

    ── COMPLEXITY ──────────────────────────────────────────────────────────
    Build  : O(N log N)
    Search : O(log N) average  |  O(N) worst case
    Delete : O(N log N)  — requires a full rebuild (standard tradeoff)

    ── WEAKNESS ────────────────────────────────────────────────────────────
    Curse of dimensionality: in high dimensions (e.g. 768D embeddings)
    almost NO subtrees get pruned because all points are equidistant
    from the split plane. KD-Tree degrades to brute force at high dims.
    That is exactly why HNSW (Day 3) was invented.
    """

    def __init__(self):
        self._root:  Optional[_Node]                    = None
        self._store: Dict[str, Tuple[np.ndarray, Any]] = {}  # id → (vector, metadata)

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def insert(self, id: str, vector: np.ndarray, metadata: Any = None) -> None:
        """Insert a vector and rebuild the tree for balance."""
        self._store[id] = (np.array(vector, dtype=np.float32), metadata or {})
        self._rebuild()

    def delete(self, id: str) -> bool:
        """Remove a vector by id and rebuild the tree."""
        if id not in self._store:
            return False
        del self._store[id]
        self._rebuild()
        return True

    # ------------------------------------------------------------------ #
    #  Tree construction                                                   #
    # ------------------------------------------------------------------ #

    def _rebuild(self) -> None:
        """Reconstruct the entire tree from the current store."""
        items = [
            (id, vec, meta)
            for id, (vec, meta) in self._store.items()
        ]
        self._root = self._build(items, depth=0)

    def _build(self, items: List[Tuple], depth: int) -> Optional[_Node]:
        if not items:
            return None

        dims = items[0][1].shape[0]
        axis = depth % dims                        # cycle through all dimensions

        items.sort(key=lambda x: x[1][axis])       # sort along current axis
        mid = len(items) // 2                      # median index

        id, vec, meta = items[mid]
        node = _Node(id, vec, meta, axis)
        node.left  = self._build(items[:mid],     depth + 1)
        node.right = self._build(items[mid + 1:], depth + 1)
        return node

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
        Find k nearest neighbors using KD-Tree traversal + pruning.

        Returns:
            results    : list of { id, distance, metadata } sorted by distance
            elapsed_ms : wall-clock time taken in milliseconds
        """
        if self._root is None:
            return [], 0.0

        dist_fn = get_metric(metric)
        query   = np.array(query, dtype=np.float32)
        heap    = []  # max-heap of size k: (-distance, id, metadata)

        start = time.perf_counter()
        self._search(self._root, query, k, dist_fn, heap)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = sorted(
            [
                {
                    "id":       id,
                    "distance": round(-neg_d, 6),
                    "metadata": meta,
                }
                for neg_d, id, meta in heap
            ],
            key=lambda x: x["distance"],
        )

        return results, round(elapsed_ms, 4)

    def _search(
        self,
        node:    Optional[_Node],
        query:   np.ndarray,
        k:       int,
        dist_fn,
        heap:    list,
    ) -> None:
        """Recursive KD-Tree search with split-plane pruning."""
        if node is None:
            return

        dist = dist_fn(query, node.vector)

        # ── Update the max-heap of k best candidates ─────────────────────
        # We store (-dist) so the heap root is the FARTHEST of the k best.
        heapq.heappush(heap, (-dist, node.id, node.metadata))
        if len(heap) > k:
            heapq.heappop(heap)   # remove farthest if we exceed k

        # ── Decide which child to visit first ────────────────────────────
        axis = node.axis
        diff = float(query[axis] - node.vector[axis])   # signed dist to split plane

        close = node.left  if diff <= 0 else node.right
        away  = node.right if diff <= 0 else node.left

        # Always recurse into the side the query falls on
        self._search(close, query, k, dist_fn, heap)

        # ── Pruning check ─────────────────────────────────────────────────
        # Only visit the far side if the perpendicular distance to the
        # split plane (abs(diff)) is less than our current worst neighbor.
        # If it's larger, no point in the far subtree can beat what we have.
        worst = -heap[0][0] if heap else float("inf")
        if abs(diff) < worst:
            self._search(away, query, k, dist_fn, heap)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"KDTree(vectors={len(self._store)})"
