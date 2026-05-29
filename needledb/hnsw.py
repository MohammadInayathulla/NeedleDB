import heapq
import math
import random
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

from .metrics import get_metric


class _HNSWNode:
    """A single node in the HNSW graph."""
    __slots__ = ("id", "vector", "metadata", "max_layer", "neighbors")

    def __init__(self, id: str, vector: np.ndarray, metadata: Any, max_layer: int):
        self.id        = id
        self.vector    = vector
        self.metadata  = metadata
        self.max_layer = max_layer
        # neighbors[layer] = list of neighbor IDs at that layer
        self.neighbors: List[List[str]] = [[] for _ in range(max_layer + 1)]


class HNSW:
    """
    HNSW — Hierarchical Navigable Small World Graph.

    The approximate nearest neighbor algorithm used by every major
    production vector database (Pinecone, Weaviate, Chroma, Qdrant, Milvus).

    ── CORE IDEA ────────────────────────────────────────────────────────────
    Build a multi-layer graph where:
        Layer 0  → ALL nodes, many short-range connections  (fine detail)
        Layer 1  → subset of nodes, medium connections
        Layer 2+ → exponentially fewer nodes, long-range connections (highway)

    Searching starts at the top layer (few nodes, fast navigation) and
    greedily descends, zooming into the right neighborhood before doing
    a thorough search at layer 0. Like using a world map → country map
    → city map instead of scanning every street at once.

    ── INSERT ───────────────────────────────────────────────────────────────
    1. Randomly assign this node a max_layer using an exponential distribution
       (most nodes → layer 0, very few → higher layers)
    2. From the graph's top layer down to max_layer+1: greedy descent with ef=1
    3. From max_layer down to layer 0: beam search (ef=ef_construction),
       connect to M nearest neighbors bidirectionally, prune if over limit
    4. If max_layer > graph's current top → this node becomes new entry point

    ── SEARCH ───────────────────────────────────────────────────────────────
    1. Start at entry point (top layer), greedy descent with ef=1 per layer
    2. At layer 0: full beam search with ef=ef_search
    3. Return top k from layer-0 results

    ── COMPLEXITY ───────────────────────────────────────────────────────────
    Insert : O(log N)
    Search : O(log N)   — approximate, may miss a small % of true neighbors
    Delete : O(M * log N)

    ── PARAMETERS ───────────────────────────────────────────────────────────
    M               : max connections per node per layer (default 16)
                      Higher M → better recall, more memory, slower insert
    ef_construction : beam width during index build (default 200)
                      Higher → better graph quality, slower inserts
    ef_search       : beam width during query (default 50)
                      Higher → better recall, slower search
    """

    def __init__(
        self,
        M:               int = 16,
        ef_construction: int = 200,
        ef_search:       int = 50,
    ):
        self.M               = M
        self.M0              = 2 * M                 # layer 0 allows more connections
        self.ef_construction = ef_construction
        self.ef_search       = ef_search
        self.mL              = 1.0 / math.log(M)    # level generation normalizer

        self._nodes:       Dict[str, _HNSWNode] = {}
        self._entry_point: Optional[str]        = None   # id of topmost node
        self._max_layer:   int                  = -1     # current highest layer

    # ------------------------------------------------------------------ #
    #  Level generation                                                    #
    # ------------------------------------------------------------------ #

    def _random_level(self) -> int:
        """
        Sample a random max layer for a new node.

        Formula from the HNSW paper:
            level = floor( -ln(uniform(0,1)) * mL )

        This produces an exponential distribution:
            ~63% of nodes stay at layer 0
            ~23% reach layer 1
            ~09% reach layer 2
            and so on — exponentially fewer at each level.
        """
        return int(-math.log(random.uniform(0.0, 1.0)) * self.mL)

    # ------------------------------------------------------------------ #
    #  Core search primitive (single layer)                                #
    # ------------------------------------------------------------------ #

    def _search_layer(
        self,
        query_vec: np.ndarray,
        entry_ids: List[str],
        ef:        int,
        layer:     int,
        dist_fn,
    ) -> List[Tuple[float, str]]:
        """
        Greedy beam search within a single graph layer.

        Uses two heaps:
            candidates : min-heap by distance — what to explore next
            found      : max-heap of size ef  — best ef neighbors seen so far

        Stops when the closest unexplored candidate is farther than
        the worst neighbor in `found` (no improvement possible).

        Returns: list of (distance, id) sorted ascending, length ≤ ef
        """
        visited: Set[str] = set(entry_ids)
        candidates: List[Tuple[float, str]] = []   # min-heap
        found:      List[Tuple[float, str]] = []   # max-heap (store neg dist)

        for eid in entry_ids:
            d = dist_fn(query_vec, self._nodes[eid].vector)
            heapq.heappush(candidates, (d, eid))
            heapq.heappush(found, (-d, eid))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)    # closest candidate
            worst_found  = -found[0][0]                  # farthest in best-ef set

            # Termination: closest unseen is already worse than our worst best
            if c_dist > worst_found:
                break

            for nb_id in self._nodes[c_id].neighbors[layer]:
                if nb_id in visited:
                    continue
                visited.add(nb_id)

                nb_dist = dist_fn(query_vec, self._nodes[nb_id].vector)
                worst   = -found[0][0]

                if nb_dist < worst or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_id))
                    heapq.heappush(found, (-nb_dist, nb_id))
                    if len(found) > ef:
                        heapq.heappop(found)    # evict farthest

        return sorted([(-neg_d, id) for neg_d, id in found])

    # ------------------------------------------------------------------ #
    #  Neighbor selection                                                  #
    # ------------------------------------------------------------------ #

    def _select_neighbors(
        self,
        candidates: List[Tuple[float, str]],
        M:          int,
    ) -> List[str]:
        """
        Pick M nearest neighbors from a candidate list.
        Simple greedy selection: sort by distance, take top M.
        """
        return [id for _, id in sorted(candidates, key=lambda x: x[0])[:M]]

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
        """
        Insert a new vector into the HNSW graph.

        Phase 1 — fast descent (layers above node's max_layer):
            Use ef=1 (greedy, single neighbor) to descend quickly
            to the layer where this node will actually live.

        Phase 2 — connect (layers max_layer down to 0):
            At each layer, run a full beam search (ef=ef_construction)
            to find the best neighbors, wire them bidirectionally,
            and prune any neighbor that now exceeds M connections.
        """
        dist_fn   = get_metric(metric)
        max_layer = self._random_level()
        node      = _HNSWNode(
            id        = id,
            vector    = np.array(vector, dtype=np.float32),
            metadata  = metadata or {},
            max_layer = max_layer,
        )
        self._nodes[id] = node

        # ── First node ever ───────────────────────────────────────────────
        if self._entry_point is None:
            self._entry_point = id
            self._max_layer   = max_layer
            return

        entry_ids = [self._entry_point]

        # ── Phase 1: descend above max_layer with ef=1 ────────────────────
        for layer in range(self._max_layer, max_layer, -1):
            results   = self._search_layer(node.vector, entry_ids, ef=1, layer=layer, dist_fn=dist_fn)
            entry_ids = [results[0][1]]

        # ── Phase 2: beam search + connect at each layer ──────────────────
        for layer in range(min(max_layer, self._max_layer), -1, -1):
            M_layer = self.M0 if layer == 0 else self.M

            candidates = self._search_layer(
                node.vector, entry_ids,
                ef=self.ef_construction, layer=layer, dist_fn=dist_fn,
            )

            neighbors = self._select_neighbors(candidates, M_layer)
            node.neighbors[layer] = neighbors

            # Wire reverse edges (bidirectional graph)
            for nb_id in neighbors:
                nb_node = self._nodes[nb_id]
                if layer <= nb_node.max_layer:
                    nb_node.neighbors[layer].append(id)

                    # Prune if this neighbor now exceeds M connections
                    if len(nb_node.neighbors[layer]) > M_layer:
                        pruned = sorted(
                            nb_node.neighbors[layer],
                            key=lambda nid: dist_fn(
                                nb_node.vector, self._nodes[nid].vector
                            ),
                        )
                        nb_node.neighbors[layer] = pruned[:M_layer]

            entry_ids = [id for _, id in candidates]

        # ── Update entry point if this node lives on a higher layer ───────
        if max_layer > self._max_layer:
            self._entry_point = id
            self._max_layer   = max_layer

    def delete(self, id: str) -> bool:
        """
        Remove a node and all its edges from the graph.
        If the deleted node was the entry point, elect a new one.
        """
        if id not in self._nodes:
            return False

        node = self._nodes[id]

        # Remove incoming edges from every neighbor
        for layer in range(node.max_layer + 1):
            for nb_id in node.neighbors[layer]:
                if nb_id in self._nodes and layer <= self._nodes[nb_id].max_layer:
                    self._nodes[nb_id].neighbors[layer] = [
                        nid for nid in self._nodes[nb_id].neighbors[layer]
                        if nid != id
                    ]

        del self._nodes[id]

        # Re-elect entry point if needed
        if self._entry_point == id:
            if self._nodes:
                self._entry_point = max(
                    self._nodes,
                    key=lambda nid: self._nodes[nid].max_layer,
                )
                self._max_layer = self._nodes[self._entry_point].max_layer
            else:
                self._entry_point = None
                self._max_layer   = -1

        return True

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
        Find k approximate nearest neighbors.

        Descends layer by layer with ef=1 (fast highway traversal),
        then does a thorough beam search at layer 0 with ef=ef_search.
        """
        if self._entry_point is None:
            return [], 0.0

        dist_fn   = get_metric(metric)
        query     = np.array(query, dtype=np.float32)
        entry_ids = [self._entry_point]

        start = time.perf_counter()

        # Fast descent: layers top → 1
        for layer in range(self._max_layer, 0, -1):
            results   = self._search_layer(query, entry_ids, ef=1, layer=layer, dist_fn=dist_fn)
            entry_ids = [results[0][1]]

        # Thorough search at layer 0
        ef      = max(self.ef_search, k)
        results = self._search_layer(query, entry_ids, ef=ef, layer=0, dist_fn=dist_fn)

        elapsed_ms = (time.perf_counter() - start) * 1000

        top_k = [
            {
                "id":       id,
                "distance": round(dist, 6),
                "metadata": self._nodes[id].metadata,
            }
            for dist, id in results[:k]
        ]

        return top_k, round(elapsed_ms, 4)

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def get_graph_info(self) -> Dict:
        """Return stats about the HNSW graph — useful for the API later."""
        if not self._nodes:
            return {"nodes": 0, "layers": 0}

        layer_counts: Dict[int, int] = {}
        for node in self._nodes.values():
            for layer in range(node.max_layer + 1):
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        avg_connections: Dict[int, float] = {}
        for layer, count in layer_counts.items():
            total = sum(
                len(n.neighbors[layer])
                for n in self._nodes.values()
                if layer <= n.max_layer
            )
            avg_connections[layer] = round(total / count, 2) if count else 0

        return {
            "nodes":           len(self._nodes),
            "max_layer":       self._max_layer,
            "entry_point":     self._entry_point,
            "layer_counts":    layer_counts,
            "avg_connections": avg_connections,
            "M":               self.M,
            "ef_construction": self.ef_construction,
            "ef_search":       self.ef_search,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self._nodes)

    def __repr__(self):
        return (
            f"HNSW(vectors={len(self._nodes)}, "
            f"max_layer={self._max_layer}, "
            f"M={self.M}, ef_construction={self.ef_construction})"
        )
