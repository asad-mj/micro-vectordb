import math
import heapq
import random
import numpy as np
from typing import Dict, List, Tuple, Set

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) Graph Index implemented from first principles.
    Provides O(log N) approximate nearest neighbor search via multi-layer skip-list graphs.
    """

    def __init__(
        self,
        dim: int,
        m_neighbors: int = 16,
        ef_construction: int = 64,
        seed: int = 42
    ):
        self.dim = dim
        self.M = m_neighbors
        self.M_max0 = m_neighbors * 2
        self.ef_construction = ef_construction
        self.level_mult = 1.0 / math.log(max(2, self.M))
        self.rng = random.Random(seed)

        self.vectors: Dict[str, np.ndarray] = {}
        self.layers: Dict[int, Dict[str, Set[str]]] = {}
        self.node_levels: Dict[str, int] = {}
        self.entry_point: str | None = None
        self.max_layer: int = -1

    def _sample_level(self) -> int:
        return int(-math.log(max(1e-9, self.rng.random())) * self.level_mult)

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Returns cosine distance in [0, 2] where 0.0 is identical."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, 1.0 - sim)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_id: str,
        ef: int,
        layer: int
    ) -> Tuple[List[Tuple[float, str]], int]:
        """
        Greedy beam search within a single layer.
        Returns sorted list of (distance, node_id) and count of visited nodes.
        """
        visited: Set[str] = {entry_id}
        d_init = self._cosine_distance(query, self.vectors[entry_id])

        # candidates is a min-heap of (dist, id)
        candidates: List[Tuple[float, str]] = [(d_init, entry_id)]
        # results is a max-heap of (-dist, id) to maintain top-ef closest
        results: List[Tuple[float, str]] = [(-d_init, entry_id)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -results[0][0]

            if c_dist > furthest_dist:
                break

            neighbors = self.layers.get(layer, {}).get(c_id, set())
            for nb_id in neighbors:
                if nb_id not in visited:
                    visited.add(nb_id)
                    dist = self._cosine_distance(query, self.vectors[nb_id])
                    if dist < -results[0][0] or len(results) < ef:
                        heapq.heappush(candidates, (dist, nb_id))
                        heapq.heappush(results, (-dist, nb_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        sorted_results = sorted([(-neg_d, nid) for neg_d, nid in results], key=lambda x: x[0])
        return sorted_results, len(visited)

    def add_item(self, item_id: str, vector: np.ndarray) -> None:
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vector.shape[0]}")

        self.vectors[item_id] = vector.astype(np.float32)
        level = self._sample_level()
        self.node_levels[item_id] = level

        for l in range(level + 1):
            if l not in self.layers:
                self.layers[l] = {}
            if item_id not in self.layers[l]:
                self.layers[l][item_id] = set()

        if self.entry_point is None:
            self.entry_point = item_id
            self.max_layer = level
            return

        curr_ep = self.entry_point

        # Phase 1: Zoom down from top layer to level + 1 with ef=1
        for l in range(self.max_layer, level, -1):
            if l in self.layers and curr_ep in self.layers[l]:
                nearest, _ = self._search_layer(vector, curr_ep, ef=1, layer=l)
                curr_ep = nearest[0][1]

        # Phase 2: Connect bidirectional edges from min(level, max_layer) down to 0
        for l in range(min(level, self.max_layer), -1, -1):
            nearest, _ = self._search_layer(vector, curr_ep, ef=self.ef_construction, layer=l)
            m_limit = self.M_max0 if l == 0 else self.M
            selected = [nid for _, nid in nearest[:m_limit] if nid != item_id]

            for nb_id in selected:
                self.layers[l][item_id].add(nb_id)
                self.layers[l][nb_id].add(item_id)

                # Prune connections if neighbor exceeds maximum degree
                if len(self.layers[l][nb_id]) > m_limit:
                    nb_vec = self.vectors[nb_id]
                    scored = [
                        (self._cosine_distance(nb_vec, self.vectors[peer]), peer)
                        for peer in self.layers[l][nb_id]
                    ]
                    scored.sort(key=lambda x: x[0])
                    self.layers[l][nb_id] = {peer for _, peer in scored[:m_limit]}

            if nearest:
                curr_ep = nearest[0][1]

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = item_id

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        ef_search: int = 32
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Executes hierarchical search across layers.
        Returns list of (id, cosine_similarity_score) and total visited nodes.
        """
        if self.entry_point is None or len(self.vectors) == 0:
            return [], 0

        curr_ep = self.entry_point
        total_visited = 0

        for l in range(self.max_layer, 0, -1):
            if l in self.layers and curr_ep in self.layers[l]:
                nearest, visited = self._search_layer(query, curr_ep, ef=1, layer=l)
                total_visited += visited
                curr_ep = nearest[0][1]

        ef = max(ef_search, top_k)
        nearest_l0, visited_l0 = self._search_layer(query, curr_ep, ef=ef, layer=0)
        total_visited += visited_l0

        results = [
            (nid, round(1.0 - dist, 4))
            for dist, nid in nearest_l0[:top_k]
        ]
        return results, total_visited
