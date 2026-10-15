import numpy as np
from typing import Dict, Any, List
from src.hnsw import HNSWIndex
from src.quantizer import ScalarQuantizerSQ8
from src.schemas import VectorRecord, SearchResultItem, SearchResponse, EngineStatsResponse

class MicroVectorDB:
    """
    Unified Vector Database Engine integrating HNSW indexing, SQ8 vector compression,
    and metadata storage.
    """

    def __init__(self, dim: int = 4, use_quantization: bool = True):
        self.dim = dim
        self.use_quantization = use_quantization
        self.index = HNSWIndex(dim=dim)
        self.quantizer = ScalarQuantizerSQ8(dim=dim)
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.quantized_codes: Dict[str, np.ndarray] = {}

    def insert_batch(self, records: List[VectorRecord]) -> int:
        if not records:
            return 0

        # Auto-adjust dimension on first insertion if database is empty
        if len(self.index.vectors) == 0:
            first_dim = len(records[0].vector)
            if first_dim != self.dim:
                self.dim = first_dim
                self.index = HNSWIndex(dim=first_dim)
                self.quantizer = ScalarQuantizerSQ8(dim=first_dim)

        matrix = np.array([r.vector for r in records], dtype=np.float32)
        if self.use_quantization:
            self.quantizer.calibrate(matrix)
            q_matrix = self.quantizer.encode(matrix)
            for idx, rec in enumerate(records):
                self.quantized_codes[rec.id] = q_matrix[idx]

        for idx, rec in enumerate(records):
            self.index.add_item(rec.id, matrix[idx])
            self.metadata_store[rec.id] = rec.metadata or {}

        return len(records)

    def query(self, query_vector: List[float], top_k: int = 5, ef_search: int = 32) -> SearchResponse:
        q_arr = np.array(query_vector, dtype=np.float32)
        if q_arr.shape[0] != self.dim:
            raise ValueError(f"Query dimension {q_arr.shape[0]} does not match index dimension {self.dim}")

        matches, visited = self.index.search(q_arr, top_k=top_k, ef_search=ef_search)
        items = [
            SearchResultItem(
                id=doc_id,
                score=score,
                metadata=self.metadata_store.get(doc_id, {})
            )
            for doc_id, score in matches
        ]
        stats = self.get_stats()
        return SearchResponse(
            results=items,
            visited_nodes=visited,
            compression_ratio=stats.compression_ratio
        )

    def get_stats(self) -> EngineStatsResponse:
        n_vecs = len(self.index.vectors)
        raw_bytes = n_vecs * self.dim * 4  # float32 = 4 bytes
        comp_bytes = n_vecs * self.dim * 1 if self.use_quantization else raw_bytes
        ratio = 4.0 if (self.use_quantization and n_vecs > 0) else 1.0

        return EngineStatsResponse(
            total_vectors=n_vecs,
            dimension=self.dim,
            max_layer=max(0, self.index.max_layer),
            quantization_enabled=self.use_quantization,
            raw_memory_bytes=raw_bytes,
            compressed_memory_bytes=comp_bytes,
            compression_ratio=ratio
        )
