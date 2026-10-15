from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class VectorRecord(BaseModel):
    id: str = Field(..., description="Unique identifier for the vector")
    vector: List[float] = Field(..., min_length=2, description="Dense float embedding")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BatchInsertRequest(BaseModel):
    records: List[VectorRecord]

class SearchRequest(BaseModel):
    query_vector: List[float] = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=50)
    ef_search: int = Field(default=32, ge=4, le=256)

class SearchResultItem(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    visited_nodes: int
    compression_ratio: float

class EngineStatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    max_layer: int
    quantization_enabled: bool
    raw_memory_bytes: int
    compressed_memory_bytes: int
    compression_ratio: float
