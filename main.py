from fastapi import FastAPI, HTTPException
from src.engine import MicroVectorDB
from src.schemas import (
    BatchInsertRequest,
    SearchRequest,
    SearchResponse,
    EngineStatsResponse
)

app = FastAPI(
    title="micro-vectordb: HNSW & SQ8 Vector Engine",
    description="High-performance ANN vector database implemented from scratch with HNSW graph indexing and 8-bit scalar quantization.",
    version="1.0.0"
)

db = MicroVectorDB(dim=4, use_quantization=True)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "engine": "micro-vectordb", "version": "1.0.0"}

@app.get("/stats", response_model=EngineStatsResponse, tags=["Telemetry"])
def get_database_stats():
    return db.get_stats()

@app.post("/insert", tags=["Operations"])
def insert_vectors(payload: BatchInsertRequest):
    try:
        inserted = db.insert_batch(payload.records)
        return {"status": "success", "inserted_count": inserted}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search", response_model=SearchResponse, tags=["Operations"])
def search_vectors(payload: SearchRequest):
    try:
        return db.query(
            query_vector=payload.query_vector,
            top_k=payload.top_k,
            ef_search=payload.ef_search
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
