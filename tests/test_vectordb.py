import numpy as np
from fastapi.testclient import TestClient
from main import app
from src.quantizer import ScalarQuantizerSQ8
from src.hnsw import HNSWIndex

def test_api_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

def test_scalar_quantizer_compression():
    rng = np.random.default_rng(0)
    data = rng.uniform(-5.0, 5.0, size=(100, 16)).astype(np.float32)
    sq = ScalarQuantizerSQ8(dim=16)
    encoded = sq.encode(data)

    assert encoded.dtype == np.uint8
    assert encoded.shape == (100, 16)
    mse = sq.compute_reconstruction_mse(data)
    assert mse < 0.01

def test_hnsw_exact_top1_recall():
    index = HNSWIndex(dim=8, m_neighbors=8, ef_construction=32, seed=42)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((50, 8)).astype(np.float32)

    for i in range(50):
        index.add_item(f"doc_{i}", vecs[i])

    results, visited = index.search(vecs[10], top_k=3, ef_search=32)
    assert len(results) == 3
    assert results[0][0] == "doc_10"
    assert results[0][1] > 0.99
    assert visited <= 50

def test_api_insert_and_search():
    with TestClient(app) as client:
        payload = {
            "records": [
                {"id": "v1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"tag": "first"}},
                {"id": "v2", "vector": [0.9, 0.1, 0.0, 0.0], "metadata": {"tag": "second"}},
                {"id": "v3", "vector": [0.0, 0.0, 1.0, 0.0], "metadata": {"tag": "orthogonal"}}
            ]
        }
        ins_res = client.post("/insert", json=payload)
        assert ins_res.status_code == 200
        assert ins_res.json()["inserted_count"] == 3

        search_res = client.post(
            "/search",
            json={"query_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2, "ef_search": 16}
        )
        assert search_res.status_code == 200
        body = search_res.json()
        assert body["results"][0]["id"] == "v1"
        assert body["compression_ratio"] == 4.0
