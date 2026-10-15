import streamlit as st
import numpy as np
import time
from src.engine import MicroVectorDB
from src.schemas import VectorRecord

st.set_page_config(page_title="micro-vectordb Lab", page_icon="⚡", layout="wide")
st.title("⚡ micro-vectordb: HNSW & SQ8 Vector Engine Workbench")
st.markdown("Interactive benchmarking of **Hierarchical Navigable Small World (HNSW)** indexing and **8-Bit Scalar Quantization (`SQ8`)** built from first principles.")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
n_samples = col_cfg1.slider("Synthetic Vector Count (N)", min_value=100, max_value=2000, value=500, step=100)
dim = col_cfg2.selectbox("Embedding Dimension (D)", options=[16, 32, 64, 128], index=1)
ef_search = col_cfg3.slider("HNSW Beam Width (ef_search)", min_value=8, max_value=128, value=32)

if st.button("Run HNSW + SQ8 Benchmark", type="primary"):
    rng = np.random.default_rng(42)
    raw_vectors = rng.standard_normal((n_samples, dim)).astype(np.float32)

    engine = MicroVectorDB(dim=dim, use_quantization=True)
    records = [
        VectorRecord(id=f"vec_{i}", vector=raw_vectors[i].tolist(), metadata={"cluster": int(i % 5)})
        for i in range(n_samples)
    ]

    t0 = time.perf_counter()
    engine.insert_batch(records)
    build_ms = (time.perf_counter() - t0) * 1000.0

    # Run query benchmark
    query_vec = raw_vectors[0] + rng.normal(0, 0.05, size=dim).astype(np.float32)
    t_q = time.perf_counter()
    res = engine.query(query_vec.tolist(), top_k=5, ef_search=ef_search)
    query_ms = (time.perf_counter() - t_q) * 1000.0

    stats = engine.get_stats()
    mse = engine.quantizer.compute_reconstruction_mse(raw_vectors)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Query Latency", f"{query_ms:.2f} ms")
    m2.metric("Graph Nodes Visited", f"{res.visited_nodes} / {n_samples}")
    m3.metric("Memory Compression", f"{stats.compression_ratio:.1f}x (SQ8)")
    m4.metric("Quantization MSE", f"{mse:.6f}")

    st.subheader("🔍 Top-K Nearest Neighbor Retrieval")
    st.json([item.model_dump() for item in res.results])
