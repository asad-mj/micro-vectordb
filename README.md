# ⚡ micro-vectordb: HNSW & SQ8 Vector Database Engine

[![VectorDB CI Pipeline](https://github.com/asad-mj/micro-vectordb/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/asad-mj/micro-vectordb/actions/workflows/ci-cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Workbench-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)

A high-performance Approximate Nearest Neighbor (ANN) vector database built **from first principles** in Python & NumPy. Implements multi-layer **Hierarchical Navigable Small World (HNSW)** graph indexing and **8-Bit Asymmetric Scalar Quantization (`SQ8`)** for $4\times$ memory compression.

---

## 📐 Mathematical & Algorithmic Formulation

### 1. Multi-Layer HNSW Graph Construction
Each inserted vector $\mathbf{x}_i \in \mathbb{R}^D$ is assigned a maximum layer $l_i$ sampled from an exponentially decaying distribution:

$$l_i = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor, \quad m_L = \frac{1}{\ln(M)}$$

During query time, greedy search traverses upper layers ($L_{\max} \to 1$) with beam width $ef=1$ to locate a coarse entry point before executing bounded best-first beam search at layer $l=0$ with width $ef_{\text{search}}$, reducing time complexity from $\mathcal{O}(N \cdot D)$ to $\mathcal{O}(\log N \cdot D)$.

### 2. Asymmetric 8-Bit Scalar Quantization (`SQ8`)
Continuous `float32` components are mapped to discrete `uint8` codes $q_j \in [0, 255]$ per dimension $j$:

$$q_j = \text{clip}\left( \left\lfloor \frac{x_j - x_{\min, j}}{\Delta_j} \right\rceil, 0, 255 \right), \quad \Delta_j = \frac{x_{\max, j} - x_{\min, j}}{255}$$

---

## 📊 Empirical Benchmarks ($N = 10,000, D = 128$)

| Indexing Method | Recall@10 | P95 Query Latency | Memory Footprint | Visited Nodes (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Brute-Force Flat (`float32`)** | 100.0% | 14.80 ms | 5.12 MB | 100.0% |
| **HNSW (`M=16, ef=32, float32`)** | **98.4%** | **0.82 ms** | 5.12 MB | **4.2%** |
| **HNSW + SQ8 Quantization (`uint8`)** | **97.9%** | **0.85 ms** | **1.28 MB (4.0x)** | **4.2%** |

---

## 🚀 Quick Start

```bash
git clone [https://github.com/asad-mj/micro-vectordb.git](https://github.com/asad-mj/micro-vectordb.git)
cd micro-vectordb

pip install -r requirements.txt
python -m pytest tests/ -v

# Start REST API Server
uvicorn main:app --reload --port 8000

# Launch Interactive Benchmark Workbench
streamlit run app.py
