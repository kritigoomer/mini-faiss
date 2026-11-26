# Mini-FAISS

A lightweight, in-memory exact k-Nearest Neighbors (k-NN) vector search engine built in C++20.

## Overview

Modern AI pipelines (such as RAG, semantic search, and image retrieval) represent text, images, and audio as dense floating-point vectors (embeddings), commonly 768 dimensions long. `mini-faiss` stores vectors in memory and performs exact brute-force squared Euclidean ($L_2^2$) distance search to return the `top-k` closest matches for a query vector.

## Key Design Choices

* **Contiguous 64-byte aligned storage:** Stores all $N \times D$ `float32` values in a single flat buffer using a 64-byte aligned allocator instead of `std::vector<std::vector<float>>`, reducing heap fragmentation, pointer chasing, and cache/TLB misses.
* **Cross-platform SIMD kernels:** Computes $L_2^2$ distances using AVX2 + FMA (8 `float32` lanes) on x86_64 and ARM NEON (4 `float32` lanes) on ARM, with a scalar tail loop for remaining elements.
* **Bounded max-heap top-$K$ selection:** Streams `(distance_sq, id)` candidates through a size-$K$ max-heap during the scan, achieving $O(N \log K)$ selection time and $O(K)$ extra memory instead of sorting all $N$ distances.

## Scope & Non-Goals

* **In scope:** Single-node, in-memory exact k-NN search and latency benchmarking.
* **Out of scope (for now):** Distributed sharding, disk persistence, deletes/updates, and approximate indexes (HNSW, IVF, quantization).

## Build and Run

Requires CMake 3.20+ and a C++20 compiler (`g++`, `clang++`, or MSVC).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mini_faiss_bench
