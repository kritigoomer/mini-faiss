# mini-faiss

A lightweight, custom-build vector database search engine.

### What does it do? 
*Stores High-Dimensional Vectors*: Modern AI (LLMs, RAG systems, and image search) converts text or images into dense arrays of numbers called *embeddings* (e.g., 768 floating-point numbers long). This engine stores them in memory.

*Performs Similarity Search (Nearest Neighbor Search)*: When you pass a query vector, it scans all stored vectors, computes the mathematical distance (Euclidean/L2 distance) between your query and every single stored vector, and returns the top-$k$ closest matches.
* *SIMD Acceleration (Single Instruction, Multiple Data)*: Normally, a computer CPU computes distance sequentially: $(a[0] - b[0])^2$, then $(a[1] - b[1])^2$, one-by-one.
* SIMD instructions (like AVX2 or AVX-512 on Intel/AMD chips, or NEON on ARM chips) allow the CPU to perform math on *multiple floats simultaneously* using wide hardware registers (e.g., processing 8 or 16 floats in a single clock cycle). This speeds up vector similarity search by several multiples, which is crucial when scaling to millions of embeddings.
