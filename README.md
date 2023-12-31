# vers

An in-memory vector database written in Rust.

Currently supports the following indexing strategies:

1. IVFFlat (k-means for partitioning)
2. Locality-sensitive hashing (LSH) heavily inspired by [fennel.ai's blog post](https://fennel.ai/blog/vector-search-in-200-lines-of-rust/).

#### DISCLAIMER: API is unstable and subject to change.

### Coming soon

1. Python bindings
2. Performance improvements (building IVFFlat index among others, vectorization)
3. Benchmarks (comparisons with popular ANN search indexes, e.g. faiss, and exhaustive searches)

Contributions are welcomed.
