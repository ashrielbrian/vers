# vers

Lightweight, simple, single instance, local in-memory vector database written in Rust.

Currently supports the following indexing strategies:

1. IVFFlat (k-means for partitioning)
2. Locality-sensitive hashing (LSH) heavily inspired by [fennel.ai's blog post](https://fennel.ai/blog/vector-search-in-200-lines-of-rust/).

### Getting Started

Like any sensible package, the API aims to be dead simple.

0. Import, obviously:

```rust
    use vers::indexes::base::{Index, Vector};
    use vers::indexes::ivfflat::IVFFlatIndex;
```

1. Build an index:

```rust
    let mut index = IVFFlatIndex::build_index(
        num_clusters,
        num_attempts,
        max_iterations,
        &vectors
    );
```

2. Add an embedding vector into the index:

```rust
    index.add(Vector(*emb), emb_unique_id);
```

3. Persist the index to disk:

```rust
    let _ = index.save_index("wiki.index");
```

4. Load the index from disk:

```rust
    let index = match IVFFlatIndex::load_index("wiki.index") {
        Ok(index) => index,
        Err(e) => panic!("Failed to load index! {}", e),
    };
```

5. And of course, actually search the index:

```rust
    let results = index.search_approximate(
        embs.get("king"),   // query vector
        10                  // top_k
    ); // kings, queen, monarch, ...
```

**That said, the API is unstable and subject to change. In particular, I really dislike having to pass in the unique vector ID into `search_approximate`.**

### Coming soon

1. Python bindings
2. Performance improvements (building IVFFlat index is slow, vectorization)
3. Benchmarks (comparisons with popular ANN search indexes, e.g. faiss, and exhaustive searches)

Contributions are welcomed.
