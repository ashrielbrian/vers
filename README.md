# vers

Lightweight, simple, single instance, local in-memory vector database written in Rust.

Currently supports the following indexing strategies:

1. IVFFlat (k-means for partitioning)
2. Locality-sensitive hashing (LSH) heavily inspired by [fennel.ai's blog post](https://fennel.ai/blog/vector-search-in-200-lines-of-rust/).
3. Hierachical Navigable Small Worlds (HNSW)

This repository is educational. It was meant to understand how ANN search algorithms work under the hood for modern machine learning and is not meant for production (as a cursory scan of the codebase can probably tell you).

_"What I cannot build, I do not understand."_ - Feynman

### Getting Started

Like any sensible package, the API aims to be dead simple.

0. Import, obviously:

```rust
    use vers::indexes::base::{Index, Vector};
    use vers::indexes::ivfflat::IVFFlatIndex;
    use vers::indexes::hnsw::HNSWIndex;
```

1. Build an index:

```rust
    let mut index = IVFFlatIndex::build_index(
        num_clusters,
        num_attempts,
        max_iterations,
        &vectors
    );

    // or hnsw
    let mut index = HNSWIndex::build_index(
        num_layers,
        ef_construction,
        ef_search,
        num_neighbours,
        vectors
    )
```

2. Add an embedding vector into the index:

```rust
    index.add(Vector(*emb).normalize(), emb_unique_id);
```

3. Persist the index to disk:

```rust
    let _ = index.save_index("wiki.index");
```

4. Load the index from disk:

```rust
    // or use HNSWIndex::load_index, ANNIndex::load_index
    let index = match IVFFlatIndex::load_index("wiki.index") {
        Ok(index) => index,
        Err(e) => panic!("Failed to load index! {}", e),
    };
```

5. And of course, actually search the index:

```rust
    let results = hnsw.search_approximate(
        embs.get("king"),   // query vector
        10                  // top_k
    ); // kings, queen, monarch, ...
```

As shown above, all the indexes share the same API, whether IVFFlat, HNSW or LSH.

### Python Bindings **(WIP!)**
`vers` now has a simple Python API using pyo3. To use vers with Python (>- 3.7),

```python
    import vers

    # load wiki sample embeddings of dims 300
    embeddings = vers.load_wiki()

    # instantiate the hnsw index with params
    hnsw = vers.HNSW(ef_construction=100, num_layers=8, ef_search=32, num_neighbours=8)

    # build the index
    hnsw.build_index(embeddings)

    # search for a query vector
    results = hnsw.search(embeddings.get("king"), top_k=10)
```

### Running adhoc benchmark tests

```
    cargo build --release
    samply record ./target/release/vers
```
