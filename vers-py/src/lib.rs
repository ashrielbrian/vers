use paste::paste;
use pyo3::prelude::*;
use std::time::Instant;
use std::{cell::RefCell, collections::HashMap};

use vers::{utils, HNSWIndex, Index, Vector};

pub mod vecs;

#[pyclass]
#[derive(Clone)]
struct Vector_F32_N300(Vector<300>);

#[pymethods]
impl Vector_F32_N300 {
    fn get_inner(&self) -> [f32; 300] {
        self.0 .0
    }
}

trait HNSWWrapper: Send + Sync {
    fn build_index(&mut self, vectors: &Vec<Vec<f32>>);
    fn get_dimensions(&self) -> usize;
}

macro_rules! create_hnsw_interfaces {
    ($($size:expr),*) => {
        $(
            paste! {
                #[pyclass]
                pub struct [<HNSWIndex$size>] {
                    inner: HNSWIndex<$size>,
                }


                impl HNSWWrapper for [<HNSWIndex$size>] {
                    fn build_index(&mut self, vectors: &Vec<Vec<f32>>) {
                        // validate dims
                        if vectors.is_empty() || vectors[0].len() != $size {
                            panic!("Vectors must be of dimension {}", $size);
                        }

                        // convert Vec<Vec<f32>> to Vec<Vector<$size>>
                        let converted_vectors: Vec<Vector<$size>> = vectors
                            .iter()
                            .map(|v| {
                                let mut arr = [0.0; $size];
                                arr.copy_from_slice(&v[..std::cmp::min(v.len(), $size)]);
                                Vector(arr)
                            })
                            .collect();

                        self.inner.create(&converted_vectors);
                    }

                    fn get_dimensions(&self) -> usize {
                        $size
                    }
                }
            }
        )*
    };
}

create_hnsw_interfaces!(300, 512, 1024, 1536);

#[pyclass]
struct HNSW {
    index: Box<dyn HNSWWrapper>,
}

#[pymethods]
impl HNSW {
    #[new]
    fn new(
        n_dims: usize,
        ef_construction: usize,
        ef_search: usize,
        num_layers: usize,
        num_neighbours: usize,
    ) -> Self {
        let index: Box<dyn HNSWWrapper> = match n_dims {
            300 => Box::new(HNSWIndex300 {
                inner: HNSWIndex::<300>::new(
                    ef_construction,
                    ef_search,
                    num_layers,
                    num_neighbours,
                ),
            }),
            512 => Box::new(HNSWIndex512 {
                inner: HNSWIndex::<512>::new(
                    ef_construction,
                    ef_search,
                    num_layers,
                    num_neighbours,
                ),
            }),
            1024 => Box::new(HNSWIndex1024 {
                inner: HNSWIndex::<1024>::new(
                    ef_construction,
                    ef_search,
                    num_layers,
                    num_neighbours,
                ),
            }),
            1536 => Box::new(HNSWIndex1536 {
                inner: HNSWIndex::<1536>::new(
                    ef_construction,
                    ef_search,
                    num_layers,
                    num_neighbours,
                ),
            }),

            _ => {
                panic!("This number of dimensions is not supported!")
            }
        };

        Self { index }
    }
}

#[pyfunction]
fn load_wiki_vector(
    path: &str,
) -> PyResult<(
    Vec<Vector_F32_N300>,
    HashMap<String, usize>,
    HashMap<usize, String>,
    Vec<(String, [f32; 300])>,
)> {
    let (all_vecs, word_to_idx, idx_to_word, test_embs) = utils::load_wiki_vector(path);
    Ok((
        all_vecs.into_iter().map(Vector_F32_N300).collect(),
        word_to_idx,
        idx_to_word,
        test_embs,
    ))
}

#[pyfunction]
fn test_wiki_ivfflat(
    vectors: Vec<Vector_F32_N300>,
    word_to_idx: HashMap<String, usize>,
    idx_to_word: HashMap<usize, String>,
    num_clusters: usize,
    num_attempts: usize,
    max_iterations: usize,
    test_embs: Vec<(String, [f32; 300])>,
) {
    let vectors: Vec<Vector<300>> = vectors.iter().map(|v| v.0).collect();

    let obj = RefCell::new(idx_to_word);
    let mut idx_to_word_borrowed = obj.borrow_mut();

    let start = Instant::now();
    utils::test_ivfflat(
        &vectors,
        &word_to_idx,
        &mut idx_to_word_borrowed,
        num_clusters,
        num_attempts,
        max_iterations,
        &test_embs,
    );
    let duration = start.elapsed();
    println!("Time taken to test IVFFlat: {:?}", duration);
}

#[pyfunction]
fn get_sum(v1: [f32; 300], v2: [f32; 300]) -> PyResult<Vector_F32_N300> {
    let v1 = Vector_F32_N300(Vector(v1));
    let v2 = Vector_F32_N300(Vector(v2));
    Ok(Vector_F32_N300(v1.0.add(&v2.0)))
}

/// A Python module implemented in Rust.
#[pymodule]
fn vers_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_sum, m)?)?;
    m.add_function(wrap_pyfunction!(load_wiki_vector, m)?)?;
    m.add_function(wrap_pyfunction!(test_wiki_ivfflat, m)?)?;
    Ok(())
}
