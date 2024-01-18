use pyo3::prelude::*;
use std::time::Instant;
use std::{cell::RefCell, collections::HashMap};

use vers::{utils, Index, Vector};

#[pyclass]
#[derive(Clone)]
struct Vector_F32_N300(Vector<300>);

#[pymethods]
impl Vector_F32_N300 {
    fn get_inner(&self) -> [f32; 300] {
        self.0 .0
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
