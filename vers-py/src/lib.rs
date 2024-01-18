use std::collections::HashMap;

use pyo3::prelude::*;

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
    Ok(())
}
