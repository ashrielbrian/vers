use serde::{Deserialize, Serialize};

use super::base::Index;
use crate::Vector;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Serialize, Deserialize)]
pub struct HNSWIndex<const N: usize> {
    ef_construction: usize,
    ef_search: usize,
    layers: Vec<HNSWLayer<N>>,
}

#[derive(Serialize, Deserialize)]
pub struct HNSWLayer<const N: usize> {
    adjacency_list: HashMap<String, Vec<String>>,
    nodes: HashMap<String, Vec<Vector<N>>>,
}

impl<const N: usize> HNSWIndex<N> {
    pub fn new(num_layers: usize, ef_construction: usize, ef_search: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| HNSWLayer {
                adjacency_list: HashMap::new(),
                nodes: HashMap::new(),
            })
            .collect();

        HNSWIndex {
            ef_search,
            ef_construction,
            layers,
        }
    }

    fn _add_node(&self) {}

    pub fn build_index(&self) {
        todo!()
    }

    pub fn _search_layer(entrypoint: Node<N>, query_vector: Vector<N>, layer: &HNSWLayer<N>) {
        // returns a list of candidates closest to the query vector for a given layer
        let mut candidates: Vec<(String, f32)> = vec![];
        let mut queue: VecDeque<String> = VecDeque::new();

        queue.push_back(entrypoint.id);

        while let Some(node) = queue.pop_front() {}
    }
}

struct Node<'a, const N: usize> {
    id: String,
    vec: &'a Vector<N>,
}

impl<const N: usize> Index<N> for HNSWIndex<N> {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize) {}

    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers[0];
        if let Some((&entrypoint_node, entrypoint_vec)) = top_layer.nodes.iter().next() {
        } else {
            println!("Top layer does not have an entrypoint.");
            return vec![];
        }
    }
}
