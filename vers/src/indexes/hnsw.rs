use serde::{Deserialize, Serialize};

use super::base::Index;
use crate::Vector;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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

#[derive(PartialEq)]
struct DistanceMaxCandidatePair {
    candidate_id: String,
    distance: f32,
}

impl Eq for DistanceMaxCandidatePair {}

impl PartialOrd for DistanceMaxCandidatePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for DistanceMaxCandidatePair {
    fn cmp(&self, other: &DistanceMaxCandidatePair) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
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

    pub fn _search_layer(
        entrypoint: Node<N>,
        query_vector: Vector<N>,
        layer: &HNSWLayer<N>,
        ef_construction: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
    ) -> Vec<(String, f32)> {
        // returns a list of candidates closest to the query vector for a given layer
        let mut candidates: Vec<(String, f32)> = vec![];
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut candidates_heap: BinaryHeap<DistanceMaxCandidatePair> = BinaryHeap::new();

        queue.push_back(entrypoint.id);

        while let Some(node) = queue.pop_front() {
            if let Some(neighbour_ids) = layer.adjacency_list.get(&node) {
                for neighbour_id in neighbour_ids {
                    let neighbour_dist =
                        query_vector.squared_euclidean(&id_to_vec.get(neighbour_id).unwrap());

                    // if the current neighbour distance is smaller than the candidate with the largest distance, replace
                    // this candidate with the current neighbour
                    if candidates.len() < ef_construction
                        || (candidates_heap.len() > 0
                            && neighbour_dist < candidates_heap.peek().unwrap().distance)
                    {
                        queue.push_back(neighbour_id.clone());
                        candidates_heap.push(DistanceMaxCandidatePair {
                            candidate_id: neighbour_id.clone(),
                            distance: neighbour_dist,
                        });
                    }
                }
            }
        }

        candidates
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
