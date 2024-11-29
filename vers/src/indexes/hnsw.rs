use serde::{Deserialize, Serialize};

use super::base::Index;
use crate::Vector;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

extern crate itertools;

#[derive(Serialize, Deserialize)]
pub struct HNSWIndex<const N: usize> {
    ef_construction: usize,
    ef_search: usize,
    layers: Vec<HNSWLayer<N>>,
}

#[derive(Serialize, Deserialize)]
pub struct HNSWLayer<const N: usize> {
    adjacency_list: HashMap<String, HashSet<String>>,
    nodes: HashMap<String, Vec<Vector<N>>>,
}

impl<const N: usize> HNSWLayer<N> {
    fn _add_edge(&mut self, u: &String, v: &String) {
        // if self.adjacency_list.contains_key(&u) {}

        if let Some(adjacency_set) = self.adjacency_list.get_mut(u) {
            adjacency_set.insert(v.clone());
        } else {
            self.adjacency_list.insert(u.clone(), HashSet::new());
            self.adjacency_list.get_mut(u).unwrap().insert(v.clone());
        }
    }
    pub fn add_edge(&mut self, u: &String, v: &String) {
        self._add_edge(u, v);
        self._add_edge(v, u);
    }
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

    pub fn build_index(&self) {
        todo!()
    }

    fn _add_node_to_layer(
        entrypoint: Node<N>,
        target_node: &Node<N>,
        layer: &HNSWLayer<N>,
        ef_construction: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
        // number of neighbours per node, also, node degrees
        m: usize,
    ) {
        // use the probability functions to get the layer to place this node at
        // run search from top layer all the way to the layer it was assigned.
        // - need to add this node on this layer and all layers below it
        // given an entrypoint, find all candidates (use _search_layer). once the candidates have been found,
        // link the node_vector to the top M candidates, where M is the number of neighbours/edges per node.
        // - add the node to the layer
        // - update the adjacency list
        let candidates = Self::_search_layer(
            entrypoint,
            target_node.vec,
            layer,
            ef_construction,
            id_to_vec,
        );

        let selected_neighbours = if candidates.len() > m {
            &candidates[candidates.len() - m..]
        } else {
            &candidates
        };

        // need to update the node relationships of all the neighbours
        // add the node to this layer, and adjacency list
    }
    pub fn _search_layer(
        entrypoint: Node<N>,
        query_vector: &Vector<N>,
        layer: &HNSWLayer<N>,
        ef_construction: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
    ) -> Vec<String> {
        // returns a list of candidates closest to the query vector for a given layer
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
                    if candidates_heap.len() < ef_construction
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

        // returns the candidates in descending order, i.e. the first ele has the largest distance to the query vector
        itertools::unfold(candidates_heap, |heap| heap.pop())
            .map(|candidate| candidate.candidate_id)
            .collect()
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
