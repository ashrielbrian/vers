use rand::Rng;
use serde::{Deserialize, Serialize};

use super::base::Index;
use super::models::{AdjacencyItem, DistanceCandidatePair, DistanceMaxCandidatePair};
use crate::Vector;
use std::cmp::min;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
extern crate itertools;

struct Node<'a, const N: usize> {
    id: usize,
    // vec isn't actually necessary since id_to_vec is usually passed around too
    vec: &'a Vector<N>,
}

#[derive(Serialize, Deserialize)]
pub struct HNSWIndex<const N: usize> {
    ef_construction: usize,
    ef_search: usize,
    num_neighbours: usize,
    layers: Vec<HNSWLayer<N>>,
    layer_multiplier: f32,
    id_to_vec: HashMap<usize, Vector<N>>,
}

#[derive(Serialize, Deserialize)]
pub struct HNSWLayer<const N: usize> {
    adjacency_list: HashMap<usize, AdjacencyItem>,
}

fn get_top_k_smallest_nodes(
    max_heap: BinaryHeap<DistanceMaxCandidatePair>,
    m: usize,
) -> Vec<&usize> {
    // returns nodes in descending order, where the first node has the largest distance
    let mut nodes_vec = itertools::unfold(max_heap, |heap| heap.pop())
        .map(|node| node.candidate_id)
        .collect::<Vec<_>>();

    nodes_vec.reverse();
    nodes_vec.truncate(m);
    nodes_vec
}

impl<const N: usize> HNSWLayer<N> {
    fn _add_edge(&mut self, u: &usize, v: &DistanceCandidatePair) {
        if let Some(adjacency_item) = self.adjacency_list.get_mut(u) {
            adjacency_item.insert(v);
        } else {
            let mut adj_item = AdjacencyItem::new();
            adj_item.insert(v);

            self.adjacency_list.insert(u.clone(), adj_item);
        }
    }

    fn _add_solitary_node(&mut self, u: &usize) {
        self.adjacency_list.insert(u.clone(), AdjacencyItem::new());
    }

    fn add_edge(&mut self, u: &usize, v: Option<&DistanceCandidatePair>) {
        match v {
            Some(other) => {
                // add both directions since undirected
                self._add_edge(u, other);
                self._add_edge(
                    &other.candidate_id,
                    &DistanceCandidatePair {
                        candidate_id: u.clone(),
                        distance: other.distance,
                    },
                );
            }
            None => {
                // rare case for the top layer entrypoint node
                self._add_solitary_node(u);
            }
        }
    }

    fn trim_edges(&mut self, node_id: &usize, max_num_edges: usize) {
        if let Some(adj_item) = self.adjacency_list.get_mut(node_id) {
            adj_item.trim(max_num_edges);
        }
    }

    fn add_node(
        &mut self,
        candidates: Vec<DistanceCandidatePair>,
        target_node: &usize,
        // number of neighbours per node, also, node degrees
        m: usize,
    ) {
        match candidates.len() {
            0 => {
                // adds an isolated node. this is an edge case, e.g. when there are no nodes in the layer initially
                // and it is the first node to be added to the layer.
                self.add_edge(target_node, None);
            }
            _ => {
                // gets only the top-m candidates to be connected as neighbours
                let selected_neighbours = if candidates.len() > m {
                    &candidates[candidates.len() - m..]
                } else {
                    &candidates
                };

                // need to update the node relationships of all the neighbours
                // add the node to this layer, and adjacency list
                // layer.add_node(target_node);
                for new_neighbour in selected_neighbours.iter() {
                    self.add_edge(target_node, Some(new_neighbour));
                }

                // check and reduce the num of edges if exceeds m, for all the neighbours.
                // this is done because these neighbours have their own neighbours, and adding an edge between
                // the new node and them could have caused these neighbours to exceed m.
                for neighbour in selected_neighbours {
                    self.trim_edges(&neighbour.candidate_id, m);
                }
            }
        }
    }

    fn search(
        &self,
        entrypoint: &Node<N>,
        query_vector: &Vector<N>,
        ef_construction: usize,
        id_to_vec: &HashMap<usize, Vector<N>>,
    ) -> Vec<DistanceCandidatePair> {
        // returns a list of candidates closest (of length ef_construction) to the query vector for a given layer
        let mut queue: VecDeque<&usize> = VecDeque::new();
        let mut candidates_heap: BinaryHeap<DistanceMaxCandidatePair> = BinaryHeap::new();
        let mut visited: HashSet<&usize> = HashSet::new();

        queue.push_back(&entrypoint.id);
        candidates_heap.push(DistanceMaxCandidatePair {
            candidate_id: &entrypoint.id,
            distance: entrypoint.vec.squared_euclidean_simd(query_vector),
        });

        while let Some(node) = queue.pop_front() {
            visited.insert(node);

            // get all neighbours in the current node, then process these neighbourhood nodes
            if let Some(adj_item) = self.adjacency_list.get(node) {
                for neighbour_id in adj_item.neighbours.iter() {
                    if visited.contains(neighbour_id) {
                        continue;
                    }

                    let neighbour_dist =
                        query_vector.squared_euclidean_simd(&id_to_vec.get(neighbour_id).unwrap());

                    // if the current neighbour distance is smaller than the candidate with the largest distance, replace
                    // this candidate with the current neighbour
                    if candidates_heap.len() < ef_construction
                        || (candidates_heap.len() > 0
                            && neighbour_dist < candidates_heap.peek().unwrap().distance)
                    {
                        queue.push_back(neighbour_id);
                        candidates_heap.push(DistanceMaxCandidatePair {
                            candidate_id: neighbour_id,
                            distance: neighbour_dist,
                        });
                    }
                }
            }
        }

        // returns the candidates in descending order, i.e. the first ele has the largest distance to the query vector
        itertools::unfold(candidates_heap, |heap| heap.pop())
            .map(|candidate| DistanceCandidatePair {
                candidate_id: candidate.candidate_id.clone(),
                distance: candidate.distance,
            })
            .collect()
    }
}

impl<const N: usize> HNSWIndex<N> {
    pub fn new(
        ef_construction: usize,
        ef_search: usize,
        num_layers: usize,
        num_neighbours: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| HNSWLayer {
                adjacency_list: HashMap::new(),
            })
            .collect();

        let id_to_vec: HashMap<usize, Vector<N>> = HashMap::new();
        let layer_multiplier = 1.0 / (num_neighbours as f32).ln();

        HNSWIndex {
            ef_construction,
            ef_search,
            num_neighbours,
            layers,
            layer_multiplier,
            id_to_vec,
        }
    }

    fn get_insertion_layer(layer_multiplier: f32, num_layers: usize) -> usize {
        let random_val: f32 = rand::thread_rng().gen();
        let l = -(random_val.ln() * (layer_multiplier as f32)) as usize;
        min(l, num_layers)

        // num_layers = 6
        // min(l, num_layers), l = 0 => 0 (case when insertion layer is at the top)
        // min(l, num_layers), l >= 6 => 6 (case when insertion layer is at the bottommost - most common case)
        // min(l, num_layers), 0 < l < 6 => l (e.g. 1, 2, 3, 4, 5. insert at l and below)
    }

    fn _add_node(&mut self, embedding: &Vector<N>, embedding_id: usize) -> Result<(), String> {
        let max_layers = self.layers.len();

        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers.last().unwrap();

        // get the layer to insert this node at, and all layers below
        let insertion_layer = Self::get_insertion_layer(self.layer_multiplier, max_layers);

        if let Some((entrypoint_node, _)) = top_layer.adjacency_list.iter().next() {
            // 1. perform search from layers top_layer to insertion_layer + 1
            let entrypoint_vec = self.id_to_vec.get(entrypoint_node).unwrap();
            let mut entrypoint = Node {
                id: entrypoint_node.clone(),
                vec: entrypoint_vec,
            };

            for layer_idx in (insertion_layer + 1..self.layers.len() - 1).rev() {
                let curr_layer = &self.layers[layer_idx];
                let candidates = curr_layer.search(
                    &entrypoint,
                    embedding,
                    self.ef_construction,
                    &self.id_to_vec,
                );

                entrypoint = Self::_get_best_candidate(candidates, &self.id_to_vec).unwrap();
            }

            // 2. insert node from layers insertion_layer to 0
            for layer_idx in (0..=insertion_layer).rev() {
                // 0. use the entrypoint from the last section
                // 1. run search on the layer like usual and get candidates
                let curr_layer = &mut self.layers[layer_idx];
                let candidates = curr_layer.search(
                    &entrypoint,
                    embedding,
                    self.ef_construction,
                    &self.id_to_vec,
                );

                // 2. get top-m candidates
                // 3. add node + top-m  as new neighbours
                let num_neighbours = if layer_idx == 0 {
                    2 * self.num_neighbours
                } else {
                    self.num_neighbours
                };

                curr_layer.add_node(candidates.clone(), &embedding_id, num_neighbours);

                // 4. get new entrypoint as the top-1 neighbour, for the next layer
                entrypoint = Self::_get_best_candidate(candidates, &self.id_to_vec).unwrap();
            }
        } else {
            // add the node to the topmost layer and all nodes below it. since this is the first node, there are no
            // candidates and we add only itself.
            for layer in &mut self.layers {
                layer.add_node(Vec::with_capacity(0), &embedding_id, self.num_neighbours);
            }
        }

        Ok(())
    }

    pub fn create(&mut self, vectors: &Vec<Vector<N>>) {
        vectors.iter().enumerate().for_each(|(idx, vec)| {
            self.id_to_vec.insert(idx, vec.clone());
            self._add_node(vec, idx);
        });
    }

    pub fn build_index(
        num_layers: usize,
        ef_construction: usize,
        ef_search: usize,
        num_neighbours: usize,
        vectors: &Vec<Vector<N>>,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| HNSWLayer {
                adjacency_list: HashMap::new(),
            })
            .collect();

        let mut id_to_vec: HashMap<usize, Vector<N>> = HashMap::new();
        vectors.iter().enumerate().for_each(|(idx, vec)| {
            id_to_vec.insert(idx, vec.clone());
        });

        let layer_multiplier = 1.0 / (num_neighbours as f32).ln();

        let mut hnsw = HNSWIndex {
            ef_search,
            ef_construction,
            layers,
            layer_multiplier,
            num_neighbours,
            id_to_vec,
        };

        println!("Done instantiating the HNSW index. Proceeding to add nodes...");

        // _add_node separately because we needed HNSWIndex instantiated beforehand
        vectors.iter().enumerate().for_each(|(idx, vec)| {
            println!("Adding node {}", idx);
            hnsw._add_node(vec, idx);
        });

        hnsw
    }

    pub fn add_node_to_layer() {
        // use the probability functions to get the layer to place this node at
        // run search from top layer all the way to the layer it was assigned.
        // - need to add this node on this layer and all layers below it
        // given an entrypoint, find all candidates (use _search_layer). once the candidates have been found,
        // link the node_vector to the top M candidates, where M is the number of neighbours/edges per node.
        // - add the node to the layer
        // - update the adjacency list
    }

    pub fn get_num_nodes_in_layers(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|layer| layer.adjacency_list.len())
            .collect()
    }

    fn _get_best_candidate<'a>(
        candidates: Vec<DistanceCandidatePair>,
        id_to_vec: &'a HashMap<usize, Vector<N>>,
    ) -> Option<Node<'a, N>> {
        if let Some(best_candidate) = candidates.last().cloned() {
            let best_candidate_id = best_candidate.candidate_id;
            return Option::Some(Node {
                vec: id_to_vec.get(&best_candidate_id).unwrap(),
                id: best_candidate_id,
            });
        }
        None
    }
}

impl<const N: usize> Index<N> for HNSWIndex<N> {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize) {
        match self._add_node(&embedding, vec_id) {
            Ok(_) => println!("Successfully added a node!"),
            Err(e) => println!("{}", e),
        }
    }

    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let ef_search = self.ef_search;

        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers.last().unwrap();

        if let Some((entrypoint_node, _)) = top_layer.adjacency_list.iter().next() {
            // start from the second last layer and work downwards to layer 0
            let entrypoint_vec = self.id_to_vec.get(entrypoint_node).unwrap();
            let mut entrypoint = Node {
                id: entrypoint_node.clone(),
                vec: entrypoint_vec,
            };

            let mut final_candidates: Vec<DistanceCandidatePair> = vec![];

            for layer_idx in (0..self.layers.len() - 1).rev() {
                let curr_layer = &self.layers[layer_idx];

                let candidates = curr_layer.search(&entrypoint, &query, ef_search, &self.id_to_vec);

                if layer_idx != 0 {
                    entrypoint = Self::_get_best_candidate(candidates, &self.id_to_vec).unwrap();
                } else {
                    final_candidates = candidates;
                }
            }

            final_candidates
                .into_iter()
                .rev()
                .take(top_k)
                .map(|candidate| (candidate.candidate_id, candidate.distance))
                .collect::<Vec<(usize, f32)>>()
        } else {
            println!("Top layer does not have an entrypoint.");
            return vec![];
        }
    }
}
