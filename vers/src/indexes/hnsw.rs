use rand::Rng;
use serde::{Deserialize, Serialize};

use super::base::Index;
use crate::Vector;
use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

extern crate itertools;

#[derive(Serialize, Deserialize)]
pub struct HNSWIndex<const N: usize> {
    ef_construction: usize,
    ef_search: usize,
    layers: Vec<HNSWLayer<N>>,
    id_to_vec: HashMap<String, Vector<N>>,
}

#[derive(Serialize, Deserialize)]
pub struct HNSWLayer<const N: usize> {
    adjacency_list: HashMap<String, HashSet<String>>,
}

impl<const N: usize> HNSWLayer<N> {
    fn _add_edge(&mut self, u: &String, v: &String) {
        if let Some(adjacency_set) = self.adjacency_list.get_mut(u) {
            adjacency_set.insert(v.clone());
        } else {
            self.adjacency_list.insert(u.clone(), HashSet::new());
            self.adjacency_list.get_mut(u).unwrap().insert(v.clone());
        }
    }

    fn _add_solitary_node(&mut self, u: &String) {
        self.adjacency_list.insert(u.clone(), HashSet::new());
    }
    pub fn add_edge(&mut self, u: &String, v: Option<&String>) {
        match v {
            Some(other) => {
                self._add_edge(u, other);
                self._add_edge(other, u);
            }
            None => {
                // rare case for the top layer entrypoint node
                self._add_solitary_node(u);
            }
        }
    }

    // pub fn add_node(&mut self, node: &Node<N>) {
    //     self.nodes.insert(node.id.clone(), node.vec.clone());
    // }
    //
    fn _add_node(
        &mut self,
        candidates: Vec<String>,
        target_node: &Node<N>,
        id_to_vec: &HashMap<String, Vector<N>>,
        // number of neighbours per node, also, node degrees
        m: usize,
    ) {
        let selected_neighbours = if candidates.len() > m {
            &candidates[candidates.len() - m..]
        } else {
            &candidates
        };

        // need to update the node relationships of all the neighbours
        // add the node to this layer, and adjacency list
        // layer.add_node(target_node);
        for new_neighbour in selected_neighbours.iter() {
            self.add_edge(&target_node.id, Some(new_neighbour));
        }

        // check and reduce the num of edges if exceeds m, for all the neighbours.
        // this is done because these neighbours have their own neighbours, and adding an edge between
        // the new node and them could have caused these neighbours to exceed m.
        for neighbour in selected_neighbours {
            self.trim_edges(neighbour, m, id_to_vec);
        }
    }

    pub fn trim_edges(
        &mut self,
        node_id: &String,
        max_num_edges: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
    ) {
        if let Some(neighbours) = self.adjacency_list.get(node_id) {
            if neighbours.len() > max_num_edges {
                let node_vec = id_to_vec.get(node_id).unwrap();
                let mut max_heap = BinaryHeap::new();

                // get the distances between the node and all its neighbours. then, remove edges
                // with the largest distances and update the adjacency list.
                // TODO: potentially can make this faster by storing the distance in the adjacency list,
                // so that instead of a hashset of strings, it would be a hashmap where the value is its distance.
                // trade-off between space and speed.
                for neighbour in neighbours.iter() {
                    let neighbour_vec = id_to_vec.get(neighbour).unwrap();
                    let dist = node_vec.squared_euclidean(neighbour_vec);
                    max_heap.push(DistanceMaxCandidatePair {
                        candidate_id: neighbour,
                        distance: dist,
                    });
                }

                let new_neighbours = get_top_k_smallest_nodes(max_heap, max_num_edges);
                self.adjacency_list.insert(
                    node_id.clone(),
                    HashSet::from_iter(new_neighbours.into_iter().cloned()),
                );
            }
        }
    }
}

fn get_top_k_smallest_nodes(
    max_heap: BinaryHeap<DistanceMaxCandidatePair>,
    m: usize,
) -> Vec<&String> {
    // returns nodes in descending order, where the first node has the largest distance
    let mut nodes_vec = itertools::unfold(max_heap, |heap| heap.pop())
        .map(|node| node.candidate_id)
        .collect::<Vec<_>>();

    nodes_vec.reverse();
    nodes_vec.truncate(m);
    nodes_vec
}

#[derive(PartialEq)]
struct DistanceMaxCandidatePair<'a> {
    candidate_id: &'a String,
    distance: f32,
}

impl Eq for DistanceMaxCandidatePair<'_> {}

impl<'a> PartialOrd for DistanceMaxCandidatePair<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<'a> Ord for DistanceMaxCandidatePair<'a> {
    fn cmp(&self, other: &DistanceMaxCandidatePair) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<const N: usize> HNSWIndex<N> {
    fn get_insertion_layer(layer_multiplier: usize, max_layers: usize) -> usize {
        let random_val: f32 = rand::thread_rng().gen();
        let l = -(random_val.ln() * (layer_multiplier as f32)) as usize;
        min(l, max_layers)
    }

    pub fn new(num_layers: usize, ef_construction: usize, ef_search: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| HNSWLayer {
                adjacency_list: HashMap::new(),
                // nodes: HashMap::new(),
            })
            .collect();

        HNSWIndex {
            ef_search,
            ef_construction,
            layers,
            id_to_vec: HashMap::new(),
        }
    }

    pub fn build_index(&self) {
        todo!()
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

    pub fn _search_layer<'a>(
        entrypoint: OwnedNode<N>,
        query_vector: &Vector<N>,
        layer: &'a HNSWLayer<N>,
        ef_construction: usize,
        id_to_vec: &'a HashMap<String, Vector<N>>,
    ) -> Vec<String> {
        // returns a list of candidates closest (of length ef_construction) to the query vector for a given layer
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut candidates_heap: BinaryHeap<DistanceMaxCandidatePair<'a>> = BinaryHeap::new();

        queue.push_back(entrypoint.id.clone());

        while let Some(node) = queue.pop_front() {
            // get all neighbours in the current node, then process these neighbourhood nodes
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
                            candidate_id: neighbour_id,
                            distance: neighbour_dist,
                        });
                    }
                }
            }
        }

        // returns the candidates in descending order, i.e. the first ele has the largest distance to the query vector
        itertools::unfold(candidates_heap, |heap| heap.pop())
            .map(|candidate| candidate.candidate_id.clone())
            .collect()
    }

    fn _get_best_candidate<'a>(
        candidates: Vec<String>,
        id_to_vec: &'a HashMap<String, Vector<N>>,
    ) -> Option<OwnedNode<'a, N>> {
        if let Some(best_candidate_id) = candidates.last().cloned() {
            return Option::Some(OwnedNode {
                vec: id_to_vec.get(&best_candidate_id).unwrap(),
                id: best_candidate_id,
            });
        }
        None
    }
}

struct Node<'a, 'b, const N: usize> {
    id: &'a String,
    vec: &'b Vector<N>,
}

struct OwnedNode<'a, const N: usize> {
    id: String,
    vec: &'a Vector<N>,
}

impl<const N: usize> Index<N> for HNSWIndex<N> {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize) {
        let ef_construction = 200;
        let layer_multiplier = 5;
        let max_layers = 5;

        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers.last().unwrap();

        // get the layer to insert this node at, and all layers below
        let insertion_layer = Self::get_insertion_layer(layer_multiplier, max_layers);

        if let Some((entrypoint_node, _)) = top_layer.adjacency_list.iter().next() {
            // perform search from layers top_layer to insertion_layer + 1
            let entrypoint_vec = self.id_to_vec.get(entrypoint_node).unwrap();
            let mut entrypoint = OwnedNode {
                id: entrypoint_node.clone(),
                vec: entrypoint_vec,
            };
            for layer_idx in (insertion_layer + 1..self.layers.len() - 1).rev() {
                let candidates = Self::_search_layer(
                    entrypoint,
                    &embedding,
                    &self.layers[layer_idx],
                    ef_construction,
                    &self.id_to_vec,
                );

                entrypoint = Self::_get_best_candidate(candidates, &self.id_to_vec).unwrap();
            }
            // insert node from layers insertion_layer to 0
            for layer_idx in (0..=insertion_layer).rev() {
                // 0. use the entrypoint from the last section
                // 1. run search on the layer like usual and get candidates
                // 2. get top-m candidates
                // 3. add node + top-m  as new neighbours
                // 4. get new entrypoint as the top-1 neighbour, for the next layer
            }
        } else {
            // add the node to the topmost layer and all nodes below it
            // for layer in layers {
            //      layer.add_node(node_id, neighbour_nodes: vec![])
            // }
        }
    }

    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let ef_search = 100;

        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers.last().unwrap();

        if let Some((entrypoint_node, _)) = top_layer.adjacency_list.iter().next() {
            // start from the second last layer and work downwards to layer 0
            let entrypoint_vec = self.id_to_vec.get(entrypoint_node).unwrap();
            for layer_idx in (0..self.layers.len() - 1).rev() {
                let curr_layer = &self.layers[layer_idx];

                let candidates = Self::_search_layer(
                    Node {
                        id: entrypoint_node,
                        vec: entrypoint_vec,
                    },
                    &query,
                    curr_layer,
                    ef_search,
                    &self.id_to_vec,
                );
            }

            vec![]
        } else {
            println!("Top layer does not have an entrypoint.");
            return vec![];
        }
    }
}
