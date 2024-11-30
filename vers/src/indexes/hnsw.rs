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
    nodes: HashMap<String, Vector<N>>,
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
    pub fn add_edge(&mut self, u: &String, v: &String) {
        self._add_edge(u, v);
        self._add_edge(v, u);
    }

    pub fn add_node(&mut self, node: &Node<N>) {
        self.nodes.insert(node.id.clone(), node.vec.clone());
    }

    pub fn trim_edges(
        &mut self,
        node_id: &String,
        max_num_edges: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
    ) {
        if let Some(neighbours) = self.adjacency_list.get_mut(node_id) {
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
                        candidate_id: neighbour.clone(),
                        distance: dist,
                    });
                }

                let new_neighbours = get_top_k_smallest_nodes(max_heap, max_num_edges);
                self.adjacency_list
                    .insert(node_id.clone(), HashSet::from_iter(new_neighbours));
            }
        }
    }
}

fn get_top_k_smallest_nodes(
    max_heap: BinaryHeap<DistanceMaxCandidatePair>,
    m: usize,
) -> Vec<String> {
    // returns nodes in descending order, where the first node has the largest distance
    let mut nodes_vec = itertools::unfold(max_heap, |heap| heap.pop())
        .map(|node| node.candidate_id)
        .collect::<Vec<_>>();

    nodes_vec.reverse();
    nodes_vec.truncate(m);
    nodes_vec
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
    fn get_insertion_layer(layer_multiplier: usize, max_layer: usize) -> usize {
        let random_val: f32 = rand::thread_rng().gen();
        let l = -(random_val.ln() * (layer_multiplier as f32)) as usize;
        min(l, max_layer)
    }

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
        layer: &mut HNSWLayer<N>,
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
        layer.add_node(target_node);
        for new_neighbour in selected_neighbours.iter() {
            layer.add_edge(&target_node.id, &new_neighbour);
        }

        // check and reduce the num of edges if exceeds m, for all the neighbours.
        // this is done because these neighbours have their own neighbours, and adding an edge between
        // the new node and them could have caused these neighbours to exceed m.
        for neighbour in selected_neighbours {
            layer.trim_edges(neighbour, m, id_to_vec);
        }
    }

    pub fn _search_layer(
        entrypoint: Node<N>,
        query_vector: &Vector<N>,
        layer: &HNSWLayer<N>,
        ef_construction: usize,
        id_to_vec: &HashMap<String, Vector<N>>,
    ) -> Vec<String> {
        // returns a list of candidates closest (of length ef_construction) to the query vector for a given layer
        let mut queue: VecDeque<&String> = VecDeque::new();
        let mut candidates_heap: BinaryHeap<DistanceMaxCandidatePair> = BinaryHeap::new();

        queue.push_back(entrypoint.id);

        while let Some(node) = queue.pop_front() {
            // get all neighbours in the current node, then process these neighbourhood nodes
            if let Some(neighbour_ids) = layer.adjacency_list.get(node) {
                for neighbour_id in neighbour_ids {
                    let neighbour_dist =
                        query_vector.squared_euclidean(&id_to_vec.get(neighbour_id).unwrap());

                    // if the current neighbour distance is smaller than the candidate with the largest distance, replace
                    // this candidate with the current neighbour
                    if candidates_heap.len() < ef_construction
                        || (candidates_heap.len() > 0
                            && neighbour_dist < candidates_heap.peek().unwrap().distance)
                    {
                        queue.push_back(neighbour_id);
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

fn _get_best_candidate<'a, 'b, const N: usize>(
    candidates: &'a Vec<String>,
    id_to_vec: &'b HashMap<String, Vector<N>>,
) -> Option<Node<'a, 'b, N>> {
    if let Some(best_candidate_id) = candidates.last() {
        return Option::Some(Node {
            id: best_candidate_id,
            vec: id_to_vec.get(best_candidate_id).unwrap(),
        });
    }
    None
}
struct Node<'a, 'b, const N: usize> {
    id: &'a String,
    vec: &'b Vector<N>,
}

impl<const N: usize> Index<N> for HNSWIndex<N> {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize) {}

    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let ef_search = 100;

        // get the topmost layer to access the entrypoint. use the closest entrypoint
        let top_layer = &self.layers.last().unwrap();

        if let Some((entrypoint_node, entrypoint_vec)) = top_layer.nodes.iter().next() {
            // start from the second last layer and work downwards to layer 0
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
