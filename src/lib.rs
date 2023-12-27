use dashmap::DashSet;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::cmp::{min, Ordering};
use std::collections::HashSet;
mod indexes;

use indexes::ivfflat::IVFFlatIndex;

use rand::seq::{index, SliceRandom};

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>(pub [u32; N]);
#[derive(Copy, Clone)]
pub struct Vector<const N: usize>(pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self.0.iter().zip(&other.0).map(|(a, b)| b - a);
        let result: [f32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        Vector(result)
    }

    pub fn dot_product(&self, other: &Vector<N>) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum::<f32>()
    }

    pub fn average(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self.0.iter().zip(&other.0).map(|(a, b)| (a + b) / 2.0);
        let results: [f32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        return Vector(results);
    }

    pub fn to_hashkey(&self) -> HashKey<N> {
        let vals = self.0.iter().map(|a| a.to_bits());
        let results: [u32; N] = vals.collect::<Vec<_>>().try_into().unwrap();
        return HashKey::<N>(results);
    }

    pub fn squared_euclidean(&self, other: &Vector<N>) -> f32 {
        return self
            .0
            .iter()
            .zip(other.0)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
    }
}

struct Hyperplane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> Hyperplane<N> {
    /// Checks if the given vector is "above" the hyperplane.
    /// The hyperplane is defined by its normal vector (and its constant)
    /// and when the dot product is taken between the normal vector and
    /// the query vector, a positive value means both vectors are in the same
    /// direction, hence "above" - since a.b = |a||b|cos(0), where 0 is the
    /// angle between vectors a and b. If the dot product is negative,
    /// 0 is > 90deg.
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        return &self.coefficients.dot_product(&point) + self.constant >= 0.0;
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode>),
}

struct InnerNode<const N: usize> {
    hyperplane: Hyperplane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

struct LeafNode(Vec<usize>);

pub struct ANNIndex<const N: usize> {
    // vector containing all the trees that make up the index, with each element in the vector
    // indicating the root node of the tree
    trees: Vec<Node<N>>,
    // stores all the vectors within the index in a global contiguous location
    values: Vec<Vector<N>>,
    ids: Vec<usize>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &Vec<usize>,
        all_vecs: &Vec<Vector<N>>,
    ) -> (Hyperplane<N>, Vec<usize>, Vec<usize>) {
        // sample two random vectors from the indexes, and use these two vectors to form a hyperplane boundary
        let samples: Vec<_> = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect();

        let (a, b) = (*samples[0], *samples[1]);

        // construct the hyperplane by getting the plane's coefficients (normal vector), and the
        // its constant (the constant term from a plane's cartesian eqn).
        let coefficients = all_vecs[a].subtract_from(&all_vecs[b]);
        let point_on_plane = all_vecs[a].average(&all_vecs[b]);
        let constant = -coefficients.dot_product(&point_on_plane);

        let hyperplane = Hyperplane {
            coefficients,
            constant,
        };

        // assign each index (which points to a specific vector in the embedding matrix) to
        // either `above` or `below` the hyperplane.
        let mut above: Vec<usize> = Vec::new();
        let mut below: Vec<usize> = Vec::new();

        for id in indexes {
            if hyperplane.point_is_above(&all_vecs[*id]) {
                above.push(*id);
            } else {
                below.push(*id);
            }
        }

        return (hyperplane, above, below);
    }

    fn build_a_tree(max_size: usize, indexes: &Vec<usize>, all_vecs: &Vec<Vector<N>>) -> Node<N> {
        if indexes.len() < max_size {
            return Node::Leaf(Box::new(LeafNode(indexes.clone())));
        } else {
            let (hyperplane, above, below) = Self::build_hyperplane(&indexes, all_vecs);

            let left_node = Self::build_a_tree(max_size, &above, all_vecs);
            let right_node = Self::build_a_tree(max_size, &below, all_vecs);

            return Node::Inner(Box::new(InnerNode {
                hyperplane,
                left_node,
                right_node,
            }));
        }
    }

    fn deduplicate(
        all_vectors: &Vec<Vector<N>>,
        all_indexes: &Vec<usize>,
        dedup_vecs: &mut Vec<Vector<N>>,
        dedup_vec_ids: &mut Vec<usize>,
    ) {
        let mut hashes_seen: HashSet<HashKey<N>> = HashSet::new();
        for id in 0..all_vectors.len() {
            let hash_key = all_vectors[id].to_hashkey();
            if !hashes_seen.contains(&hash_key) {
                hashes_seen.insert(hash_key);

                // requires the Copy trait to copy a vector from `all_vectors` into `dedup_vecs`
                dedup_vecs.push(all_vectors[id]);
                dedup_vec_ids.push(all_indexes[id]);
            }
        }
    }

    pub fn build_index(
        num_trees: usize,
        max_size: usize,
        vectors: Vec<Vector<N>>,
        vector_ids: Vec<usize>,
    ) -> ANNIndex<N> {
        let mut dedup_vecs = vec![];
        let mut dedup_vec_ids = vec![];
        Self::deduplicate(&vectors, &vector_ids, &mut dedup_vecs, &mut dedup_vec_ids);

        // TODO: can parallelize this
        // maps each index to the unique vector
        let all_indexes_from_unique_vecs = (0..dedup_vecs.len()).collect();

        let trees: Vec<Node<N>> = (0..num_trees)
            .into_par_iter()
            .map(|_| Self::build_a_tree(max_size, &all_indexes_from_unique_vecs, &dedup_vecs))
            .collect();

        ANNIndex {
            trees,
            values: dedup_vecs,
            ids: dedup_vec_ids,
        }
    }

    fn tree_result(
        &self,
        query: Vector<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &DashSet<usize>,
    ) -> i32 {
        match tree {
            Node::Leaf(leaf_node) => {
                let leaf_values_index = &(leaf_node.0);
                let mut num_candidates = 0;
                if leaf_values_index.len() < n as usize {
                    // take all the candidates
                    num_candidates = leaf_values_index.len();

                    for i in leaf_values_index {
                        candidates.insert(*i);
                    }
                } else {
                    // only take candidate whose vectors are those closest to the query in distance
                    num_candidates = n as usize;

                    let top_candidates: Vec<usize> = leaf_values_index
                        .into_iter()
                        .map(|idx| {
                            let curr_vector = &self.values[*idx];
                            (idx, curr_vector.squared_euclidean(&query))
                        })
                        .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .take(n as usize)
                        .map(|(idx, distance)| *idx)
                        .collect();

                    for i in top_candidates {
                        candidates.insert(i);
                    }
                }

                num_candidates as i32
            }
            Node::Inner(inner_node) => {
                let is_above = inner_node.hyperplane.point_is_above(&query);
                let (main, backup) = match is_above {
                    true => (&inner_node.right_node, &inner_node.left_node),
                    false => (&inner_node.left_node, &inner_node.right_node),
                };

                match self.tree_result(query, n, main, candidates) {
                    k if k < n => self.tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }

    pub fn search_approximate(&self, query: Vector<N>, top_k: i32) -> Vec<(usize, f32)> {
        // using dashset instead of hashset as it support concurrent mutations to the set
        let candidates = DashSet::new();

        self.trees.par_iter().for_each(|tree| {
            self.tree_result(query, top_k, tree, &candidates);
        });

        candidates
            .into_iter()
            .map(|idx| (idx, self.values[idx].squared_euclidean(&query)))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(top_k as usize)
            // TODO: consider removing the self.ids, as it's slightly misleading. this vector is deduplicated, so that the
            // the number of elements in self.ids is the same as the unique vectors, but the elements themselves are indices
            // to the non-dedup'ed IDs
            .map(|(idx, distance)| (self.ids[idx], distance))
            .collect()
    }
}
