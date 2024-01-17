use super::base::Index;
use crate::indexes::base::Vector;
use itertools::Itertools;
use rand::{self, Rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct IVFFlatIndex<const N: usize> {
    num_centroids: usize,
    values: Vec<Vector<N>>,
    centroids: Vec<Vector<N>>,
    assignments: Vec<usize>,
    ids: Vec<Vec<usize>>,
}

impl<const N: usize> IVFFlatIndex<N> {
    fn initialize_centroids(all_vecs: &Vec<Vector<N>>, k: usize) -> Vec<Vector<N>> {
        let mut rng = rand::thread_rng();

        (0..k)
            .map(|_| {
                let random_index = rng.gen_range(0..all_vecs.len());
                all_vecs[random_index].clone()
            })
            .collect()
    }

    fn assign_to_clusters(all_vecs: &Vec<Vector<N>>, centroids: &Vec<Vector<N>>) -> Vec<usize> {
        all_vecs
            .par_iter()
            .map(|data_point| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_to_centroid_a = data_point.squared_euclidean(a);
                        let dist_to_centroid_b = data_point.squared_euclidean(b);

                        dist_to_centroid_a.partial_cmp(&dist_to_centroid_b).unwrap()
                    })
                    .unwrap()
                    .0
            })
            .collect()
    }
    fn update_centroids(data: &Vec<Vector<N>>, assignments: &[usize], k: usize) -> Vec<Vector<N>> {
        let mut sums = vec![Vector([0.0; N]); k];
        let mut counts = vec![0; k];

        // Sum all vectors for each centroid and count the number of vectors per centroid
        for (data_point, &cluster_index) in data.iter().zip(assignments.iter()) {
            sums[cluster_index] = sums[cluster_index].add(data_point);
            counts[cluster_index] += 1;
        }

        // Calculate the average for each centroid
        let mut new_centroids = Vec::with_capacity(k);
        for (sum, &count) in sums.iter().zip(counts.iter()) {
            if count > 0 {
                // If there are vectors assigned to the centroid, calculate the average
                new_centroids.push(sum.divide_by_scalar(count as f32));
            } else {
                // If no vectors are assigned, we can either choose a random vector or reinitialize the centroid
                // For simplicity, we'll reinitialize to a zero vector here
                new_centroids.push(Vector([0.0; N]));
            }
        }

        new_centroids
    }

    pub fn build_kmeans(
        data: &Vec<Vector<N>>,
        k: usize,
        max_iterations: usize,
    ) -> (Vec<Vector<N>>, Vec<usize>) {
        let mut centroids = Self::initialize_centroids(data, k);

        for i in 0..max_iterations {
            let assignments = Self::assign_to_clusters(data, &centroids);
            let new_centroids = Self::update_centroids(data, &assignments, k);

            let centroid_hashes: Vec<_> = centroids
                .iter()
                .map(|v: &Vector<N>| v.to_hashkey())
                .collect();
            let new_centroid_hashes: Vec<_> =
                new_centroids.iter().map(|v| v.to_hashkey()).collect();

            if centroid_hashes == new_centroid_hashes {
                break;
            }

            centroids = new_centroids;
        }

        let assignments = Self::assign_to_clusters(data, &centroids);
        (centroids, assignments)
    }

    pub fn build_index(
        num_clusters: usize,
        num_attempts: usize,
        max_iterations: usize,
        vectors: &Vec<Vector<N>>,
    ) -> Self {
        let mut best_cost = f32::INFINITY;
        let mut best_centroids: Vec<Vector<N>> = Vec::new();
        let mut best_assignments: Vec<usize> = Vec::new();
        for i in 0..num_attempts {
            let (centroids, assignments) =
                Self::build_kmeans(&vectors, num_clusters, max_iterations);
            let cost = Self::calculate_kmeans_cost(&vectors, &centroids, &assignments);

            if cost < best_cost {
                best_cost = cost;
                best_centroids = centroids;
                best_assignments = assignments;
            }
        }

        let mut ids: Vec<Vec<usize>> = vec![vec![]; num_clusters];
        best_assignments
            .iter()
            .enumerate()
            .for_each(|(vec_id, cluster_id)| ids[*cluster_id].push(vec_id));

        IVFFlatIndex {
            num_centroids: num_clusters,
            values: vectors.clone(),
            centroids: best_centroids,
            assignments: best_assignments,
            ids: ids,
        }
    }

    fn calculate_kmeans_cost(
        data: &Vec<Vector<N>>,
        centroids: &Vec<Vector<N>>,
        assignments: &Vec<usize>,
    ) -> f32 {
        data.iter()
            .zip(assignments)
            .map(|(data_point, cluster_index)| {
                data_point.squared_euclidean(&centroids[*cluster_index])
            })
            .fold(0.0, |acc, val| acc + val)
    }
}

impl<const N: usize> Index<N> for IVFFlatIndex<N> {
    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        // get nearest clusters, and sort distances
        let nearest_centroids: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, centroid.squared_euclidean(&query)))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .collect();

        let mut candidates: Vec<(usize, f32)> = Vec::new();

        // search nearest cluster
        let mut curr_cluster = 0;
        let mut remainder = top_k;
        while candidates.len() < top_k {
            let cluster_id = nearest_centroids[curr_cluster];
            let vec_ids_in_cluster = &self.ids[cluster_id.0];

            let potential_candidates: Vec<(usize, f32)> = vec_ids_in_cluster
                .iter()
                .map(|vec_id| (vec_id, self.values[*vec_id]))
                .map(|(vec_id, vec)| (*vec_id, vec.squared_euclidean(&query)))
                .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .take(top_k)
                .collect();

            // return top-k from nearest cluster
            if potential_candidates.len() < remainder {
                // if insufficient results, search the other clusters until we get k results
                remainder -= potential_candidates.len();
                candidates.extend(potential_candidates);
                curr_cluster += 1;
            } else if potential_candidates.len() > remainder {
                for i in 0..remainder {
                    candidates.push(potential_candidates[i]);
                }
                break;
            } else {
                candidates.extend(potential_candidates);
                break;
            }
        }

        candidates
    }

    fn add(&mut self, embedding: Vector<N>, vec_id: usize) {
        let closest_centroid_id = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.squared_euclidean(&embedding)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let vec_id = self.assignments.len();
        self.values.push(embedding);
        self.assignments.push(closest_centroid_id.0);
        self.ids[closest_centroid_id.0].push(vec_id);
    }
}
