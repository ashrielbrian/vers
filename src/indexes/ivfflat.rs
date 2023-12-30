use crate::indexes::base::{HashKey, Vector};
use rand::{self, Rng};
pub struct IVFFlatIndex<const N: usize> {
    num_centroids: usize,
    values: Vec<Vector<N>>,
    centroids: Vec<Vector<N>>,
    assignments: Vec<usize>,
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
            .iter()
            .map(|data_point| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_to_centroid_a = data_point.cosine_similarity(a);
                        let dist_to_centroid_b = data_point.cosine_similarity(b);

                        dist_to_centroid_a.partial_cmp(&dist_to_centroid_b).unwrap()
                    })
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn update_centroids(data: &Vec<Vector<N>>, assignments: &[usize], k: usize) -> Vec<Vector<N>> {
        // k Vectors of N size
        let mut new_centroids = vec![Vector([0.0; N]); k];

        for (data_point, cluster_index) in data.iter().zip(assignments) {
            // TODO: technically this adds a data point of zeros to each centroid, skewing the average
            // need to remove the effect of this zeros datapoint from each centroid.
            new_centroids[*cluster_index] = new_centroids[*cluster_index].average(data_point);
        }

        new_centroids
    }

    pub fn build_kmeans(
        data: &Vec<Vector<N>>,
        k: usize,
        max_iterations: usize,
    ) -> (Vec<Vector<N>>, Vec<usize>) {
        let mut centroids = Self::initialize_centroids(data, k);

        for _ in 0..max_iterations {
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
        data: Vec<Vector<N>>,
        num_attempts: usize,
        k: usize,
        max_iterations: usize,
    ) -> Self {
        let mut best_cost = f32::INFINITY;
        let mut best_centroids: Vec<Vector<N>> = Vec::new();
        let mut best_assignments: Vec<usize> = Vec::new();
        for _ in (0..num_attempts) {
            let (centroids, assignments) = Self::build_kmeans(&data, k, max_iterations);
            let cost = Self::calculate_kmeans_cost(&data, &centroids, &assignments);

            if cost < best_cost {
                best_cost = cost;
                best_centroids = centroids;
                best_assignments = assignments;
            }
        }

        IVFFlatIndex {
            num_centroids: k,
            values: data,
            centroids: best_centroids,
            assignments: best_assignments,
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
