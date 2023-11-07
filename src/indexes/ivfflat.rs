use crate::{indexes::base::Vector, HashKey};
use rand::{self, Rng};
pub struct IVFFlatIndex {
    s: String,
}

fn cosine_similarity<const N: usize>(a: &Vector<N>, b: &Vector<N>) -> f32 {
    let dot_product = a.dot_product(b);

    let norm_a = a.dot_product(a).sqrt();
    let norm_b = b.dot_product(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

fn initialize_centroids<const N: usize>(data: &[Vector<N>], k: usize) -> Vec<Vector<N>> {
    let mut rng = rand::thread_rng();

    (0..k)
        .map(|_| {
            let random_index = rng.gen_range(0..data.len());
            data[random_index].clone()
        })
        .collect()
}

fn assign_to_clusters<const N: usize>(data: &[Vector<N>], centroids: &[Vector<N>]) -> Vec<usize> {
    data.iter()
        .map(|data_point| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_to_centroid_a = cosine_similarity(data_point, a);
                    let dist_to_centroid_b = cosine_similarity(data_point, b);

                    dist_to_centroid_a.partial_cmp(&dist_to_centroid_b).unwrap()
                })
                .unwrap()
                .0
        })
        .collect()
}

fn update_centroids<const N: usize>(
    data: &[Vector<N>],
    assignments: &[usize],
    k: usize,
) -> Vec<Vector<N>> {
    let mut new_centroids = vec![Vector([0.0; N]); k];

    for (data_point, cluster_index) in data.iter().zip(assignments) {
        new_centroids[*cluster_index] = new_centroids[*cluster_index].average(data_point);
    }

    new_centroids
}

pub fn kmeans<const N: usize>(
    data: &[Vector<N>],
    k: usize,
    max_iterations: usize,
) -> (Vec<Vector<N>>, Vec<usize>) {
    let mut centroids = initialize_centroids(data, k);

    for _ in 0..max_iterations {
        let assignments = assign_to_clusters(data, &centroids);
        let new_centroids = update_centroids(data, &assignments, k);

        let centroid_hashes: Vec<_> = centroids
            .iter()
            .map(|v: &Vector<N>| v.to_hashkey())
            .collect();
        let new_centroid_hashes: Vec<_> = new_centroids.iter().map(|v| v.to_hashkey()).collect();

        if centroid_hashes == new_centroid_hashes {
            break;
        }

        centroids = new_centroids;
    }

    let assignments = assign_to_clusters(data, &centroids);
    (centroids, assignments)
}
