use serde::{Deserialize, Serialize};

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>(pub [u32; N]);

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Vector<const N: usize>(#[serde(with = "serde_arrays")] pub [f32; N]);

pub trait Index<const N: usize> {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize);
    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)>;
}

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

    pub fn cosine_similarity(&self, other: &Vector<N>) -> f32 {
        self.dot_product(other) / (self.dot_product(self).sqrt() * other.dot_product(other).sqrt())
    }
}
