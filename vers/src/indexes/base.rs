use bincode;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufReader, BufWriter},
    path::Path,
};

use std::simd::{f32x32, num::SimdFloat};

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>(pub [u32; N]);

// need for alignment otherwise will result in simd failures
#[repr(align(64))]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Vector<const N: usize>(#[serde(with = "serde_arrays")] pub [f32; N]);

// the trait bound DeserialzedOwned is required for bincode::deserialize_from, basically
// specifying that any type that implements Index, must also implement Deserialize, Serialize(and Sized).
pub trait Index<const N: usize>: Sized + DeserializeOwned + Serialize {
    fn add(&mut self, embedding: Vector<N>, vec_id: usize);
    fn search_approximate(&self, query: Vector<N>, top_k: usize) -> Vec<(usize, f32)>;

    fn save_index(&self, file_path: &str) -> io::Result<()> {
        // open file with a buffer
        let file = fs::File::create(file_path)?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, &self).map_err(|err| {
            // ensures func returns a Result with io::Error
            io::Error::new(
                io::ErrorKind::Other,
                format!("Serialization error: {}", err),
            )
        })
    }

    fn load_index(file_path: impl AsRef<Path>) -> io::Result<Self>
    where
        Self: Sized,
    {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);

        bincode::deserialize_from(reader).map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Deserialization error: {}", err),
            )
        })
    }
}

impl<const N: usize> Vector<N> {
    pub fn add(&self, other: &Vector<N>) -> Vector<N> {
        let result: [f32; N] = self
            .0
            .iter()
            .zip(other.0)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Vector(result)
    }

    pub fn divide_by_scalar(&self, scalar: f32) -> Vector<N> {
        let result: [f32; N] = self
            .0
            .iter()
            .map(|a| a / scalar)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Vector(result)
    }

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
        let norm_a = self.dot_product(self).sqrt();
        let norm_b = other.dot_product(other).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        self.dot_product(other) / (norm_a * norm_b)
    }

    pub fn squared_euclidean_simd(&self, other: &Vector<N>) -> f32 {
        /// this implementation uses 32 lanes specifically.
        const LANES: isize = 32;

        assert_eq!(self.0.len(), other.0.len());

        let vec_len = self.0.len() as isize;
        let chunks = vec_len / LANES;
        let mut res = 0.0;

        // (u - v)(u - v).sum()
        let u_ptr: *const f32 = self.0.as_ptr();
        let v_ptr: *const f32 = other.0.as_ptr();

        // deal with excess elements that cannot fit into the simd chunk
        for i in (LANES * chunks)..vec_len {
            unsafe {
                res += (*u_ptr.offset(i) - *v_ptr.offset(i)).powi(2);
            }
        }

        let simd_u_ptr = u_ptr as *const f32x32;
        let simd_v_ptr = v_ptr as *const f32x32;

        for i in 0..chunks {
            unsafe {
                let temp = *simd_u_ptr.offset(i) - *simd_v_ptr.offset(i);
                res += (temp * temp).reduce_sum();
            }
        }

        res
    }
}
