use bincode;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufReader, BufWriter},
    path::Path,
};

use std::simd::{f32x4, f32x64, num::SimdFloat};

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>(pub [u32; N]);

// need for alignment otherwise will result in simd failures
#[repr(align(256))]
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Vector<const N: usize>(#[serde(with = "serde_arrays")] pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn to_vec(&self) -> Vec<f32> {
        self.0.to_vec()
    }
}

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

    pub fn magnitude(&self) -> f32 {
        self.dot_product(self).sqrt()
    }

    pub fn normalize(&self) -> Vector<N> {
        let magnitude = self.magnitude();
        if magnitude < 1e-6 {
            return self.clone(); // or return zero vector depending on your needs
        }
        self.divide_by_scalar(magnitude)
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
        const EPSILON: f32 = 1e-5;

        let self_dot = self.dot_product(self);
        let other_dot = other.dot_product(other);

        // Check if the vectors are actually normalized (should be very close to 1.0)
        if (self_dot - 1.0).abs() > EPSILON || (other_dot - 1.0).abs() > EPSILON {
            println!(
                "Warning: Vectors may not be properly normalized. Self dot: {}, Other dot: {}",
                self_dot, other_dot
            );
        }

        let norm_a = self.magnitude();
        let norm_b = other.magnitude();

        if norm_a < EPSILON || norm_b < EPSILON {
            return EPSILON;
        }

        self.dot_product(other) / (norm_a * norm_b)
    }

    pub fn cosine_similarity_simd(&self, _other: &Vector<N>) -> f32 {
        // to be implemented
        return 0.0;
    }

    pub fn squared_euclidean_simd(&self, other: &Vector<N>) -> f32 {
        // Constants for different SIMD widths
        const LANES_64: isize = 64;
        const LANES_4: isize = 4;

        assert_eq!(self.0.len(), other.0.len());

        let vec_len = self.0.len() as isize;
        let mut res = 0.0;

        // (u - v)(u - v).sum()
        let u_ptr: *const f32 = self.0.as_ptr();
        let v_ptr: *const f32 = other.0.as_ptr();

        // Calculate number of 64-wide chunks
        let chunks_64 = vec_len / LANES_64;

        // Process 64-element chunks
        if chunks_64 > 0 {
            let simd_u_ptr_64 = u_ptr as *const f32x64;
            let simd_v_ptr_64 = v_ptr as *const f32x64;

            for i in 0..chunks_64 {
                unsafe {
                    let temp = *simd_u_ptr_64.offset(i) - *simd_v_ptr_64.offset(i);
                    res += (temp * temp).reduce_sum();
                }
            }
        }

        // Calculate remaining elements after 64-wide chunks
        let processed_64 = chunks_64 * LANES_64;
        let remaining = vec_len - processed_64;

        // Process remaining elements with 4-wide vectors
        let chunks_4 = remaining / LANES_4;

        if chunks_4 > 0 {
            // Offset pointers to start after the 64-wide chunks
            let u_ptr_4 = unsafe { u_ptr.offset(processed_64) };
            let v_ptr_4 = unsafe { v_ptr.offset(processed_64) };

            let simd_u_ptr_4 = u_ptr_4 as *const f32x4;
            let simd_v_ptr_4 = v_ptr_4 as *const f32x4;

            for i in 0..chunks_4 {
                unsafe {
                    let temp = *simd_u_ptr_4.offset(i) - *simd_v_ptr_4.offset(i);
                    res += (temp * temp).reduce_sum();
                }
            }
        }

        // deal with excess elements that cannot fit into the simd chunk
        let processed_4 = processed_64 + (chunks_4 * LANES_4);
        let final_remaining = vec_len - processed_4;

        // Process remaining individual elements
        if final_remaining > 0 {
            for i in processed_4..vec_len {
                unsafe {
                    let diff = *u_ptr.offset(i) - *v_ptr.offset(i);
                    res += diff * diff;
                }
            }
        }

        res
    }
}
