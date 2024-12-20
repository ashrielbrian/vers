use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

use crate::Vector;

#[derive(PartialEq)]
pub struct DistanceMaxCandidatePair<'a, const N: usize> {
    pub candidate_id: &'a usize,
    pub candidate_vec: &'a Vector<N>,
    pub distance: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DistanceCandidatePair {
    pub candidate_id: usize,
    pub distance: f32,
}

impl<const N: usize> Eq for DistanceMaxCandidatePair<'_, N> {}

impl<'a, const N: usize> PartialOrd for DistanceMaxCandidatePair<'a, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<'a, const N: usize> Ord for DistanceMaxCandidatePair<'a, N> {
    fn cmp(&self, other: &DistanceMaxCandidatePair<N>) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// we need the following traits so we can use DistanceCandidatePair within a HashSet
impl Hash for DistanceCandidatePair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.candidate_id.hash(state);
    }
}

impl Eq for DistanceCandidatePair {}

impl PartialEq for DistanceCandidatePair {
    fn eq(&self, other: &Self) -> bool {
        self.candidate_id == other.candidate_id
    }
}

impl<'a> PartialOrd for DistanceCandidatePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for DistanceCandidatePair {
    fn cmp(&self, other: &DistanceCandidatePair) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct AdjacencyItem {
    pub max_heap: BinaryHeap<DistanceCandidatePair>,
    pub neighbours: HashSet<usize>,
}

impl AdjacencyItem {
    pub fn new() -> Self {
        AdjacencyItem {
            max_heap: BinaryHeap::new(),
            neighbours: HashSet::new(),
        }
    }
    pub fn insert(&mut self, u: &DistanceCandidatePair) {
        self.neighbours.insert(u.candidate_id.clone());
        self.max_heap.push(u.clone());
    }

    pub fn len(&self) -> usize {
        self.neighbours.len()
    }

    pub fn trim(&mut self, max_neighbours: usize) {
        // removes elements with the largest distances first until <= num_neighbours
        while self.max_heap.len() > max_neighbours {
            let to_remove = self.max_heap.pop().unwrap();
            self.neighbours.remove(&to_remove.candidate_id);
        }
    }

    pub fn max_distance(&self) -> f32 {
        self.max_heap.peek().unwrap().distance
    }
}

impl Serialize for AdjacencyItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Since BinaryHeap does not implement Serialize/Deserialize, we need to convert it to a Vec first.
        let max_heap_vec: Vec<DistanceCandidatePair> = self.max_heap.clone().into_sorted_vec();
        let neighbours_vec: Vec<usize> = self.neighbours.iter().cloned().collect();

        let adj_item = AdjacencyItemSer {
            max_heap: max_heap_vec,
            neighbours: neighbours_vec,
        };

        adj_item.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AdjacencyItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let adj_item = AdjacencyItemSer::deserialize(deserializer)?;

        let max_heap = BinaryHeap::from(adj_item.max_heap);
        let neighbours = adj_item.neighbours.into_iter().collect();

        Ok(AdjacencyItem {
            max_heap,
            neighbours,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct AdjacencyItemSer {
    max_heap: Vec<DistanceCandidatePair>,
    neighbours: Vec<usize>,
}
