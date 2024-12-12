use std::collections::HashMap;
use std::time::Instant;

use vers::utils;
use vers::{Index, Vector};

use itertools::Itertools;
use rand::Rng;
use std::cmp::min;

use std::{
    fs,
    io::{self, BufReader, BufWriter},
    path::Path,
};

use vers::{AdjacencyItem, DistanceCandidatePair};
const DIM: usize = 300;

fn test_adj_serde() {
    let mut i = AdjacencyItem::new();
    i.insert(&DistanceCandidatePair {
        candidate_id: 1,
        distance: 2.0,
    });

    let file = fs::File::create("test.pkl");
    let writer = BufWriter::new(file.unwrap());

    bincode::serialize_into(writer, &i)
        .map_err(|err| {
            // ensures func returns a Result with io::Error
            io::Error::new(
                io::ErrorKind::Other,
                format!("Serialization error: {}", err),
            )
        })
        .unwrap();

    let file = fs::File::open("test.pkl");
    let reader = BufReader::new(file.unwrap());

    let new_i: AdjacencyItem = bincode::deserialize_from(reader)
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Deserialization error: {}", err),
            )
        })
        .unwrap();

    println!("{:?}", new_i.neighbours);
}
fn main() {
    let (wiki, mut word_to_idx, mut idx_to_word, test_embs) =
        utils::load_wiki_vector::<DIM>("../wiki-news-300d-1M.vec");

    let start = Instant::now();

    // utils::test_ivfflat(
    //     &wiki,
    //     &mut word_to_idx,
    //     &mut idx_to_word,
    //     20,
    //     3,
    //     10,
    //     &test_embs,
    // );

    utils::test_hnsw(
        &wiki,
        &mut word_to_idx,
        &mut idx_to_word,
        6,
        200,
        32,
        16,
        &test_embs,
    );

    // utils::test_lsh(&wiki, &word_to_idx, &idx_to_word, 8, 100);

    // fn get_insertion_layer(layer_multiplier: f32, num_layers: usize) -> usize {
    //     let random_val: f32 = rand::thread_rng().gen();
    //     let l = -(random_val.ln() * layer_multiplier) as usize;
    //     min(l, num_layers - 1)

    //     // num_layers = 6
    //     // min(l, num_layers), l = 0 => 0 (case when insertion layer is at the top)
    //     // min(l, num_layers), l >= 6 => 6 (case when insertion layer is at the bottommost - most common case)
    //     // min(l, num_layers), 0 < l < 6 => l (e.g. 1, 2, 3, 4, 5. insert at l and below)
    // }
    // let mut results: HashMap<usize, usize> = HashMap::new();
    // for _ in 0..10000000 {
    //     let result = get_insertion_layer(1.0 / (16 as f32).ln(), 6);
    //     *results.entry(result).or_insert(0) += 1;
    // }
    // for (layer, count) in results.iter().sorted_by_key(|&(k, _)| k) {
    //     println!("Layer {}: {}", layer, count);
    // }
    let duration = start.elapsed();
    println!("Time taken to test: {:?}", duration);
}
