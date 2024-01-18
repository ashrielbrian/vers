use std::collections::HashMap;
use std::time::Instant;

use vers::utils;
use vers::{Index, Vector};

const DIM: usize = 300;

fn main() {
    let (wiki, mut word_to_idx, mut idx_to_word, test_embs) =
        utils::load_wiki_vector::<DIM>("../wikidata.vec");

    let start = Instant::now();
    utils::test_ivfflat(
        &wiki,
        &mut word_to_idx,
        &mut idx_to_word,
        20,
        3,
        10,
        &test_embs,
    );
    let duration = start.elapsed();
    println!("Time taken to test IVFFlat: {:?}", duration);

    // utils::test_lsh(&wiki, &word_to_idx, &idx_to_word, 8, 100);
}
