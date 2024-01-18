use std::collections::HashMap;

use vers::utils;
use vers::{Index, Vector};

const DIM: usize = 300;

fn main() {
    let (wiki, mut word_to_idx, mut idx_to_word, test_embs) =
        utils::load_wiki_vector::<DIM>("../wikidata.vec");

    utils::test_ivfflat(
        &wiki,
        &mut word_to_idx,
        &mut idx_to_word,
        20,
        3,
        10,
        &test_embs,
    );

    utils::test_lsh(&wiki, &word_to_idx, &idx_to_word, 8, 100);
}
