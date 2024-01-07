use itertools::Itertools;
use std::collections::HashMap;
use vers::utils;
use vers::ANNIndex;
use vers::IVFFlatIndex;
use vers::{Index, Vector};

const DIM: usize = 300;

fn search_exhaustive<const N: usize>(
    vector_data: &Vec<Vector<N>>,
    query: &Vector<N>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let candidates: Vec<(usize, f32)> = vector_data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.squared_euclidean(&query)))
        .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .take(top_k)
        .collect();

    candidates
}

fn test_lsh<const N: usize>(
    vectors: &Vec<Vector<N>>,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &HashMap<usize, String>,
    num_trees: usize,
    max_node_size: usize,
) {
    println!("LSH Index:-----");
    let index_file_name = "lsh.index";

    // build the LSH index
    let vector_ids: Vec<usize> = (0..vectors.len()).collect();
    let index = ANNIndex::build_index(num_trees, max_node_size, vectors, &vector_ids);

    // search the index
    let results = index.search_approximate(vectors[*word_to_idx.get("priceless").unwrap()], 20);

    // visualize the results
    for (i, (results_idx, distance)) in results.iter().enumerate() {
        println!(
            "{i}. Word: {}. Distance: {}",
            idx_to_word.get(results_idx).unwrap(),
            distance.sqrt()
        )
    }

    // persist the index
    let _ = index.save_index(index_file_name);

    // load the index
    let _reload_index: ANNIndex<N> = ANNIndex::load_index(index_file_name).unwrap();
}

fn test_ivfflat<const N: usize>(
    vectors: &Vec<Vector<N>>,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &mut HashMap<usize, String>,
    num_clusters: usize,
    num_attempts: usize,
    max_iterations: usize,
    test_embs: &Vec<(String, [f32; N])>,
) {
    println!("IVFFlat Index:-----");
    let index_file_name = "ivfflat.index";

    // build the IVFFlat index
    let mut index = IVFFlatIndex::build_index(num_clusters, num_attempts, max_iterations, &vectors);

    // test adding new embeddings to the index
    for ((word, emb), vec_id) in test_embs.into_iter().zip([999993])
    // since there are 999994 total wiki vectors, and we omitted one element (queen)
    {
        println!("Inserting {} {}", word, vec_id);
        idx_to_word.insert(vec_id, word.to_string());
        index.add(Vector(*emb), vec_id);
    }

    // persist the index
    match index.save_index(index_file_name) {
        Ok(_) => println!("Index saved successfully!"),
        Err(e) => eprintln!("Index save failed: {}", e),
    };

    // load the index
    let reload_index = match IVFFlatIndex::load_index(index_file_name) {
        Ok(index) => index,
        Err(e) => panic!("Failed to load index! {}", e),
    };

    // search the index
    let results = reload_index.search_approximate(vectors[*word_to_idx.get("king").unwrap()], 20);

    // visualize the results
    for (i, (results_idx, distance)) in results.iter().enumerate() {
        println!(
            "{i}. Word: {}. Distance: {}",
            idx_to_word.get(results_idx).unwrap(),
            distance.sqrt()
        )
    }
}
fn main() {
    let (wiki, mut word_to_idx, mut idx_to_word, test_embs) =
        utils::load_wiki_vector::<DIM>("wiki-news-300d-1M.vec");

    test_ivfflat(
        &wiki,
        &mut word_to_idx,
        &mut idx_to_word,
        10,
        3,
        5,
        &test_embs,
    );

    test_lsh(&wiki, &word_to_idx, &idx_to_word, 8, 100);
}
