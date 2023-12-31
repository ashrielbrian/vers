use itertools::Itertools;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use vers::indexes::base::{Index, Vector};
use vers::indexes::ivfflat::IVFFlatIndex;
use vers::indexes::lsh::ANNIndex;

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

fn load_wiki_vector<const N: usize>(
    file_path: &str,
) -> (
    Vec<Vector<N>>,
    HashMap<String, usize>,
    HashMap<usize, String>,
    Vec<(String, [f32; N])>,
) {
    // this function loads all the words from the wikipedia dataset, but deliberately excludes `queen`
    // so it can be added back manually later.
    let vector_file = fs::File::open(file_path).expect("Should be able to read file");
    let reader = std::io::BufReader::new(vector_file);

    let mut curr_idx: usize = 0;
    let mut all_vecs = Vec::new();
    let mut word_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx_to_word: HashMap<usize, String> = HashMap::new();
    let mut test_embs: Vec<(String, [f32; N])> = Vec::new();

    for wrapped_line in reader.lines().skip(1) {
        let line = wrapped_line.unwrap();

        let mut split_by_spaces = line.split_whitespace();
        let word = split_by_spaces.next().unwrap();
        let emb: [f32; N] = split_by_spaces
            .into_iter()
            .map(|d| d.parse::<f32>().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        if word == "queen" {
            println!("{} {}", word, curr_idx);
            test_embs.push((word.to_string(), emb));
            continue;
        }

        word_to_idx.insert(word.to_owned(), curr_idx);
        idx_to_word.insert(curr_idx, word.to_owned());

        curr_idx += 1;
        all_vecs.push(Vector(emb))
    }

    // verify does not contain `queen`
    match word_to_idx.get("queen") {
        Some(_) => println!("Found queen!"),
        None => println!("No queen, not to worry!"),
    }

    println!(
        "{} {} {}",
        all_vecs.len(),
        word_to_idx.len(),
        idx_to_word.len()
    );
    (all_vecs, word_to_idx, idx_to_word, test_embs)
}

fn build_index<const N: usize>(
    vector_data: &Vec<Vector<N>>,
    num_trees: usize,
    max_node_size: usize,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &HashMap<usize, String>,
) -> ANNIndex<N> {
    let vector_ids: Vec<usize> = (0..vector_data.len()).collect();

    let index = ANNIndex::build_index(num_trees, max_node_size, vector_data, &vector_ids);

    let benchmark_idxs: Vec<&usize> = vector_ids
        .choose_multiple(&mut rand::thread_rng(), 10)
        .collect();
    let mut benchmark_vectors = Vec::new();
    for idx in &benchmark_idxs {
        benchmark_vectors.push(vector_data[**idx]);
    }

    for (idx, vector) in benchmark_idxs.iter().zip(benchmark_vectors) {
        let results = index.search_approximate(vector, 10);
        let search_word = idx_to_word.get(idx).unwrap();

        println!("---> SEARCH WORD: {search_word}");
        for (i, (results_idx, distance)) in results.iter().enumerate() {
            println!(
                "{i}. Word: {}. Distance: {}",
                idx_to_word.get(results_idx).unwrap(),
                distance.sqrt()
            )
        }
    }

    println!("----------------------------------");
    let selected_words = ["king", "prince"];
    let selected_vector_ids: Vec<&usize> = selected_words
        .iter()
        .map(|w| word_to_idx.get(*w).unwrap())
        .collect();

    for (vec_id, search_word) in selected_vector_ids.iter().zip(selected_words) {
        let results = index.search_approximate(vector_data[**vec_id], 5);

        println!("---> SEARCH APPRX WORD: {search_word}");
        for (i, (results_idx, distance)) in results.iter().enumerate() {
            println!(
                "{i}. Word: {}. Distance: {}",
                idx_to_word.get(results_idx).unwrap(),
                distance.sqrt()
            )
        }

        // search exhaustively as the recall benchmark
        let exhaustive_candidates = search_exhaustive(&vector_data, &vector_data[**vec_id], 5);
        println!("---> SEARCH EXHAUSTIVELY WORD: {search_word}");
        for (i, (results_idx, distance)) in exhaustive_candidates.iter().enumerate() {
            println!(
                "{i}. Word: {}. Distance: {}",
                idx_to_word.get(results_idx).unwrap(),
                distance.sqrt()
            )
        }
    }

    index
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
        load_wiki_vector::<DIM>("wiki-news-300d-1M.vec");

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
