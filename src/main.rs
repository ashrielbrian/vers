use itertools::Itertools;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use vers::indexes::base::{Index, Vector};
use vers::indexes::ivfflat::IVFFlatIndex;
use vers::indexes::lsh::ANNIndex;

fn search_exhaustive<const N: usize>(
    vector_data: &Vec<Vector<N>>,
    query: &Vector<N>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    vector_data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.squared_euclidean(&query)))
        .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .take(top_k)
        .collect()
}

fn load_wiki_vector<const N: usize>(
    file_path: &str,
) -> (
    Vec<Vector<N>>,
    HashMap<String, usize>,
    HashMap<usize, String>,
    Vec<(String, [f32; N])>,
) {
    let vector_file = fs::File::open(file_path).expect("Should be able to read file");
    let reader = std::io::BufReader::new(vector_file);

    let mut curr_idx: usize = 0;
    let mut all_vecs = Vec::new();
    let mut word_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx_to_word: HashMap<usize, String> = HashMap::new();

    let mut embs: Vec<(String, [f32; N])> = Vec::new();

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

        if word == "queen"
            || word == "princess"
            || word == "kings"
            || word == "princes"
            || word == "King"
            || word == "monarch"
            || word == "Prince"
        {
            println!("{} {}", word, curr_idx);
            embs.push((word.to_string(), emb));
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
    (all_vecs, word_to_idx, idx_to_word, embs)
}

fn build_index<const N: usize>(
    vector_data: &Vec<Vector<N>>,
    num_trees: usize,
    max_node_size: usize,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &HashMap<usize, String>,
) -> IVFFlatIndex<N> {
    let vector_ids: Vec<usize> = (0..vector_data.len()).collect();

    // let index = ANNIndex::build_index(num_trees, max_node_size, vector_data, &vector_ids);
    let index: IVFFlatIndex<N> = IVFFlatIndex::build_index(5, 4, 5, vector_data);

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
fn main() {
    const DIM: usize = 300;
    let (wiki, mut word_to_idx, mut idx_to_word, embs) =
        load_wiki_vector::<DIM>("wiki-news-300d-1M.vec");

    let mut index = build_index(&wiki, 8, 6, &word_to_idx, &idx_to_word);

    // NOT THIS!!
    for ((word, emb), vec_id) in embs
        .into_iter()
        .zip([999987, 999988, 999989, 999990, 999991, 999992, 999993])
    {
        println!("Inserting {} {}", word, vec_id);
        idx_to_word.insert(vec_id, word.to_string());
        index.add(Vector(emb), vec_id);
    }

    println!("QUEENNNNNN ----------------------------------");
    let selected_words = ["king", "prince"];
    let selected_vector_ids: Vec<&usize> = selected_words
        .iter()
        .map(|w| word_to_idx.get(*w).unwrap())
        .collect();

    for (vec_id, search_word) in selected_vector_ids.iter().zip(selected_words) {
        let results = index.search_approximate(wiki[**vec_id], 5);

        println!("---> SEARCH APPRX WORD: {search_word}");
        for (i, (results_idx, distance)) in results.iter().enumerate() {
            println!(
                "{i}. Word: {}. Distance: {}",
                idx_to_word.get(results_idx).unwrap(),
                distance.sqrt()
            )
        }
    }

    match index.save_index("here.index") {
        Ok(_) => println!("Index saved successfully!"),
        Err(e) => eprintln!("Index save failed: {}", e),
    }

    let reload_index: IVFFlatIndex<DIM> = match IVFFlatIndex::load_index("here.index") {
        Ok(index) => index,
        Err(e) => panic!("Failed to load index! {}", e),
    };

    let results = reload_index.search_approximate(wiki[*word_to_idx.get("king").unwrap()], 20);
    for (i, (results_idx, distance)) in results.iter().enumerate() {
        println!(
            "{i}. Word: {}. Distance: {}",
            idx_to_word.get(results_idx).unwrap(),
            distance.sqrt()
        )
    }
}
