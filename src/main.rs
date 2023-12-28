use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use vers::indexes::ivfflat::IVFFlatIndex;
use vers::indexes::lsh::{ANNIndex, Vector};

fn load_wiki_vector<const N: usize>(
    file_path: &str,
) -> (
    Vec<Vector<N>>,
    HashMap<String, usize>,
    HashMap<usize, String>,
) {
    let vector_file = fs::File::open(file_path).expect("Should be able to read file");
    let reader = std::io::BufReader::new(vector_file);

    let mut curr_idx: usize = 0;
    let mut all_vecs = Vec::new();
    let mut word_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx_to_word: HashMap<usize, String> = HashMap::new();

    for wrapped_line in reader.lines().skip(1) {
        let line = wrapped_line.unwrap();

        let mut split_by_spaces = line.split_whitespace();
        let word = split_by_spaces.next().unwrap();
        word_to_idx.insert(word.to_owned(), curr_idx);
        idx_to_word.insert(curr_idx, word.to_owned());

        curr_idx += 1;

        let emb: [f32; N] = split_by_spaces
            .into_iter()
            .map(|d| d.parse::<f32>().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        all_vecs.push(Vector(emb))
    }

    (all_vecs, word_to_idx, idx_to_word)
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
                "{i}. Word: {} Distance: {}",
                idx_to_word.get(results_idx).unwrap(),
                distance.sqrt()
            )
        }
    }

    index
}
fn main() {
    const DIM: usize = 300;
    let (wiki, word_to_idx, idx_to_word) = load_wiki_vector::<DIM>("wiki-news-300d-1M.vec");

    let index = build_index(&wiki, 3, 25, &word_to_idx, &idx_to_word);

    let val = Vector([1.0, 2.0, 3.0]);
    println!("{:?}", val);
}
