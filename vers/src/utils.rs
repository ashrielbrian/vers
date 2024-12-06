use crate::{ANNIndex, HNSWIndex, IVFFlatIndex, Index, Vector};
use itertools::Itertools;
use std::collections::HashMap;
use std::fs;
use std::io::BufRead;

pub fn load_wiki_vector<const N: usize>(
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

pub fn search_exhaustive<const N: usize>(
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

pub fn test_lsh<const N: usize>(
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

fn run_test<const N: usize, T: Index<N>>(
    index: &mut T,
    index_file_name: &str,
    vectors: &Vec<Vector<N>>,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &mut HashMap<usize, String>,
    test_embs: &Vec<(String, [f32; N])>,
) {
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
    println!("Words closest to `king`:");
    for (i, (results_idx, distance)) in results.iter().enumerate() {
        println!(
            "{i}. Word: {}. Distance: {}",
            idx_to_word.get(results_idx).unwrap(),
            distance.sqrt()
        )
    }
}

pub fn test_ivfflat<const N: usize>(
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
    let mut ivfflat =
        IVFFlatIndex::build_index(num_clusters, num_attempts, max_iterations, &vectors);

    run_test(
        &mut ivfflat,
        index_file_name,
        vectors,
        word_to_idx,
        idx_to_word,
        test_embs,
    );
}

pub fn test_hnsw<const N: usize>(
    vectors: &Vec<Vector<N>>,
    word_to_idx: &HashMap<String, usize>,
    idx_to_word: &mut HashMap<usize, String>,
    num_layers: usize,
    layer_multiplier: usize,
    ef_construction: usize,
    ef_search: usize,
    num_neighbours: usize,
    test_embs: &Vec<(String, [f32; N])>,
) {
    println!("HNSW Index:-----");
    let index_file_name = "hnsw.index";

    let mut hnsw = HNSWIndex::build_index(
        num_layers,
        layer_multiplier,
        ef_construction,
        ef_search,
        num_neighbours,
        vectors,
    );

    run_test(
        &mut hnsw,
        index_file_name,
        vectors,
        word_to_idx,
        idx_to_word,
        test_embs,
    );

    println!("Nodes in layers: {:?}", hnsw.get_num_nodes_in_layers())
}
