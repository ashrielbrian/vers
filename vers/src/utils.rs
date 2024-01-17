use crate::indexes::base::Vector;
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
