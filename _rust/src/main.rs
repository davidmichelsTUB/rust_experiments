// mod countvectorizer;
// mod pariter;
// use regex::Regex;
// use std::collections::HashSet;

// fn main() {
//     let corpus = vec![
//         "This is the first document.".to_string(),
//         "This document is the second document.".to_string(),
//         "And this is the third one.".to_string(),
//         "Is this the first document?".to_string(),
//     ];
//     let tokenizer = Regex::new(r"\b\w\w+\b").unwrap();
//     let stopwords: HashSet<String> = HashSet::new();
//     let (vocabulary, values, j_indices, indptr) =
//         countvectorizer::compute_count_vectorizer_fit(corpus, stopwords, tokenizer, 1);

//     println!("{:?}", vocabulary);
// }

fn main() {
    println!("Hello, world!");
}
